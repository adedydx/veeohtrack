"""Stage 2 tests — two-line tripwire counting, direction, and swap pairing.

Hand-made grounded tracks only (no detector/hardware), per the §8 build order.
Geometry for these tests: the centerline runs along the x-axis from (0,0) to
(4,0) with a 1 m band, so L1 is y = -0.5 and L2 is y = +0.5. Moving in +y
(entering through L1) is IN; -y (entering through L2) is OUT. The wire spans
x ∈ [0, 4].
"""
import math

from veeohtrack.calibrate import GroundPlane
from veeohtrack.counting import Tripwire, count_crossings
from veeohtrack.types import (
    BBox,
    Detection,
    DetClass,
    Direction,
    Observation,
    RawObservation,
    Track,
)

FPS = 15.0
WIRE = Tripwire(a=(0.0, 0.0), b=(4.0, 0.0), band_width_m=1.0, positive_direction=Direction.IN)


def _obs(frame_index: int, x: float, y: float, cls=DetClass.PERSON) -> Observation:
    det = Detection(cls, BBox(x, y, x + 1.0, y + 1.0), 0.9)
    return Observation(RawObservation(det, frame_index, frame_index / FPS), (x, y))


def _track(ys, x=2.0, track_id=1, cls=DetClass.PERSON, start_frame=0):
    """A track at fixed x walking through the given y positions, one per frame."""
    obs = [_obs(start_frame + i, x, y, cls) for i, y in enumerate(ys)]
    return Track(track_id=track_id, cls=cls, observations=obs)


def _line(y0, y1, n):
    return [y0 + (y1 - y0) * i / (n - 1) for i in range(n)]


INBOUND = _line(-1.5, 1.5, 16)     # 0.2 m per frame, -y side to +y side
OUTBOUND = _line(1.5, -1.5, 16)


def test_inbound_traversal_counts_once_in():
    [c] = count_crossings([_track(INBOUND)], WIRE)
    assert c.direction is Direction.IN
    assert c.track_id == 1
    assert c.swap_resolved is False
    assert math.isclose(c.ground_xy[1], -0.5, abs_tol=1e-9)   # entered through L1


def test_outbound_traversal_counts_out():
    [c] = count_crossings([_track(OUTBOUND)], WIRE)
    assert c.direction is Direction.OUT
    assert math.isclose(c.ground_xy[1], 0.5, abs_tol=1e-9)    # entered through L2


def test_clean_approach_has_high_confidence():
    [c] = count_crossings([_track(INBOUND)], WIRE)
    assert c.committed_direction_confidence > 0.9


def test_timestamp_is_interpolated_band_entry():
    # Enters L1 (y=-0.5) halfway between frame 2 (y=-0.6) and frame 3 (y=-0.4).
    ys = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.6, 0.8]
    [c] = count_crossings([_track(ys)], WIRE)
    assert math.isclose(c.timestamp, 2.5 / FPS, rel_tol=1e-9)
    assert c.frame_index == 3                                  # first frame inside


# --- The band as jitter guard ----------------------------------------------

def test_jitter_on_one_line_does_not_count():
    ys = [-0.8, -0.6, -0.45, -0.55, -0.45, -0.6, -0.8]        # wobbling across L1
    assert count_crossings([_track(ys)], WIRE) == []


def test_step_in_and_retreat_does_not_count():
    ys = [-1.0, -0.6, -0.2, 0.2, -0.2, -0.6, -1.0]            # deep into band, back out L1
    assert count_crossings([_track(ys)], WIRE) == []


def test_track_that_never_reaches_the_band_returns_nothing():
    assert count_crossings([_track(_line(-2.0, -0.7, 10))], WIRE) == []


def test_crossing_past_the_wire_end_is_not_counted():
    assert count_crossings([_track(INBOUND, x=10.0)], WIRE) == []


def test_crossing_back_counts_twice():
    ys = INBOUND + OUTBOUND[1:]                                # over, then back again
    crossings = count_crossings([_track(ys)], WIRE)
    assert [c.direction for c in crossings] == [Direction.IN, Direction.OUT]


# --- Fixed bugs --------------------------------------------------------------

def test_point_exactly_on_a_line_is_still_counted():
    # Regression: a foot-point landing exactly on a line used to drop the crossing.
    ys = [-1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0]
    [c] = count_crossings([_track(ys)], WIRE)
    assert c.direction is Direction.IN


def test_only_person_tracks_are_counted():
    # Regression: a bicycle track used to be counted, so a cyclist (person +
    # bike) counted twice.
    person = _track(INBOUND, track_id=1)
    bike = _track(INBOUND, track_id=2, cls=DetClass.BICYCLE)
    dog = _track(INBOUND, track_id=3, cls=DetClass.DOG)
    crossings = count_crossings([person, bike, dog], WIRE)
    assert [c.track_id for c in crossings] == [1]


def test_id_swap_inside_the_band_still_counts_both_people():
    # Regression: A walks +y and B walks -y; they meet mid-band and the tracker
    # swaps their identities. Each track then enters and leaves through the same
    # line. Pairing recovers both crossings, each in its own approach direction.
    a_body = _line(-1.5, 1.5, 16)
    b_body = _line(1.5, -1.5, 16)
    swap = 8                                                   # bodies meet near y=0
    track_a = _track(a_body[:swap] + b_body[swap:], track_id=1)
    track_b = _track(b_body[:swap] + a_body[swap:], track_id=2)
    crossings = count_crossings([track_a, track_b], WIRE)
    assert {(c.track_id, c.direction) for c in crossings} == {
        (1, Direction.IN),
        (2, Direction.OUT),
    }
    assert all(c.swap_resolved for c in crossings)


def test_id_swap_just_past_the_band_keeps_totals():
    # The bodies meet beyond L2 (y≈0.8). Track A traverses in, then carries B's
    # tail back out; track B never enters the band. Totals stay 1 in + 1 out.
    a_body = [-1.2 + 0.2 * f for f in range(21)]
    b_body = [2.8 - 0.2 * f for f in range(21)]
    track_a = _track(a_body[:11] + b_body[11:], track_id=1)
    track_b = _track(b_body[:11] + a_body[11:], track_id=2)
    crossings = count_crossings([track_a, track_b], WIRE)
    assert sorted(c.direction.value for c in crossings) == ["in", "out"]


def test_simultaneous_retreats_far_apart_are_not_paired():
    # Two people each step in and back out, at the same time but 3 m apart:
    # no meeting, so no swap — neither is counted.
    a = _track([-1.0, -0.6, -0.2, -0.6, -1.0], x=0.5, track_id=1)
    b = _track([1.0, 0.6, 0.2, 0.6, 1.0], x=3.5, track_id=2)
    assert count_crossings([a, b], WIRE) == []


def test_fast_mover_jumping_the_whole_band_in_one_step_counts():
    [c] = count_crossings([_track([-1.6, -1.2, -0.8, 1.0, 1.4])], WIRE)
    assert c.direction is Direction.IN


def test_from_pixels_projects_the_centerline():
    wire = Tripwire.from_pixels((0.0, 0.0), (4.0, 0.0), GroundPlane.identity(), band_width_m=1.0)
    assert (wire.a, wire.b) == ((0.0, 0.0), (4.0, 0.0))
    assert len(count_crossings([_track(INBOUND)], wire)) == 1
