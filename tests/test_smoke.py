"""Stage-0 smoke test: the package imports and the settled type model is wired.

Flow under test:  Detection → tracker → RawTrack → projection → Track.
Intentionally light — its job is to make `pytest` green and pin the boundary
shapes, not to test stage 1+ logic that does not exist yet.
"""
from veeohtrack import __version__
from veeohtrack.calibrate import GroundPlane
from veeohtrack.interfaces import Detector, FrameSource, Storage, Tracker
from veeohtrack.types import (
    Association,
    BBox,
    CrossingRecord,
    Detection,
    DetClass,
    Direction,
    Observation,
    RawObservation,
    RawTrack,
    Track,
    UserType,
)


def _det() -> Detection:
    """A bare, time-free detection — what the detector actually emits."""
    return Detection(DetClass.PERSON, BBox(0, 0, 2, 2), 0.9)


def _raw(frame_index: int, timestamp: float) -> RawObservation:
    """A detection stamped with the temporal context the tracker supplies."""
    return RawObservation(_det(), frame_index, timestamp)


def test_package_imports():
    assert __version__ == "0.0.0"


def test_foot_point_is_bottom_center():
    box = BBox(x1=10.0, y1=20.0, x2=30.0, y2=80.0)
    assert box.foot_point == (20.0, 80.0)
    assert box.center == (20.0, 50.0)


def test_enum_values_match_schema_vocab():
    # These strings are the source of truth shared with schema/crossing.sql.
    assert Direction.IN.value == "in"
    assert UserType.DOG_WALKER.value == "dog_walker"
    assert Association.NONE.value == "none"
    assert DetClass.BICYCLE.value == "bicycle"


def test_rawtrack_length_counts_frame_span_not_observation_count():
    track = RawTrack(
        track_id=1,
        cls=DetClass.PERSON,
        observations=[_raw(100, 0.0), _raw(104, 0.27)],
        gap_count=3,
    )
    # 100..104 inclusive = 5 frames spanned, even though only 2 observations exist.
    assert track.length_frames == 5


def test_projection_wraps_raw_observation_and_stores_ground_position():
    gp = GroundPlane.identity()
    raw = _raw(7, 0.5)
    obs = Observation(raw=raw, ground_xy=gp.project(raw.foot_point))
    # Temporal + pixel fields stay reachable; ground position is stored once.
    assert obs.frame_index == 7
    assert obs.timestamp == 0.5
    assert obs.detection is raw.detection         # content still reachable
    assert obs.foot_point == raw.foot_point       # pixel space
    assert obs.ground_xy == raw.foot_point        # identity plane → same coords


def test_grounded_track_carries_observations_and_quality_signals():
    obs = [
        Observation(_raw(10, 0.0), (0.0, 0.0)),
        Observation(_raw(12, 0.13), (1.0, 0.0)),
    ]
    track = Track(track_id=2, cls=DetClass.PERSON, observations=obs, instability_flag=True)
    assert track.length_frames == 3
    assert track.instability_flag is True


def test_groundplane_projection_applies_scale_and_divide():
    # Scale x by 0.5, y by 0.25, with a non-trivial perspective term.
    gp = GroundPlane(((0.5, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 2.0)))
    # (x/w, y/w) = (0.5*8 / 2, 0.25*8 / 2) = (2.0, 1.0)
    assert gp.project((8.0, 8.0)) == (2.0, 1.0)


def test_interface_protocols_are_runtime_checkable():
    for proto in (Detector, FrameSource, Storage, Tracker):
        assert not isinstance(object(), proto)


def test_crossing_record_binds_to_schema():
    rec = CrossingRecord(
        crossing_id="c-1",
        session_id="s-1",
        node_id="n-1",
        track_id=1,
        schema_version="crossing@v1",
        model_version="yolov8s@abc123",
        config_version="cfg@def456",
        rec_ld_ts="2026-08-26T16:00:00Z",
        direction=Direction.IN,
        committed_direction_confidence=0.9,
        user_type=UserType.WALKER,
        track_length_frames=42,
    )
    assert rec.association is Association.NONE
    assert rec.ambiguous_reason is None
    assert rec.track_instability_flag == 0
