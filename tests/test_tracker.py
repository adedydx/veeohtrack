"""Stage 3 tests — ByteTrack-style association on synthetic detection streams.

Hand-made detection streams only (no detector/hardware), per the §8 build order.
The checks mirror the stage's "done when": stable IDs across frames, brief
low-confidence gaps recovered, expired IDs emitted, and each track carrying the
quality signals (length, gap_count, instability_flag) Stage 4 and V3 consume.
"""
import numpy as np

from veeohtrack.tracker import ByteTracker
from veeohtrack.types import BBox, Detection, DetClass, Frame

FPS = 15.0


def _frame(i: int) -> Frame:
    return Frame(index=i, timestamp=i / FPS, image=np.zeros((1, 1), dtype=np.uint8))


def _person(x: float, conf: float = 0.9, w: float = 20.0, h: float = 40.0) -> Detection:
    """A person box at left-edge x (fixed size), moving along the trail."""
    return Detection(DetClass.PERSON, BBox(x, 100.0, x + w, 100.0 + h), conf)


def _run(tracker, streams):
    """Feed one detection list per frame; return the active RawTracks per frame."""
    return [tracker.update(dets, _frame(i)) for i, dets in enumerate(streams)]


def test_moving_object_keeps_one_stable_id():
    tracker = ByteTracker()
    streams = [[_person(100 + 2 * i)] for i in range(6)]
    per_frame = _run(tracker, streams)
    ids = {tracks[0].track_id for tracks in per_frame}
    assert ids == {1}                      # same ID the whole way
    assert all(len(tracks) == 1 for tracks in per_frame)


def test_two_people_get_distinct_ids():
    tracker = ByteTracker()
    streams = [[_person(100 + 2 * i), _person(400 - 2 * i)] for i in range(5)]
    per_frame = _run(tracker, streams)
    last = {t.track_id for t in per_frame[-1]}
    assert len(last) == 2                   # two persistent identities


def test_clean_track_has_no_gaps_and_is_stable():
    tracker = ByteTracker()
    _run(tracker, [[_person(100 + 2 * i)] for i in range(5)])
    track = tracker.finish()[0]
    assert track.gap_count == 0
    assert track.instability_flag is False
    assert track.length_frames == 5        # frames 0..4 inclusive


def test_low_confidence_detection_rescues_the_track():
    tracker = ByteTracker()
    # Three clean high-conf frames, then a low-confidence (0.2) detection where
    # the track is predicted to be. Pass 2 should still claim it under the same ID.
    streams = [
        [_person(100)],
        [_person(102)],
        [_person(104)],
        [_person(106, conf=0.2)],          # low confidence
    ]
    per_frame = _run(tracker, streams)
    assert per_frame[-1][0].track_id == 1  # same ID, rescued by the second pass
    assert per_frame[-1][0].instability_flag is True


def test_brief_occlusion_is_bridged_under_the_same_id():
    tracker = ByteTracker()
    streams = [
        [_person(100)],
        [_person(102)],
        [_person(104)],
        [],                                 # dropped/occluded frame — no detection
        [_person(108)],                     # reappears where prediction expects it
    ]
    per_frame = _run(tracker, streams)
    assert per_frame[2] and per_frame[2][0].track_id == 1
    assert per_frame[3] == []               # nothing active during the gap
    recovered = per_frame[4][0]
    assert recovered.track_id == 1          # same ID, not a new one
    assert recovered.gap_count == 1         # one frame was bridged
    assert recovered.instability_flag is True
    assert recovered.length_frames == 5     # span 0..4 includes the gap frame


def test_track_expires_after_the_grace_period():
    tracker = ByteTracker(track_buffer=3)
    _run(tracker, [[_person(100)], [_person(102)]])
    # Four empty frames: exceeds track_buffer=3, so the track must expire.
    tail = _run(tracker, [[], [], [], []])
    assert all(active == [] for active in tail)     # nothing active anymore
    finished = tracker.drain_finished()
    assert len(finished) == 1 and finished[0].track_id == 1
    assert tracker.drain_finished() == []           # each handed off once


def test_lone_low_confidence_detection_creates_no_track():
    tracker = ByteTracker()
    active = tracker.update([_person(100, conf=0.2)], _frame(0))
    assert active == []                     # new tracks only from high confidence
    assert tracker.finish() == []


def test_tracks_never_absorb_another_class():
    # Regression: the rider's box is missed in the frame the bike is first
    # detected. The person track used to swallow the bike box, then fragment
    # into a new ID. Now the bike starts its own track and the person keeps one.
    tracker = ByteTracker()
    for i in range(6):
        x = 100 + 2 * i
        person = Detection(DetClass.PERSON, BBox(x, 60, x + 20, 140), 0.9)
        bike = Detection(DetClass.BICYCLE, BBox(x - 5, 95, x + 25, 145), 0.9)
        dets = [person] if i < 3 else ([bike] if i == 3 else [person, bike])
        tracker.update(dets, _frame(i))
    tracks = tracker.finish()
    people = [t for t in tracks if t.cls is DetClass.PERSON]
    bikes = [t for t in tracks if t.cls is DetClass.BICYCLE]
    assert len(people) == 1 and len(bikes) == 1
    for t in tracks:
        assert all(o.detection.cls is t.cls for o in t.observations)
    assert people[0].gap_count == 1                 # the missed rider frame was bridged


def test_sub_floor_detection_is_dropped_entirely():
    tracker = ByteTracker()
    active = tracker.update([_person(100, conf=0.05)], _frame(0))   # below LOW_THRESH
    assert active == []
