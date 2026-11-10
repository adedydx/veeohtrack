"""Stage 1 tests — windowed speed/heading fit and the projection step.

All inputs are hand-made position sequences (no detector, no hardware), per the
§8 build order: a failure here is unambiguously in the logic. The headline
"done when" check is that a synthetic walker reads ~1.3 m/s and a jogger ~2.8.
"""
import math

from veeohtrack.calibrate import GroundPlane
from veeohtrack.kinematics import (
    WINDOW_SECONDS,
    estimate_speed,
    estimate_track_speed,
    window,
)
from veeohtrack.project import project_track
from veeohtrack.types import (
    BBox,
    Detection,
    DetClass,
    Observation,
    RawObservation,
    RawTrack,
    Track,
)

FPS = 15.0


def _obs(frame_index: int, t: float, x: float, y: float) -> Observation:
    """An observation already grounded at (x, y) metres."""
    det = Detection(DetClass.PERSON, BBox(x, y, x + 1.0, y + 1.0), 0.9)
    return Observation(raw=RawObservation(det, frame_index, t), ground_xy=(x, y))


def _straight_line(speed: float, heading=(1.0, 0.0), n=16, fps=FPS, start=0.0):
    """n observations moving at constant `speed` m/s along a unit `heading`."""
    hx, hy = heading
    out = []
    for i in range(n):
        t = start + i / fps
        d = speed * t
        out.append(_obs(i, t, d * hx, d * hy))
    return out


def test_walker_reads_about_1_3_mps():
    est = estimate_speed(_straight_line(1.3))
    assert est is not None
    assert math.isclose(est.speed_mps, 1.3, abs_tol=1e-6)


def test_jogger_reads_about_2_8_mps():
    est = estimate_speed(_straight_line(2.8))
    assert math.isclose(est.speed_mps, 2.8, abs_tol=1e-6)


def test_clean_line_has_negligible_residual():
    est = estimate_speed(_straight_line(2.0))
    assert est.residual < 1e-9


def test_heading_is_unit_vector_in_travel_direction():
    est = estimate_speed(_straight_line(1.5, heading=(0.0, -1.0)))
    assert est.heading is not None
    hx, hy = est.heading
    assert math.isclose(hx, 0.0, abs_tol=1e-6)
    assert math.isclose(hy, -1.0, abs_tol=1e-6)
    assert math.isclose(math.hypot(hx, hy), 1.0, abs_tol=1e-6)


def test_robust_fit_shrugs_off_a_truncated_footpoint_outlier():
    # A steady 1.3 m/s walker, but one frame's foot-point is truncated upward
    # by occlusion and jumps 1.5 m off-track. Theil–Sen should barely move.
    obs = _straight_line(1.3)
    bad = obs[8]
    obs[8] = _obs(bad.frame_index, bad.timestamp, bad.ground_xy[0], bad.ground_xy[1] + 1.5)

    est = estimate_speed(obs)
    assert math.isclose(est.speed_mps, 1.3, abs_tol=0.05)  # still essentially 1.3
    assert est.residual > 0.0                              # but the outlier shows up as residual


def test_window_tracks_recent_pace_not_the_whole_track():
    # First second slow (1.0 m/s), then speeds up to 3.0 m/s. A 1 s trailing
    # window ending at the last point should report ~3.0, not the blended mean.
    slow = _straight_line(1.0, n=15, start=0.0)
    x0 = slow[-1].ground_xy[0]
    t0 = slow[-1].timestamp
    fast = [
        _obs(15 + i, t0 + (i + 1) / FPS, x0 + 3.0 * ((i + 1) / FPS), 0.0)
        for i in range(15)
    ]
    est = estimate_speed(slow + fast, window_seconds=1.0)
    assert math.isclose(est.speed_mps, 3.0, abs_tol=0.05)


def test_window_helper_keeps_only_the_trailing_second():
    obs = _straight_line(1.0, n=30)  # 2 seconds at 15 fps
    win = window(obs, WINDOW_SECONDS)
    assert all(o.timestamp >= obs[-1].timestamp - WINDOW_SECONDS for o in win)
    assert win[-1] is obs[-1]


def test_too_few_points_returns_none():
    # A two-point stub with the default 3-point floor → not estimable here;
    # the caller routes such short tracks to `ambiguous` (Rung 0).
    assert estimate_speed(_straight_line(1.3, n=2)) is None
    assert estimate_speed([]) is None


def test_near_stationary_subject_has_no_heading():
    still = [_obs(i, i / FPS, 0.0, 0.0) for i in range(10)]
    est = estimate_speed(still)
    assert est is not None
    assert math.isclose(est.speed_mps, 0.0, abs_tol=1e-9)
    assert est.heading is None


def test_projection_then_speed_end_to_end():
    # RawTrack (pixel space) → project with identity plane → Track → speed.
    # Identity plane means pixel == metre, so a 1.3 "m/s" pixel track reads 1.3.
    raws = [
        RawObservation(Detection(DetClass.PERSON, BBox(x, 0, x + 1, 2), 0.9), i, i / FPS)
        for i, x in ((i, 1.3 * (i / FPS)) for i in range(16))
    ]
    raw_track = RawTrack(track_id=1, cls=DetClass.PERSON, observations=raws, gap_count=2)

    track = project_track(raw_track, GroundPlane.identity())
    assert isinstance(track, Track)
    assert track.gap_count == 2                       # quality signal carried through
    # Foot-point is bottom-center = (x+0.5, 2); x-slope is still 1.3.
    est = estimate_track_speed(track)
    assert math.isclose(est.speed_mps, 1.3, abs_tol=1e-6)
