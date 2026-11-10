"""Stage 4 tests — the classifier's priority ladder on hand-made grounded tracks.

Checks each rung and each `ambiguous` escape, and that every output satisfies
the invariants `schema/crossing.sql` enforces with CHECK constraints.
"""
import math

from veeohtrack.classifier import Classification, ClassifierConfig, classify
from veeohtrack.types import (
    AmbiguousReason,
    Association,
    BBox,
    Detection,
    DetClass,
    Observation,
    RawObservation,
    Track,
    UserType,
)

FPS = 15.0


def _track(speed, n=45, cls=DetClass.PERSON, track_id=1, y=0.0, frames=None, jitter=None):
    """Grounded track moving along +x at `speed` m/s; optional explicit frames
    (to make gaps) and a per-frame y jitter function (to make a noisy fit)."""
    frames = list(range(n)) if frames is None else frames
    obs = []
    for f in frames:
        t = f / FPS
        x, yy = speed * t, y + (jitter(f) if jitter else 0.0)
        det = Detection(cls, BBox(x, yy, x + 1, yy + 1), 0.9)
        obs.append(Observation(RawObservation(det, f, t), (x, yy)))
    return Track(track_id=track_id, cls=cls, observations=obs)


def _crossing_t(track):
    return track.observations[-1].timestamp


def _assert_schema_invariants(c: Classification):
    assert (c.user_type is UserType.AMBIGUOUS) == (c.ambiguous_reason is not None)
    if c.user_type is UserType.CYCLIST:
        assert c.association is Association.BICYCLE
    if c.user_type is UserType.DOG_WALKER:
        assert c.association is Association.DOG


def _classify(person, companions=()):
    c = classify(person, _crossing_t(person), companions)
    _assert_schema_invariants(c)
    return c


# --- Rung 2: speed bands ----------------------------------------------------

def test_walker():
    c = _classify(_track(1.3))
    assert c.user_type is UserType.WALKER
    assert math.isclose(c.speed_mps, 1.3, abs_tol=1e-6)


def test_runner():
    assert _classify(_track(3.0)).user_type is UserType.RUNNER


def test_borderline_pace_is_ambiguous_speed_band():
    c = _classify(_track(2.2))           # between walk_max 2.0 and run_min 2.5
    assert c.user_type is UserType.AMBIGUOUS
    assert c.ambiguous_reason is AmbiguousReason.RUNG2_SPEED_BAND
    assert c.speed_mps is not None       # speed still logged for tuning


def test_fast_bare_person_hits_the_runner_ceiling():
    c = _classify(_track(5.0))           # e-scooter / rollerblader: no bike detected
    assert c.ambiguous_reason is AmbiguousReason.SPEED_CEILING


# --- Rung 0: track quality --------------------------------------------------

def test_short_track_is_gated_before_speed():
    c = _classify(_track(1.3, n=5))
    assert c.ambiguous_reason is AmbiguousReason.RUNG0_TRACK_QUALITY
    assert c.speed_mps is None           # gated out pre-speed → NULL in the record


def test_impossible_speed_is_an_id_switch_signature():
    c = _classify(_track(20.0))
    assert c.ambiguous_reason is AmbiguousReason.RUNG0_TRACK_QUALITY


def test_noisy_fit_is_gated_on_residual():
    zigzag = lambda f: 1.5 if f % 2 else -1.5    # foot-point bouncing ±1.5 m
    c = _classify(_track(1.3, jitter=zigzag))
    assert c.ambiguous_reason is AmbiguousReason.RUNG0_TRACK_QUALITY
    assert c.speed_fit_residual > 0.5


def test_long_but_sparse_track_has_no_signal():
    # Spans 41 frames (passes the length gate) but only one point lands in the
    # final 1 s speed window, so there is no usable speed and no companion.
    c = _classify(_track(1.3, frames=[0, 20, 40]))
    assert c.ambiguous_reason is AmbiguousReason.NO_SIGNAL


# --- Rung 1: association ----------------------------------------------------

def test_person_with_bike_is_a_cyclist():
    person = _track(3.2)
    bike = _track(3.2, cls=DetClass.BICYCLE, track_id=2, y=0.3)
    c = _classify(person, [bike])
    assert c.user_type is UserType.CYCLIST
    assert c.association is Association.BICYCLE
    assert c.association_frame_count >= 3


def test_walking_a_bike_is_still_a_cyclist():
    # Walking pace, but a bike moves with them: association beats speed.
    person = _track(1.3)
    bike = _track(1.3, cls=DetClass.BICYCLE, track_id=2, y=0.4)
    assert _classify(person, [bike]).user_type is UserType.CYCLIST


def test_fast_cyclist_is_rescued_from_the_ceiling_by_association():
    person = _track(6.0)
    bike = _track(6.0, cls=DetClass.BICYCLE, track_id=2, y=0.3)
    assert _classify(person, [bike]).user_type is UserType.CYCLIST


def test_person_with_dog_is_a_dog_walker():
    person = _track(1.3)
    dog = _track(1.3, cls=DetClass.DOG, track_id=3, y=1.0)
    c = _classify(person, [dog])
    assert c.user_type is UserType.DOG_WALKER
    assert c.association is Association.DOG


def test_intermittent_bike_in_k_frames_still_associates():
    # Bike only detected in a handful of approach frames — the window forgives it.
    person = _track(3.2)
    bike = _track(3.2, cls=DetClass.BICYCLE, track_id=2, y=0.3, frames=[20, 26, 32, 38, 44])
    assert _classify(person, [bike]).user_type is UserType.CYCLIST


def test_bike_in_fewer_than_k_frames_does_not_associate():
    person = _track(3.0)
    bike = _track(3.0, cls=DetClass.BICYCLE, track_id=2, y=0.3, frames=[42, 44])
    c = _classify(person, [bike])
    assert c.user_type is UserType.RUNNER
    assert c.association is Association.NONE


def test_bike_passing_the_other_way_does_not_associate():
    # A cyclist going the opposite direction is briefly close, but its velocity
    # doesn't match — passing, not accompanying.
    person = _track(3.0)
    bike_obs = []
    for f in range(45):
        t = f / FPS
        x = 3.0 * (44 / FPS) - 3.0 * (t - 44 / FPS)   # meets the person at the end
        det = Detection(DetClass.BICYCLE, BBox(x, 0.3, x + 1, 1.3), 0.9)
        bike_obs.append(Observation(RawObservation(det, f, t), (x, 0.3)))
    bike = Track(track_id=2, cls=DetClass.BICYCLE, observations=bike_obs)
    assert _classify(person, [bike]).user_type is UserType.RUNNER


def test_far_away_bike_does_not_associate():
    person = _track(3.0)
    bike = _track(3.0, cls=DetClass.BICYCLE, track_id=2, y=10.0)   # 10 m across
    assert _classify(person, [bike]).user_type is UserType.RUNNER


def test_rung0_wins_over_association():
    # A corrupt (too short) track never reaches the association rung.
    person = _track(3.2, n=5)
    bike = _track(3.2, n=5, cls=DetClass.BICYCLE, track_id=2, y=0.3)
    c = _classify(person, [bike])
    assert c.ambiguous_reason is AmbiguousReason.RUNG0_TRACK_QUALITY
    assert c.association is Association.NONE


def test_thresholds_come_from_config():
    cfg = ClassifierConfig(walk_max_mps=1.0, run_min_mps=1.2)
    person = _track(1.3)
    c = classify(person, _crossing_t(person), config=cfg)
    assert c.user_type is UserType.RUNNER   # same pace, different threshold set
