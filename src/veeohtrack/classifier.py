"""The classifier — a priority ladder from track evidence to a user_type (§5.4).

This is Stage 4, the last pure-logic stage. For a person at their crossing
moment it checks evidence in a deliberate order, and the first confident answer
wins:

- **Rung 0 — track-quality gate.** Too short, an impossible speed (the
  ID-switch signature), or a fit too noisy to trust → `ambiguous`. Checked first
  so corrupt tracks never reach the real logic.
- **Rung 1 — association.** Has a bicycle or dog been moving *with* this person
  over the approach — sustained proximity in at least k frames *and* matched
  velocity, not a single-frame overlap? → cyclist / dog_walker. Association sits
  above speed because casual cyclists overlap runner speeds (field-confirmed,
  §6) and a person walking a bike is still a cyclist.
- **Rung 2 — speed bands.** No association → windowed ground speed: below
  `walk_max` → walker; above `run_min` → runner; in between → `ambiguous`; and
  above `runner_ceiling` with no bike → `ambiguous` (an e-scooter or rollerblader
  has no COCO class and would otherwise be silently mislabelled a runner).

Every rung has an escape to `ambiguous`, and each escape is recorded as an
`AmbiguousReason`. That is the accuracy story (§7): `ambiguous` is the system
being honest, not an error, and the reason says which kind of "don't know" it was.

Thresholds live in `ClassifierConfig`, not in code: `walk_max` / `run_min` /
the ceiling are still-open tunables (§10) set against real footage in Stage B,
and a label's meaning depends on which set produced it (hence `config_version`
on the crossing record, §5.7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .kinematics import WINDOW_SECONDS, estimate_speed, window
from .types import AmbiguousReason, Association, DetClass, Track, UserType


@dataclass(frozen=True)
class ClassifierConfig:
    """Tunable thresholds. Defaults are the provisional desk values (§5.4, §6)."""
    # Rung 0 — track quality
    min_track_frames: int = 10           # shorter tracks give too few points for a stable fit
    max_plausible_speed_mps: float = 12.0  # beyond any trail user; a jump this big is an ID switch
    max_fit_residual_m: float = 0.5      # foot-point jitter too large to trust the speed (§5.3)

    # Rung 1 — association
    association_window_seconds: float = 3.0  # the multi-second approach, not a snapshot (§5.4.1)
    association_min_frames: int = 3      # k: companion co-located in >= k approach frames
    association_max_distance_m: float = 2.0  # bike under rider / dog on a leash
    association_max_velocity_diff_mps: float = 0.75  # moving *with* the person, not passing

    # Rung 2 — speed bands
    speed_window_seconds: float = WINDOW_SECONDS
    walk_max_mps: float = 2.0
    run_min_mps: float = 2.5
    runner_ceiling_mps: float = 4.0      # no recreational runner sustains > 4 m/s over the zone


@dataclass(frozen=True)
class Classification:
    """The classifier's output — the class/kinematics half of a crossing record.

    Field names match `CrossingRecord` / `schema/crossing.sql` so assembly is a
    straight copy. Invariants (enforced by construction, mirrored as CHECKs in
    the DDL): `ambiguous_reason` is set iff `user_type` is ambiguous; a cyclist
    carries a bicycle association; a dog_walker carries a dog association.
    """
    user_type: UserType
    ambiguous_reason: AmbiguousReason | None = None
    association: Association = Association.NONE
    association_frame_count: int = 0
    speed_mps: float | None = None       # None if gated out before speed (Rung 0 length)
    speed_fit_residual: float | None = None


_COMPANION = {DetClass.BICYCLE: Association.BICYCLE, DetClass.DOG: Association.DOG}
_ASSOCIATED_TYPE = {Association.BICYCLE: UserType.CYCLIST, Association.DOG: UserType.DOG_WALKER}


def _ambiguous(reason: AmbiguousReason, speed=None, residual=None) -> Classification:
    return Classification(
        UserType.AMBIGUOUS, reason, speed_mps=speed, speed_fit_residual=residual
    )


def _association(
    person: Track,
    companions: Sequence[Track],
    crossing_t: float,
    cfg: ClassifierConfig,
) -> tuple[Association, int]:
    """Find the bicycle/dog moving with this person over the approach window.

    A companion qualifies if it is within `association_max_distance_m` of the
    person in at least `association_min_frames` shared frames of the window
    (intermittent detection is fine — that's the point of a window, §5.4.1) and
    its velocity over the window matches the person's. Of the qualifiers, the
    one present in the most frames wins.
    """
    w = cfg.association_window_seconds
    approach = window(person.observations, w, end_timestamp=crossing_t)
    if not approach:
        return Association.NONE, 0
    person_est = estimate_speed(person.observations, w, end_timestamp=crossing_t)

    best, best_frames = Association.NONE, 0
    for comp in companions:
        kind = _COMPANION.get(comp.cls)
        if kind is None:
            continue
        at_frame = {o.frame_index: o.ground_xy for o in comp.observations}
        frames = 0
        for o in approach:
            xy = at_frame.get(o.frame_index)
            if xy is None:
                continue
            dx, dy = xy[0] - o.ground_xy[0], xy[1] - o.ground_xy[1]
            if (dx * dx + dy * dy) ** 0.5 <= cfg.association_max_distance_m:
                frames += 1
        if frames < cfg.association_min_frames:
            continue

        comp_est = estimate_speed(comp.observations, w, end_timestamp=crossing_t)
        if person_est is None or comp_est is None:
            continue
        dvx = person_est.velocity[0] - comp_est.velocity[0]
        dvy = person_est.velocity[1] - comp_est.velocity[1]
        if (dvx * dvx + dvy * dvy) ** 0.5 > cfg.association_max_velocity_diff_mps:
            continue                     # close by, but passing rather than accompanying

        if frames > best_frames:
            best, best_frames = kind, frames
    return best, best_frames


def classify(
    person: Track,
    crossing_t: float,
    companions: Sequence[Track] = (),
    config: ClassifierConfig | None = None,
) -> Classification:
    """Label one person track at its crossing instant via the priority ladder.

    `companions` are the other grounded tracks alive around the crossing (the
    tracker's bicycle/dog tracks); non-bicycle/dog tracks are ignored.
    """
    cfg = config or ClassifierConfig()

    # Rung 0 — track quality, part 1: too short to fit at all.
    if person.length_frames < cfg.min_track_frames:
        return _ambiguous(AmbiguousReason.RUNG0_TRACK_QUALITY)

    est = estimate_speed(
        person.observations, cfg.speed_window_seconds, end_timestamp=crossing_t
    )
    speed = est.speed_mps if est else None
    residual = est.residual if est else None

    # Rung 0, part 2: an impossible speed (ID switch) or a fit too noisy to trust.
    if est is not None and (
        est.speed_mps > cfg.max_plausible_speed_mps or est.residual > cfg.max_fit_residual_m
    ):
        return _ambiguous(AmbiguousReason.RUNG0_TRACK_QUALITY, speed, residual)

    # Rung 1 — association beats speed.
    assoc, frames = _association(person, companions, crossing_t, cfg)
    if assoc is not Association.NONE:
        return Classification(
            _ASSOCIATED_TYPE[assoc],
            association=assoc,
            association_frame_count=frames,
            speed_mps=speed,
            speed_fit_residual=residual,
        )

    # Rung 2 — speed bands.
    if est is None:
        return _ambiguous(AmbiguousReason.NO_SIGNAL)
    if speed < cfg.walk_max_mps:
        return Classification(UserType.WALKER, speed_mps=speed, speed_fit_residual=residual)
    if speed > cfg.runner_ceiling_mps:
        return _ambiguous(AmbiguousReason.SPEED_CEILING, speed, residual)
    if speed > cfg.run_min_mps:
        return Classification(UserType.RUNNER, speed_mps=speed, speed_fit_residual=residual)
    return _ambiguous(AmbiguousReason.RUNG2_SPEED_BAND, speed, residual)
