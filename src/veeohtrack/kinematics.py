"""Speed and heading from a grounded track (§5.3).

This is Stage 1 — the first pure-logic stage. It consumes `Track` observations
(whose foot-points are already projected to ground-plane metres by `project.py`)
and answers two questions the classifier and counter need: **how fast** is this
subject moving, and **which way**.

The one load-bearing rule from the design: **never difference consecutive
frames.** Two adjacent positions are so close in time that detector/foot-point
noise dwarfs the real displacement — a steady 1.3 m/s walker can read anywhere
from 0.2 to 2.5 m/s frame-to-frame (even negative), straddling the walker/runner
boundary from noise alone. Instead we fit a straight line through a *trailing ~1-second window*
of positions; the slope of that line is the velocity. Averaging many points
cancels the noise and yields a stable number.

Two deliberate choices, both from §5.3:

- **A window, not the whole track.** The estimate should reflect the subject's
  pace *near the tripwire*, not an average blurred across pace changes earlier
  in the track.
- **A robust fit (Theil–Sen), default-on.** Foot-point jitter is not always
  zero-mean: a truncated box (occlusion) or a gait swing can bias a few points
  one way. Plain least-squares would let those outliers drag the slope; the
  median-of-slopes estimator shrugs them off. It costs nothing here, so it is
  the default rather than an upgrade.

Kinematics never sees the homography matrix — `project.py` already grounded the
points. This module is pure arithmetic on (time, ground-position) pairs: numpy
carries the vectorised fit, and there is no hardware dependency, so it runs and
unit-tests anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .types import Observation, Track

# Trailing window length for the fit. ~1 s balances enough points to average
# noise against staying local to the subject's current pace (§5.3).
WINDOW_SECONDS = 1.0

# Minimum points in the window for a trustworthy fit. Two points define a line
# but carry no redundancy (zero residual, no noise rejection); three is the
# floor at which the residual becomes meaningful. A window with fewer usable
# points is a short/interrupted track — the caller routes that to `ambiguous`
# via the Rung-0 gate (§5.4), so here we simply return None.
MIN_FIT_POINTS = 3

# Below this speed the heading is numerically meaningless (a near-stationary
# subject has no reliable direction), so we report heading as None.
_HEADING_EPSILON_MPS = 1e-6


@dataclass(frozen=True)
class SpeedEstimate:
    """The result of a windowed fit over one track's recent motion.

    `speed_mps` feeds the classifier's speed bands (§5.4, Rung 2); `heading`
    feeds direction (§5.5 / §5.2.1); `residual` is the fit-quality signal that
    doubles as a foot-point-jitter flag and is persisted as `speed_fit_residual`
    on the crossing record (§5.7). `n_points` and `duration_s` describe the
    window the estimate was actually computed over.
    """
    speed_mps: float
    velocity: tuple[float, float]        # (vx, vy) in metres/second, ground plane
    heading: tuple[float, float] | None  # unit vector, or None if ~stationary
    residual: float                      # RMS distance of points to the fitted line (m)
    n_points: int
    duration_s: float


def window(
    observations: Sequence[Observation],
    window_seconds: float = WINDOW_SECONDS,
    end_timestamp: float | None = None,
) -> list[Observation]:
    """Return the observations in the trailing window ending at `end_timestamp`.

    `end_timestamp` defaults to the last observation's timestamp — i.e. "the
    most recent ~1 second". Kept as a separate, testable function because the
    counter will want the same window ending at the crossing instant.
    """
    if not observations:
        return []
    end = observations[-1].timestamp if end_timestamp is None else end_timestamp
    start = end - window_seconds
    return [o for o in observations if start <= o.timestamp <= end]


def _theil_sen(ts: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Robust line fit: slope = median of all pairwise slopes; intercept =
    median of (y - slope*t). Resistant to a minority of outlier points.

    The pairwise slopes are the upper triangle of the (i, j) grid, computed as
    whole arrays rather than a Python double loop.
    """
    i, j = np.triu_indices(ts.size, k=1)
    dt = ts[j] - ts[i]
    nonzero = dt != 0                    # drop same-timestamp pairs (no basis)
    if not nonzero.any():                # all timestamps identical — no motion basis
        return 0.0, float(np.median(ys))
    slope = float(np.median((ys[j] - ys[i])[nonzero] / dt[nonzero]))
    intercept = float(np.median(ys - slope * ts))
    return slope, intercept


def estimate_speed(
    observations: Sequence[Observation],
    window_seconds: float = WINDOW_SECONDS,
    min_points: int = MIN_FIT_POINTS,
    end_timestamp: float | None = None,
) -> SpeedEstimate | None:
    """Fit ground velocity over the trailing window; None if too few points.

    Each ground coordinate is fit against time independently with Theil–Sen
    (x(t) and y(t)); the velocity is the pair of slopes and the speed is its
    magnitude. The residual is the RMS Euclidean distance from each observed
    point to the line it predicts — a compact track-quality signal.
    """
    win = window(observations, window_seconds, end_timestamp)
    if len(win) < min_points:
        return None

    ts = np.array([o.timestamp for o in win], dtype=float)
    xs = np.array([o.ground_xy[0] for o in win], dtype=float)
    ys = np.array([o.ground_xy[1] for o in win], dtype=float)

    vx, x0 = _theil_sen(ts, xs)
    vy, y0 = _theil_sen(ts, ys)

    speed = float(np.hypot(vx, vy))
    heading = (vx / speed, vy / speed) if speed > _HEADING_EPSILON_MPS else None

    # RMS distance of each point from the fitted trajectory.
    dx = xs - (x0 + vx * ts)
    dy = ys - (y0 + vy * ts)
    residual = float(np.sqrt(np.mean(dx * dx + dy * dy)))

    return SpeedEstimate(
        speed_mps=speed,
        velocity=(vx, vy),
        heading=heading,
        residual=residual,
        n_points=len(win),
        duration_s=float(ts[-1] - ts[0]),
    )


def estimate_track_speed(
    track: Track,
    window_seconds: float = WINDOW_SECONDS,
    min_points: int = MIN_FIT_POINTS,
    end_timestamp: float | None = None,
) -> SpeedEstimate | None:
    """Convenience wrapper: estimate speed for a whole `Track`'s observations."""
    return estimate_speed(track.observations, window_seconds, min_points, end_timestamp)
