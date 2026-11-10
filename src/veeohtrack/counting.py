"""Two-line tripwire counting — person crossings with a committed direction (§5.5).

This is Stage 2. It consumes grounded `Track`s (from `project.py`) and answers
the Tier-1/Tier-2 questions (§7): *did a person cross*, and *which way*. It does
*not* classify them — user_type is the classifier's job (Stage 4); a `Crossing`
here is the half of the record that counting owns.

The tripwire is a **band** between two parallel lines, L1 and L2, either side of
a centerline. A crossing is a *traversal*: entering the band through one line
and leaving through the other. Design commitments:

- **Ground-plane metres, not pixels.** Kinematics already grounds every
  observation, and a straight trail line stays straight there (perspective
  bends it in pixel space). The site config's pixel centerline (§5.5) is
  projected once via `Tripwire.from_pixels`; the band width is in metres.

- **The band is the jitter guard.** Foot-point jitter on one line can't fake a
  traversal of the whole band, and someone who steps in and turns back leaves
  through the line they entered. Neither counts.

- **Direction comes from the line entered first.** That entry happens on the
  approach, before two converging subjects meet inside the band, so it is the
  approach-history commit of §5.2.1 option 2 made geometric.

- **ID swaps inside the band are paired back up.** When two people pass in
  opposite directions and the tracker swaps their identities mid-band, each
  resulting track enters and leaves through the *same* line — indistinguishable,
  on its own, from a step-in-and-retreat. But a swap leaves a signature no
  genuine retreat does: two such tracks, entering from *opposite* sides, in the
  band at the same time, meeting at the same spot. Those are paired and both
  counted, each in the direction of its own (pre-swap) approach, and flagged
  `swap_resolved`. Totals and directional split survive the swap (§5.2.1, §7).

- **People only.** A cyclist is a person track plus a bicycle track and a
  dog-walker a person plus a dog; counting every class would count each twice.
  Bicycle/dog tracks matter only as association evidence for the classifier.

Every traversal of a track counts — someone who crosses and later crosses back
is two crossings. Repeat counting from jitter is prevented by the band, not by
one-count-per-track-ID.

Swap partners must be in the same `count_crossings` call: the pipeline should
batch finished tracks over a window wider than the tracker's grace period.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .kinematics import WINDOW_SECONDS, window
from .types import Direction, DetClass, Observation, Track

# Distance between L1 and L2. Wide enough that foot-point jitter can't span it,
# narrow enough that a fast cyclist still lands a frame or two inside.
# Provisional — a config-tunable like the tripwire position itself (§10).
BAND_WIDTH_M = 1.0

# Two retreating tracks are treated as one swapped pair only if they came this
# close to each other while both were in the band (a swap needs a meeting).
SWAP_MAX_DISTANCE_M = 1.0

# Window for committed_direction_confidence — the ~1 s before band entry.
APPROACH_WINDOW_SECONDS = WINDOW_SECONDS


@dataclass(frozen=True)
class Tripwire:
    """A band across the trail, in ground-plane metres.

    `a`→`b` is the centerline; L1 and L2 sit `band_width_m / 2` either side of
    it along its normal, L1 on the −normal side. A subject entering through L1
    (moving toward +normal) is counted as `positive_direction`; through L2,
    the opposite. Which physical heading is "in" is a siting decision.
    """
    a: tuple[float, float]
    b: tuple[float, float]
    band_width_m: float = BAND_WIDTH_M
    positive_direction: Direction = Direction.IN

    def __post_init__(self):
        if self.a == self.b:
            raise ValueError("degenerate tripwire (endpoints coincide)")
        if self.band_width_m <= 0:
            raise ValueError("band_width_m must be positive")

    @property
    def _normal_unit(self) -> tuple[float, float]:
        dx, dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        length = math.hypot(dx, dy)
        return -dy / length, dx / length

    def offset(self, p: tuple[float, float]) -> float:
        """Signed distance of a point from the centerline (+ toward L2)."""
        nx, ny = self._normal_unit
        return (p[0] - self.a[0]) * nx + (p[1] - self.a[1]) * ny

    def zone(self, p: tuple[float, float]) -> int:
        """-1 beyond L1, +1 beyond L2, 0 inside the band. Half-open, so a point
        exactly on a line still belongs to exactly one zone."""
        o = self.offset(p)
        half = self.band_width_m / 2.0
        if o < -half:
            return -1
        if o >= half:
            return 1
        return 0

    def along(self, p: tuple[float, float]) -> float:
        """Position of p's projection along the centerline, 0 at `a`, 1 at `b`."""
        dx, dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        return ((p[0] - self.a[0]) * dx + (p[1] - self.a[1]) * dy) / (dx * dx + dy * dy)

    def direction_for_entry(self, entry_side: int) -> Direction:
        """Entering from beyond L1 (side -1) is the positive direction."""
        if entry_side == -1:
            return self.positive_direction
        return Direction.OUT if self.positive_direction is Direction.IN else Direction.IN

    @classmethod
    def from_pixels(
        cls, a_px, b_px, ground, band_width_m=BAND_WIDTH_M, positive_direction=Direction.IN
    ) -> "Tripwire":
        """Build from the site config's two pixel centerline endpoints,
        projecting both through the homography once (§5.5)."""
        return cls(ground.project(a_px), ground.project(b_px), band_width_m, positive_direction)


@dataclass(frozen=True)
class Crossing:
    """The counter's output — the count + direction half of a crossing record.

    Classification (user_type, association, speed) is attached later by the
    classifier; identity/provenance by the assembly step before storage (§5.7).
    """
    track_id: int
    cls: DetClass
    frame_index: int                       # first frame inside the band after entry
    timestamp: float                       # band-entry instant, seconds (session clock)
    ground_xy: tuple[float, float]         # where it entered the band (m)
    direction: Direction
    committed_direction_confidence: float  # 0..1, cleanliness of the pre-entry heading
    track_length_frames: int
    swap_resolved: bool = False            # counted by pairing an in-band ID swap


@dataclass(frozen=True)
class _BandVisit:
    """One stretch of a track from outside the band, through it, to outside."""
    entry_side: int                        # -1 / +1: the side it came from
    exit_side: int                         # -1 / +1: the side it left to
    t_entry: float
    t_exit: float
    entry_point: tuple[float, float]
    entry_frame_index: int

    @property
    def is_traversal(self) -> bool:
        return self.entry_side != self.exit_side


def _boundary_crossing(tripwire: Tripwire, p: Observation, q: Observation, side: int):
    """Interpolate where the step p→q crosses the line on `side` (L1 = -1,
    L2 = +1). Returns (time, point)."""
    c = side * tripwire.band_width_m / 2.0
    op, oq = tripwire.offset(p.ground_xy), tripwire.offset(q.ground_xy)
    f = (c - op) / (oq - op)
    point = (
        p.ground_xy[0] + f * (q.ground_xy[0] - p.ground_xy[0]),
        p.ground_xy[1] + f * (q.ground_xy[1] - p.ground_xy[1]),
    )
    return p.timestamp + f * (q.timestamp - p.timestamp), point


def _band_visits(track: Track, tripwire: Tripwire) -> list[_BandVisit]:
    """Split a track into outside→band→outside visits within the wire's extent.

    Observations inside the band are skipped; each time the track reaches an
    outside zone after leaving one, the stretch in between is a visit. A
    single step may jump the whole band (fast mover or a dropped frame).
    """
    obs = track.observations
    visits: list[_BandVisit] = []
    last_side, last_k = None, None
    for k, o in enumerate(obs):
        z = tripwire.zone(o.ground_xy)
        if z == 0:
            continue
        if last_side is not None and not (z == last_side and k == last_k + 1):
            t_in, p_in = _boundary_crossing(tripwire, obs[last_k], obs[last_k + 1], last_side)
            t_out, p_out = _boundary_crossing(tripwire, obs[k - 1], obs[k], z)
            if 0.0 <= tripwire.along(p_in) <= 1.0 and 0.0 <= tripwire.along(p_out) <= 1.0:
                visits.append(_BandVisit(
                    entry_side=last_side,
                    exit_side=z,
                    t_entry=t_in,
                    t_exit=t_out,
                    entry_point=p_in,
                    entry_frame_index=obs[last_k + 1].frame_index,
                ))
        last_side, last_k = z, k
    return visits


def _approach_coherence(track: Track, tripwire: Tripwire, t_entry: float) -> float:
    """|net normal displacement| / total |normal displacement| over the window
    before entry: 1.0 if every step moved the same way, lower if it jittered."""
    approach = window(track.observations, APPROACH_WINDOW_SECONDS, end_timestamp=t_entry)
    steps = [
        tripwire.offset(approach[k + 1].ground_xy) - tripwire.offset(approach[k].ground_xy)
        for k in range(len(approach) - 1)
    ]
    total = sum(abs(s) for s in steps)
    return abs(sum(steps)) / total if total > 0 else 0.0


def _meeting_distance(a: Track, b: Track, t0: float, t1: float) -> float:
    """Closest the two tracks came on shared frames within [t0, t1]."""
    b_at = {o.frame_index: o.ground_xy for o in b.observations if t0 <= o.timestamp <= t1}
    best = math.inf
    for o in a.observations:
        xy = b_at.get(o.frame_index)
        if xy is not None:
            best = min(best, math.hypot(o.ground_xy[0] - xy[0], o.ground_xy[1] - xy[1]))
    return best


def _pair_swaps(retreats, max_distance: float):
    """Pair retreats that entered from opposite sides, overlapped in the band,
    and met — the signature of an in-band ID swap. Closest meetings first."""
    candidates = []
    for i, (ta, va) in enumerate(retreats):
        for j in range(i + 1, len(retreats)):
            tb, vb = retreats[j]
            if ta.track_id == tb.track_id or va.entry_side == vb.entry_side:
                continue
            t0, t1 = max(va.t_entry, vb.t_entry), min(va.t_exit, vb.t_exit)
            if t0 > t1:
                continue
            d = _meeting_distance(ta, tb, t0, t1)
            if d <= max_distance:
                candidates.append((d, i, j))
    used: set[int] = set()
    pairs = []
    for _, i, j in sorted(candidates):
        if i not in used and j not in used:
            used.update((i, j))
            pairs.append((retreats[i], retreats[j]))
    return pairs


def _crossing(track: Track, visit: _BandVisit, tripwire: Tripwire, swap: bool) -> Crossing:
    return Crossing(
        track_id=track.track_id,
        cls=track.cls,
        frame_index=visit.entry_frame_index,
        timestamp=visit.t_entry,
        ground_xy=visit.entry_point,
        direction=tripwire.direction_for_entry(visit.entry_side),
        committed_direction_confidence=_approach_coherence(track, tripwire, visit.t_entry),
        track_length_frames=track.length_frames,
        swap_resolved=swap,
    )


def count_crossings(
    tracks: Sequence[Track],
    tripwire: Tripwire,
    swap_max_distance_m: float = SWAP_MAX_DISTANCE_M,
) -> list[Crossing]:
    """Count person crossings across a batch of finished tracks, by time.

    Each traversal of the band is one crossing. Retreats (in and back out the
    same line) don't count on their own; pairs of them matching an in-band ID
    swap are counted once each, flagged `swap_resolved`. Non-person tracks are
    ignored.
    """
    crossings: list[Crossing] = []
    retreats: list[tuple[Track, _BandVisit]] = []
    for track in tracks:
        if track.cls is not DetClass.PERSON:
            continue
        for visit in _band_visits(track, tripwire):
            if visit.is_traversal:
                crossings.append(_crossing(track, visit, tripwire, swap=False))
            else:
                retreats.append((track, visit))
    for (ta, va), (tb, vb) in _pair_swaps(retreats, swap_max_distance_m):
        crossings.append(_crossing(ta, va, tripwire, swap=True))
        crossings.append(_crossing(tb, vb, tripwire, swap=True))
    return sorted(crossings, key=lambda c: c.timestamp)
