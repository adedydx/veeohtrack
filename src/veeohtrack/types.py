"""Core data contracts shared across every pipeline stage.

These are the load-bearing shapes: `Detection` is what the detector emits and the
tracker consumes; `Track` is what the tracker emits and kinematics / counting /
classification consume; `CrossingRecord` is what the classifier produces and
storage writes — and it mirrors `schema/crossing.sql` field-for-field.

Nothing here does pipeline work; these are just the types the stages agree on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keep numpy out of the import path for the pure-logic/type layer
    import numpy as np


# ---------------------------------------------------------------------------
# Constrained vocabularies — these mirror the CHECK constraints in
# schema/crossing.sql. Keep the string values in sync with the DDL.
# ---------------------------------------------------------------------------
class DetClass(str, Enum):
    """The only COCO classes that survive the detector boundary (§5.1)."""
    PERSON = "person"
    BICYCLE = "bicycle"
    DOG = "dog"


class UserType(str, Enum):
    WALKER = "walker"
    RUNNER = "runner"
    CYCLIST = "cyclist"
    DOG_WALKER = "dog_walker"
    AMBIGUOUS = "ambiguous"


class Direction(str, Enum):
    IN = "in"
    OUT = "out"


class Association(str, Enum):
    NONE = "none"
    BICYCLE = "bicycle"
    DOG = "dog"


class AmbiguousReason(str, Enum):
    RUNG0_TRACK_QUALITY = "rung0_track_quality"
    RUNG2_SPEED_BAND = "rung2_speed_band"
    SPEED_CEILING = "speed_ceiling"
    NO_SIGNAL = "no_signal"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixel coordinates, xyxy (top-left, bottom-right)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center — the only part of the box on the ground plane (§5.3)."""
        return ((self.x1 + self.x2) / 2.0, self.y2)


# ---------------------------------------------------------------------------
# Frames & detections
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    """One captured frame. `capture.py` yields these; the detector consumes them."""
    index: int
    timestamp: float          # seconds, monotonic within a session
    image: "np.ndarray"


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame — the detector's output unit (§5.1).

    Pure per-frame content: *what* was detected, not *when*. The frame index and
    timestamp are context the pipeline supplies (the detector only ever sees the
    frame it was handed), so they're attached one layer up on `RawObservation`.
    """
    cls: DetClass
    bbox: BBox
    confidence: float

    @property
    def foot_point(self) -> tuple[float, float]:
        return self.bbox.foot_point


# ---------------------------------------------------------------------------
# Tracks
#
# Flow:  Detection --tracker--> RawTrack[RawObservation]
#                  --projection--> Track[Observation] --> downstream
#
# Content (Detection) gains one concern per layer: RawObservation adds temporal
# context (frame index + timestamp), Observation adds ground position. The
# tracker works in pixel space and is calibration-agnostic: it emits a RawTrack
# of RawObservations. The projection step (calibrate.GroundPlane) grounds each
# one, producing a Track of Observations that kinematics and counting consume.
# Both track types carry the same identity and quality signals; only the
# observation element type differs.
# ---------------------------------------------------------------------------
def _frame_span(items: list) -> int:
    """Frame span first→last inclusive (not len; gaps still count)."""
    if not items:
        return 0
    return items[-1].frame_index - items[0].frame_index + 1


@dataclass(frozen=True)
class RawObservation:
    """A detection sequenced into a track — content plus its temporal context (§5.2).

    The detector emits bare `Detection`s (what, not when); the tracker — which is
    handed the `Frame` — stamps the frame index and timestamp here as it appends
    to a RawTrack. Within a single detect() batch these fields are identical
    across every detection, so they only carry information once detections are
    sequenced across frames, which is exactly what this layer represents.
    """
    detection: Detection
    frame_index: int
    timestamp: float

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    @property
    def foot_point(self) -> tuple[float, float]:
        return self.detection.foot_point


@dataclass
class RawTrack:
    """The tracker's output (§5.2) — pixel space, no ground plane.

    Quality signals (`gap_count`, `instability_flag`, derived `length_frames`)
    are set here and carried through projection onto the Track; they feed the
    classifier's Rung-0 gate (§5.4), V3 drift (§5.6), and the CrossingRecord.
    """
    track_id: int
    cls: DetClass
    observations: list[RawObservation] = field(default_factory=list)
    gap_count: int = 0
    instability_flag: bool = False

    @property
    def length_frames(self) -> int:
        return _frame_span(self.observations)

    @property
    def last_seen_frame(self) -> int | None:
        return self.observations[-1].frame_index if self.observations else None


@dataclass(frozen=True)
class Observation:
    """A raw observation whose foot-point has been projected to the ground plane (§5.3).

    Produced by the projection step. `ground_xy` is computed once here and reused
    by both kinematics (windowed speed fit) and counting (crossing test), so the
    homography is applied exactly once per observation. The pixel-space and
    temporal fields stay reachable through the wrapped `raw` observation (e.g.
    `detection.bbox` for a pixel-space tripwire).
    """
    raw: RawObservation
    ground_xy: tuple[float, float]       # metres on the ground plane

    @property
    def detection(self) -> Detection:
        return self.raw.detection

    @property
    def frame_index(self) -> int:
        return self.raw.frame_index

    @property
    def timestamp(self) -> float:
        return self.raw.timestamp

    @property
    def confidence(self) -> float:
        return self.raw.confidence

    @property
    def foot_point(self) -> tuple[float, float]:
        """Pixel-space ground-contact point (pre-projection)."""
        return self.raw.foot_point


@dataclass
class Track:
    """The projection step's output — a grounded track kinematics/counting read.

    Same identity and quality signals as the RawTrack it came from; its
    observations additionally carry ground positions. Direction is deliberately
    NOT here — the committed direction (§5.2.1) is owned by the counting object.
    """
    track_id: int
    cls: DetClass
    observations: list[Observation] = field(default_factory=list)
    gap_count: int = 0
    instability_flag: bool = False

    @property
    def length_frames(self) -> int:
        return _frame_span(self.observations)

    @property
    def last_seen_frame(self) -> int | None:
        return self.observations[-1].frame_index if self.observations else None


# ---------------------------------------------------------------------------
# The persisted record — binds to schema/crossing.sql (§5.7)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossingRecord:
    """One counted tripwire crossing. Field-for-field mirror of the DDL.

    Grouping and nullability follow schema/crossing.sql; keep the two in sync.
    """
    # Identity / dedup
    crossing_id: str
    session_id: str
    node_id: str
    track_id: int

    # Provenance (V3 attribution, §5.6)
    schema_version: str                  # crossing-row DDL shape; distinct from config_version
    model_version: str
    config_version: str

    # The count (Tier 1/2, §7)
    rec_ld_ts: str                       # ISO-8601 UTC crossing instant
    direction: Direction
    committed_direction_confidence: float

    # The class (Tier 3, §7)
    user_type: UserType
    track_length_frames: int

    ambiguous_reason: AmbiguousReason | None = None
    association: Association = Association.NONE
    association_frame_count: int = 0

    # Kinematics (tuning + audit)
    speed_mps: float | None = None
    speed_fit_residual: float | None = None

    # Track quality (Rung 0 + V3 drift, §5.2/§5.6)
    track_gap_count: int = 0
    track_instability_flag: int = 0      # 0/1 to match the DDL

    # Detector health (V3 drift, §5.6)
    detection_confidence: float | None = None
