"""The interface seams the pipeline is built against (§4).

Each is a structural `Protocol`, so an implementation just needs the right shape
— no base class to inherit. This is what lets a `MockDetector` (canned
detections) or a recorded-file `FrameSource` stand in for hardware, and what
makes the same-model / different-silicon benchmark a drop-in swap.

Stage 0 defines the contracts only; implementations arrive in later stages.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from .types import CrossingRecord, Detection, Frame, RawTrack


@runtime_checkable
class FrameSource(Protocol):
    """`capture.py` — a live camera or a recorded video file, interchangeably."""

    @property
    def fps(self) -> float: ...

    def __iter__(self) -> Iterator[Frame]: ...


@runtime_checkable
class Detector(Protocol):
    """`detect.py` — a frame in, a list of detections out. Hides all Hailo/HEF
    specifics (§4). Implementations filter to the DetClass set at this boundary."""

    def detect(self, frame: Frame) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    """The ByteTrack wrapper (§5.2) — per-frame detections to stable-ID tracks.

    Emits RawTracks (pixel space); the projection step grounds them into Tracks.
    This is the seam that holds both the detections and the `Frame`, so it stamps
    each detection's temporal context (frame index + timestamp) into the
    RawObservations it appends — the detector itself emits time-free Detections.

    `update` returns the tracks active this frame; counting needs *finished*
    tracks, which are handed off by `drain_finished` (expired since the last
    call) and `finish` (end of stream: flush everything still alive).
    """

    def update(self, detections: list[Detection], frame: Frame) -> list[RawTrack]: ...

    def drain_finished(self) -> list[RawTrack]: ...

    def finish(self) -> list[RawTrack]: ...


@runtime_checkable
class Storage(Protocol):
    """`storage.py` — SQLite sink in V1, MQTT publisher in V2 (§4)."""

    def write(self, record: CrossingRecord) -> None: ...

    def close(self) -> None: ...
