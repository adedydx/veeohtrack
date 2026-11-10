"""ByteTrack-style tracker — per-frame detections to stable-ID tracks (§5.2).

This is Stage 3. It implements the `Tracker` seam (`interfaces.py`): detections
plus the `Frame` in, a list of `RawTrack`s out, in pixel space. The projection
step grounds those into `Track`s for kinematics (Stage 1) and counting (Stage 2).

Why a tracker at all: without persistent identity there is no speed, no
direction, and every person is recounted every frame. The tracker's whole job is
to decide "this detection is the same object as that one last frame."

This is a compact, dependency-light implementation of the ByteTrack *association
scheme*, faithful to the parts the design leans on (§5.2):

- **Two-pass matching.** Confident detections are matched to existing tracks
  first; then *leftover low-confidence* detections are matched to tracks that
  found no partner. A weak detection sitting where a track was predicted is
  almost certainly the same person briefly occluded — not noise. This is the
  direct answer to the trail's worst cases (dogs behind legs, people passing).
- **Motion prediction bridges brief gaps.** Each track carries a constant-
  velocity estimate (a lightweight stand-in for ByteTrack's Kalman filter) and
  is kept alive for a grace period (`track_buffer` frames) when unmatched, so a
  short full occlusion is recovered under the same ID rather than fragmenting
  into a new one.
- **No appearance embeddings.** Association is pure geometry (IoU of predicted
  vs. detected boxes). This is a deliberate *privacy* choice (§5.2), not only a
  cost one — no person-identifying feature vector is ever computed.

What it deliberately does *not* try to do is be a perfect tracker. Per §5.2 the
goal is to be *honest about track quality*: every emitted `RawTrack` carries
`gap_count` (frames bridged by prediction) and `instability_flag` (recovered via
a gap or a low-confidence rescue). Those feed the classifier's Rung-0 gate
(§5.4) and V3 drift monitoring (§5.6) — short or patched tracks get routed to
`ambiguous` rather than trusted.

A production node could vendor the official ByteTrack behind this same seam; the
interface (and everything downstream) would not change.
"""
from __future__ import annotations

import numpy as np

from .types import BBox, Detection, Frame, RawObservation, RawTrack

# Confidence split for the two passes. >= HIGH is matched first; [LOW, HIGH) is
# the "leftover low-confidence" pool matched second; < LOW is dropped entirely.
HIGH_THRESH = 0.5
LOW_THRESH = 0.1

# Minimum IoU (predicted track box vs. detection box) to associate them.
MATCH_IOU = 0.3

# Frames a track survives unmatched before it expires (the grace period that
# bridges brief full occlusions). ~2 s at 15 fps.
TRACK_BUFFER = 30


def _to_xyxy(box: BBox) -> np.ndarray:
    return np.array([box.x1, box.y1, box.x2, box.y2], dtype=float)


def _boxes(dets: list[Detection]) -> np.ndarray:
    if not dets:
        return np.zeros((0, 4), dtype=float)
    return np.stack([_to_xyxy(d.bbox) for d in dets])


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in `a` against every box in `b` → (len(a), len(b))."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _class_gated_iou(track_boxes: np.ndarray, track_classes: list, dets: list[Detection]) -> np.ndarray:
    """IoU with cross-class pairs zeroed. A rider's box and their bike's box
    overlap heavily; without this gate a person track can absorb a bicycle
    detection, jump to its box, and fragment the person into a new ID."""
    iou = _iou_matrix(track_boxes, _boxes(dets))
    if iou.size:
        same = np.array([[tc is d.cls for d in dets] for tc in track_classes], dtype=bool)
        iou = np.where(same, iou, 0.0)
    return iou


def _greedy_match(iou: np.ndarray, min_iou: float):
    """Greedily pair rows to columns by descending IoU, above `min_iou`.

    Returns (matches, unmatched_rows, unmatched_cols). Deterministic; ample for
    the single-stream trail workload (linear assignment is an easy later swap).
    """
    iou = iou.copy()
    matches: list[tuple[int, int]] = []
    while iou.size and iou.max() >= min_iou:
        i, j = np.unravel_index(int(np.argmax(iou)), iou.shape)
        matches.append((int(i), int(j)))
        iou[i, :] = -1.0
        iou[:, j] = -1.0
    matched_rows = {i for i, _ in matches}
    matched_cols = {j for _, j in matches}
    unmatched_rows = [i for i in range(iou.shape[0]) if i not in matched_rows]
    unmatched_cols = [j for j in range(iou.shape[1]) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


class _Track:
    """Mutable per-object state while the track is alive."""

    def __init__(self, track_id: int, det: Detection, frame: Frame):
        self.track_id = track_id
        self.cls = det.cls
        self.box = _to_xyxy(det.bbox)
        self.velocity = np.zeros(4, dtype=float)
        self.last_frame = frame.index
        self.time_since_update = 0
        self.gap_count = 0
        self.instability_flag = False
        self.observations = [RawObservation(det, frame.index, frame.timestamp)]

    def predict(self, frame_index: int) -> np.ndarray:
        """Where this track's box is expected now, by constant velocity."""
        return self.box + self.velocity * (frame_index - self.last_frame)

    def update_with(self, det: Detection, frame: Frame, via_low: bool) -> None:
        new_box = _to_xyxy(det.bbox)
        steps = frame.index - self.last_frame
        if steps > 0:
            self.velocity = (new_box - self.box) / steps
        if self.time_since_update > 0:        # we bridged a gap to get here
            self.gap_count += self.time_since_update
            self.instability_flag = True
        if via_low:                           # rescued by a low-confidence detection
            self.instability_flag = True
        self.box = new_box
        self.last_frame = frame.index
        self.time_since_update = 0
        self.observations.append(RawObservation(det, frame.index, frame.timestamp))

    def mark_missed(self) -> None:
        self.time_since_update += 1

    def to_rawtrack(self) -> RawTrack:
        return RawTrack(
            track_id=self.track_id,
            cls=self.cls,
            observations=list(self.observations),
            gap_count=self.gap_count,
            instability_flag=self.instability_flag,
        )


class ByteTracker:
    """Two-pass IoU tracker. Satisfies the `Tracker` protocol (`interfaces.py`).

    `update` is called once per frame with that frame's detections (possibly an
    empty list on a dropped frame) and returns the tracks active *this* frame.
    Tracks that age out of the grace period are handed off via
    `drain_finished()`, and the pipeline runs projection → counting on them.
    Association is per class: a track only ever matches its own class.
    """

    def __init__(
        self,
        high_thresh: float = HIGH_THRESH,
        low_thresh: float = LOW_THRESH,
        match_iou: float = MATCH_IOU,
        track_buffer: int = TRACK_BUFFER,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou = match_iou
        self.track_buffer = track_buffer
        self._tracks: list[_Track] = []
        self._removed: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection], frame: Frame) -> list[RawTrack]:
        high = [d for d in detections if d.confidence >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        pred = np.stack([t.predict(frame.index) for t in self._tracks]) if self._tracks else np.zeros((0, 4))

        # Pass 1: existing tracks ↔ high-confidence detections of the same class.
        classes = [t.cls for t in self._tracks]
        m1, unmatched_rows, _ = _greedy_match(_class_gated_iou(pred, classes, high), self.match_iou)
        for ti, di in m1:
            self._tracks[ti].update_with(high[di], frame, via_low=False)

        # Pass 2: tracks still unmatched ↔ leftover low-confidence detections.
        rem = unmatched_rows
        rem_pred = np.stack([self._tracks[i].predict(frame.index) for i in rem]) if rem else np.zeros((0, 4))
        rem_classes = [self._tracks[i].cls for i in rem]
        m2, still_unmatched_pos, _ = _greedy_match(
            _class_gated_iou(rem_pred, rem_classes, low), self.match_iou
        )
        for ri, di in m2:
            self._tracks[rem[ri]].update_with(low[di], frame, via_low=True)

        # Tracks that found no partner in either pass: age them.
        for pos in still_unmatched_pos:
            self._tracks[rem[pos]].mark_missed()

        # New tracks are born only from unmatched high-confidence detections.
        matched_high = {di for _, di in m1}
        for di, det in enumerate(high):
            if di not in matched_high:
                self._tracks.append(_Track(self._next_id, det, frame))
                self._next_id += 1

        # Expire tracks past the grace period.
        alive: list[_Track] = []
        for t in self._tracks:
            if t.time_since_update > self.track_buffer:
                self._removed.append(t)
            else:
                alive.append(t)
        self._tracks = alive

        return [t.to_rawtrack() for t in self._tracks if t.time_since_update == 0]

    def drain_finished(self) -> list[RawTrack]:
        """Tracks that expired past the grace period since the last drain, for
        the pipeline to finalize into crossings. Each is returned once."""
        out = [t.to_rawtrack() for t in self._removed]
        self._removed.clear()
        return out

    def finish(self) -> list[RawTrack]:
        """End of stream: flush every still-alive track as finished."""
        out = self.drain_finished() + [t.to_rawtrack() for t in self._tracks]
        self._tracks = []
        return out
