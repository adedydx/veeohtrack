"""The projection step — grounds a RawTrack into a Track (§5.3).

This is the thin "apply" in the three-places-one-job split that `calibrate.py`
describes: the homography *lives* in the local site config, `GroundPlane` *wraps*
it, and this step *applies* it. It walks a tracker-emitted `RawTrack` (pixel
space) and projects each observation's foot-point to ground-plane metres,
producing a `Track` of `Observation`s that kinematics and counting consume.

Two things to note:

- The projection happens exactly once per observation here; both kinematics
  (windowed speed fit) and counting (crossing test) then reuse the stored
  `ground_xy`, so the homography is never re-applied downstream.
- The track's identity and quality signals (`gap_count`, `instability_flag`)
  are carried straight through — projection changes *where* the observations
  are, not *how trustworthy* the track is.
"""
from __future__ import annotations

from .calibrate import GroundPlane
from .types import Observation, RawTrack, Track


def project_track(raw: RawTrack, ground: GroundPlane) -> Track:
    """Project a pixel-space RawTrack onto the ground plane.

    Each `RawObservation`'s foot-point (the only part of the box on the ground,
    §5.3) is mapped through the homography once and stored as `ground_xy`. The
    raw observation stays wrapped, so pixel-space and temporal fields remain
    reachable for anything that still needs them (e.g. a pixel-space tripwire).
    """
    grounded = [
        Observation(raw=obs, ground_xy=ground.project(obs.foot_point))
        for obs in raw.observations
    ]
    return Track(
        track_id=raw.track_id,
        cls=raw.cls,
        observations=grounded,
        gap_count=raw.gap_count,
        instability_flag=raw.instability_flag,
    )
