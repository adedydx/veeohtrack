"""Camera↔ground calibration (§5.3, §6).

`GroundPlane` is the wrapper in the "three places, one job each" split: the 3×3
homography *lives* in the local site config (§6, uncommitted), `GroundPlane`
*wraps* it and provides the projection, and a thin pipeline step *applies* it
(RawTrack → Track). Kinematics never sees the matrix — it consumes ground
positions that projection already computed.

Fitting the matrix from image↔ground correspondences (DLT) is the
calibration-construction step and lands in a later stage; this stage-0 type is
the wrapper plus the projection it provides. Tests use a synthetic matrix
(`identity`, or a known scale) so the pure-logic stages are exercised without
the real, still-to-be-independently-verified transform.
"""
from __future__ import annotations

from dataclasses import dataclass

Matrix3 = tuple  # 3×3 row-major, homogeneous image→ground


@dataclass(frozen=True)
class GroundPlane:
    """Wraps a solved homography; projects pixel foot-points to ground metres."""
    matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def project(self, pixel_xy: tuple[float, float]) -> tuple[float, float]:
        """Map a pixel point to ground-plane metres (homography + perspective divide)."""
        u, v = pixel_xy
        m = self.matrix
        x = m[0][0] * u + m[0][1] * v + m[0][2]
        y = m[1][0] * u + m[1][1] * v + m[1][2]
        w = m[2][0] * u + m[2][1] * v + m[2][2]
        if w == 0:
            raise ValueError("degenerate projection (w = 0)")
        return (x / w, y / w)

    @classmethod
    def identity(cls) -> "GroundPlane":
        """Ground == pixel. For tests that need a known, trivial transform."""
        return cls(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
