# VeeOhTracker — Architecture (Summary)

**An edge computer-vision system for trail usage analytics.** It counts and classifies trail users (walkers, runners, cyclists, dog-walkers) and their direction of travel. All inference runs on-device; only counts and metadata leave the node, never video.

> Full design rationale, trade-off analysis, and measurements live in `comprehensive-architecture.local.md` (local, untracked). This file is the pushed summary.

## Staged plan
- **V1 — Single node:** local detect → track → classify → log a full per-crossing record to SQLite. No network. *Done when* logged counts match a hand tally.
- **V2 — Fleet:** 2–4 nodes ship counts/metadata (never video) to a central store; dashboard + device health.
- **V3 — Edge MLOps:** model registry, OTA rollout with auto-rollback, drift detection on proxy signals. The distinctive stage.
- **V4 — Product:** public traffic page, trends, weather correlation.

## Hardware
Raspberry Pi 5 + Raspberry Pi AI Kit (Hailo-8L, 13 TOPS) + Active Cooler, HighPi Pro 5S case, Camera Module 3 (66°) on a 500mm cable, Talentcell 12V battery (true 5V/5A).

## Key decisions
- **Hailo-8L over -8:** single-stream detection is framerate/logic-bound, not TOPS-bound; the Pi 5's single PCIe lane can't feed the bigger part anyway.
- **Standard 66° lens over wide:** peripheral distortion degrades tracker ID consistency; trail width fits at 66° with standoff.
- **Interface seams from the start** (`detect.py`, `capture.py`, `storage.py`): swap detector/frame-source/sink without rewrites — enables desk testing (`MockDetector`), the multi-accelerator benchmark, and file-vs-camera development.

## Pipeline design
Core insight: **"runner vs. cyclist vs. dog-walker" is three kinds of inference, not one.** Detector (spatial) → tracker (temporal identity) → derived attributes (speed, association) → rule-based labeling at the tripwire.

- **Detector:** stock YOLOv8s COCO `.hef`, filtered to person/bicycle/dog. Fine-tuning deferred until Stage B evidence.
- **Tracker (ByteTrack):** two-pass matching recovers brief low-confidence occlusions; no appearance embeddings (cost + privacy). Known gap: same-instant, opposite-direction, same-speed ID swaps are information-free to a motion-only model — they corrupt only per-class directional splits, **not** totals. Mitigated by committing direction from *approach history* before tracks converge.
- **Kinematics:** never difference consecutive frames — fit a line over a ~1s window of ground-plane foot points (via homography); slope = speed. Prefer a robust fit; large residuals route to `ambiguous`.
- **Classifier (priority ladder):** Rung 0 track-quality gate → Rung 1 bicycle/dog association → Rung 2 speed bands, every rung with an `ambiguous` escape. Association sits above speed because casual cyclists overlap runner speeds (field-confirmed). Reliable `bicycle`/`dog` recall is the load-bearing, least-certain assumption.
- **Drift (V3):** watch several proxies (confidence PSI, detection rate, track quality, ambiguous rate) against a time-conditioned baseline; alarm only on corroborated divergence. A trigger for re-measurement, not a verdict.
- **V1 log schema (§5.7 in full doc):** one flat per-crossing record — identity, provenance (`model_version`/`config_version`), count, class, kinematics, track quality, detector health. DDL in `schema/crossing.sql`; a field omitted here forces a fleet-wide retrofit.

## Error budget (tiered)
- **Tier 1 — total throughput:** tightest; survives ID swaps, risked only by missed detections / fragmentation. Validated against a *reconciled* (event-by-event) hand tally, not a bulk count.
- **Tier 2 — directional split:** survives symmetric swaps in aggregate.
- **Tier 3 — per-class-per-direction:** reported, not targeted in V1.

**Accuracy and coverage are separate budgeted numbers** (`ambiguous` is honesty, not error). All numeric targets are set after the first Stage B run, floored by hand-tally reliability.

## V1 build order
Pure-logic stages first, against recorded footage, so failures are unambiguously in the logic:
**0** repo skeleton → **1** kinematics → **2** counting → **3** tracker → **4** classifier → **5** join on survey footage (needs a detector stand-in: software YOLOv8s) → **6–9** hardware bring-up, detector, capture, field integration → **10** Stage B (ground truth, fill TBDs). Stages 1–5 need no hardware.
