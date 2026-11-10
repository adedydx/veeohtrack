"""Render the V1 log-schema entity-relationship diagram.

V1 physically stores ONE flat table (`crossing`). This ERD shows the *logical*
model that flat record denormalizes: the entities collapsed into the row via its
id/version columns, plus the relationships and cardinalities among them. Those
boxed entities are what become first-class tables in V2 (node / session) and V3
(model + config registries).

Run: .render-venv/bin/python schema/render_erd.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

W, H = 1120.0, 928.0
HEADER_H = 34.0
ROW_H = 26.0

# palette
DARK   = "#0d1117"
PURPLE = "#8250df"; PURPLE_T = "#f3ebff"
GREEN  = "#1a7f37"; GREEN_T  = "#e2fbe8"
GRAY   = "#57606a"; GRAY_T   = "#eef1f4"
PK_C   = "#9a6700"   # gold
FK_C   = "#0969da"   # blue
INK    = "#1f2328"
MUTE   = "#6e7781"

fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)          # top-origin: y increases downward
ax.axis("off")

# ---- title ---------------------------------------------------------------
ax.text(40, 34, "VeeOhTracker — V1 Log Schema (Entity–Relationship)",
        fontsize=17, fontweight="bold", color=INK, va="center")
ax.text(40, 58,
        "Logical model. V1 persists this as ONE flat table (crossing); boxed entities are denormalized into its id/version columns.",
        fontsize=9.5, color=MUTE, va="center")


def entity(x, y, w, title, badge, rows, header_c, tint_c, dashed=False):
    """rows: list of (kind, text) where kind in {'PK','FK',''}."""
    h = HEADER_H + len(rows) * ROW_H
    ls = (0, (5, 3)) if dashed else "solid"
    lw = 1.4 if dashed else 1.2
    body = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0,rounding_size=8",
                          linewidth=lw, edgecolor=header_c,
                          facecolor=tint_c, linestyle=ls, zorder=2)
    ax.add_patch(body)
    # header band
    hd = FancyBboxPatch((x, y), w, HEADER_H,
                        boxstyle="round,pad=0,rounding_size=8",
                        linewidth=0, facecolor=header_c, zorder=3)
    ax.add_patch(hd)
    ax.add_patch(Rectangle((x, y + HEADER_H - 10), w, 10,
                           linewidth=0, facecolor=header_c, zorder=3))
    ax.text(x + 12, y + HEADER_H / 2, title, fontsize=12.5,
            fontweight="bold", color="white", va="center",
            family="DejaVu Sans Mono", zorder=4)
    if badge:
        ax.text(x + w - 10, y + HEADER_H / 2, badge, fontsize=8.2,
                color="white", alpha=0.85, va="center", ha="right", zorder=4)
    # rows
    for i, (kind, text) in enumerate(rows):
        ry = y + HEADER_H + i * ROW_H + ROW_H / 2
        if i:
            ax.plot([x + 6, x + w - 6], [y + HEADER_H + i * ROW_H] * 2,
                    color="#00000012", lw=0.6, zorder=3)
        if kind:
            bc = PK_C if kind == "PK" else FK_C
            ax.text(x + 12, ry, kind, fontsize=8.5, fontweight="bold",
                    color=bc, va="center", family="DejaVu Sans Mono", zorder=4)
        ax.text(x + 46, ry, text, fontsize=10, color=INK, va="center",
                family="DejaVu Sans Mono",
                fontweight="bold" if kind == "PK" else "normal", zorder=4)
    return h


# ---- geometry ------------------------------------------------------------
LEFT_W = 236
node   = dict(x=40,  y=80,  w=LEFT_W)
sess   = dict(x=40,  y=214, w=LEFT_W)
track  = dict(x=40,  y=372, w=LEFT_W)
rawobs = dict(x=40,  y=556, w=LEFT_W)
detn   = dict(x=40,  y=660, w=LEFT_W)
cross  = dict(x=418, y=150, w=344)
modl   = dict(x=832, y=150, w=248)
conf   = dict(x=832, y=356, w=248)

h_node = entity(**node,  title="Node", badge="V2 device table",
                header_c=PURPLE, tint_c=PURPLE_T, rows=[
                    ("PK", "node_id"),
                    ("",   "first_seen"),
                    ("",   "last_seen")])
h_sess = entity(**sess,  title="Session", badge="V2 session table",
                header_c=GREEN, tint_c=GREEN_T, rows=[
                    ("PK", "session_id"),
                    ("FK", "node_id"),
                    ("",   "started_at"),
                    ("",   "ended_at")])
h_trk  = entity(**track, title="Track", badge="→ crossing.track_*",
                header_c=GREEN, tint_c=GREEN_T, rows=[
                    ("PK", "(session_id, track_id)"),
                    ("FK", "session_id"),
                    ("",   "track_length_frames"),
                    ("",   "track_gap_count"),
                    ("",   "track_instability_flag")])
h_raw  = entity(**rawobs, title="RawObservation", badge="",
                header_c=GRAY, tint_c=GRAY_T, dashed=True, rows=[
                    ("",   "frame_index"),
                    ("",   "timestamp")])
h_det  = entity(**detn,  title="Detection", badge="NOT persisted (V1)",
                header_c=GRAY, tint_c=GRAY_T, dashed=True, rows=[
                    ("",   "bbox"),
                    ("",   "class"),
                    ("",   "confidence")])
h_crs  = entity(**cross, title="Crossing", badge="physical table — V1",
                header_c=DARK, tint_c="#f6f8fa", rows=[
                    ("PK", "crossing_id"),
                    ("FK", "session_id, track_id  → Track"),
                    ("FK", "model_version  → ModelVersion"),
                    ("FK", "config_version → ConfigVersion"),
                    ("",   "schema_version"),
                    ("",   "rec_ld_ts"),
                    ("",   "direction"),
                    ("",   "committed_direction_confidence"),
                    ("",   "user_type"),
                    ("",   "ambiguous_reason"),
                    ("",   "association"),
                    ("",   "association_frame_count"),
                    ("",   "speed_mps"),
                    ("",   "speed_fit_residual"),
                    ("",   "detection_confidence")])
h_mod  = entity(**modl,  title="ModelVersion", badge="V3 registry",
                header_c=PURPLE, tint_c=PURPLE_T, rows=[
                    ("PK", "model_version"),
                    ("",   "artifact_hash"),
                    ("",   "exported_at")])
h_cfg  = entity(**conf,  title="ConfigVersion", badge="V3 registry",
                header_c=PURPLE, tint_c=PURPLE_T, rows=[
                    ("PK", "config_version"),
                    ("",   "walk_max, run_min"),
                    ("",   "tripwire_pts"),
                    ("",   "hash")])


# ---- crow's-foot connectors ---------------------------------------------
def _uv(a, b):
    v = np.array(b, float) - np.array(a, float)
    n = np.hypot(*v)
    return v / (n if n else 1.0)


def _glyph(P, d, kind):
    """Draw cardinality glyph at endpoint P; d points from P into the line."""
    P = np.array(P, float)
    u = np.array([-d[1], d[0]])       # perpendicular
    def L(p, q):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=1.3, zorder=5,
                solid_capstyle="round")
    if kind in ("many", "zeromany"):
        k = P + d * 16
        L(k, P - u * 8); L(k, P + u * 8); L(k, P)
        if kind == "zeromany":
            c = P + d * 26
            ax.add_patch(Circle(c, 5, fill=True, facecolor="white",
                                edgecolor=INK, lw=1.2, zorder=5))
    else:  # one / zeroone
        k = P + d * 14
        L(k - u * 8, k + u * 8)
        if kind == "zeroone":
            c = P + d * 26
            ax.add_patch(Circle(c, 5, fill=True, facecolor="white",
                                edgecolor=INK, lw=1.2, zorder=5))


def rel(P1, P2, c1, c2, label, dashed=False, lx=None, ly=None):
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], color=MUTE, lw=1.3,
            linestyle=(0, (5, 3)) if dashed else "solid", zorder=1)
    _glyph(P1, _uv(P1, P2), c1)
    _glyph(P2, _uv(P2, P1), c2)
    mx = lx if lx is not None else (P1[0] + P2[0]) / 2
    my = ly if ly is not None else (P1[1] + P2[1]) / 2
    ax.text(mx, my, label, fontsize=8.8, color=INK, va="center", ha="center",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"),
            zorder=6)


# left spine: Node 1—* Session 1—* Track 1—* Detection
cx = node["x"] + LEFT_W / 2
rel((cx, node["y"] + h_node), (cx, sess["y"]), "one", "many", "records")
rel((cx, sess["y"] + h_sess), (cx, track["y"]), "one", "many", "contains")
rel((cx, track["y"] + h_trk), (cx, rawobs["y"]), "one", "many", "aggregates",
    dashed=True)
rel((cx, rawobs["y"] + h_raw), (cx, detn["y"]), "one", "one", "wraps",
    dashed=True)

# Track 1 —— 0..1 Crossing
rel((track["x"] + LEFT_W, track["y"] + 60), (cross["x"], cross["y"] + 150),
    "one", "zeroone", "produces")

# ModelVersion 1 —* Crossing   /   ConfigVersion 1 —* Crossing
rel((modl["x"], modl["y"] + 54), (cross["x"] + cross["w"], cross["y"] + 96),
    "one", "many", "detected by")
rel((conf["x"], conf["y"] + 60), (cross["x"] + cross["w"], cross["y"] + 300),
    "one", "many", "labeled under")


# ---- footnotes / legend --------------------------------------------------
fy = 800
ax.plot([40, W - 40], [fy - 18, fy - 18], color="#00000010", lw=1)
notes = [
    ("V1 realization:  every relationship above is a foreign-key column carried on the single crossing row —",  INK, False),
    ("node_id, session_id, (session_id, track_id), model_version, config_version — so the whole graph is one flat table.", INK, False),
    ("V2/V3:  Node & Session become their own tables (fleet + telemetry); ModelVersion & ConfigVersion become", INK, False),
    ("registries that V3 joins for drift attribution.", INK, False),
    ("Detection (content) + RawObservation (frame_index/timestamp, stamped by the tracker) are conceptual only — never stored in V1;", MUTE, True),
    ("one crossing-moment confidence lands as crossing.detection_confidence.", MUTE, True),
]
for i, (txt, col, ital) in enumerate(notes):
    ax.text(40, fy + i * 17, txt, fontsize=9, color=col, va="center",
            style="italic" if ital else "normal")

# cardinality legend
ax.text(40, H - 20, "PK primary key   FK foreign key   |— one   —<  many   O— zero-or-one",
        fontsize=8.6, color=MUTE, va="center", family="DejaVu Sans Mono")

fig.savefig("schema/crossing_erd.png", dpi=150)
fig.savefig("schema/crossing_erd.svg")
print("wrote schema/crossing_erd.png and schema/crossing_erd.svg")
