"""Render the LungNoduleAgent-Enhanced architecture diagram to figures/architecture.png.

A layered, minimal-arrow system-architecture diagram: each layer is a labelled
band; repeated sub-modules (experts, specialists) use a depth-stacked box to
signal parallelism instead of drawing N boxes; related safety modules are
grouped into one bordered sub-panel with one inbound and one outbound arrow
rather than individual criss-crossing arrows; memory and the knowledge graph
are shown as side panels (infrastructure, not a step in the flow) with a single
connecting arrow each. Every arrow present carries a distinct meaning.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch,
                                Rectangle, RegularPolygon)

BLUE = "#3B5C93"
BLUE_L = "#DCE6F5"
ORANGE = "#C96A2A"
ORANGE_L = "#F6E4D4"
GREEN = "#3E7A4C"
GREEN_L = "#DEEFE1"
RED = "#A63D3D"
RED_L = "#F5DCDC"
PURPLE = "#5B4C8A"
PURPLE_L = "#E7E2F3"
GREY = "#53575D"
GREY_L = "#ECEDEF"
INK = "#1A1C1F"
WHITE = "#FFFFFF"


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, fc=WHITE, ec=INK, fs=9.6, bold=False, lw=1.3,
         z=4, title=None, title_fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))
    ty = y + h / 2
    if title:
        ax.text(x + w / 2, y + h - 0.55, title, ha="center", va="top",
                fontsize=title_fs, fontweight="bold", color=ec, zorder=z + 1)
        ty = y + h / 2 - 0.35
    ax.text(x + w / 2, ty, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=z + 1, fontweight="bold" if bold else "normal",
            linespacing=1.4)
    return x, y, w, h


def _stack(ax, x, y, w, h, n, text, fc, ec, fs=9.0, offset=0.7):
    for i in range(n - 1, 0, -1):
        ax.add_patch(FancyBboxPatch(
            (x + i * offset, y - i * offset * 0.5), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.0, edgecolor=ec, facecolor=fc, alpha=0.5, zorder=3))
    return _box(ax, x, y, w, h, text, fc=fc, ec=ec, fs=fs, z=5)


def _arr(ax, p0, p1, color=INK, lw=1.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                 linewidth=lw, color=color, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=8,
                 shrinkA=2, shrinkB=2))


def _route(ax, p0, p1, via_y, color=INK, lw=1.8, ls="-"):
    """Clean right-angle (orthogonal) connector routed through the empty gap
    between two bands at height `via_y`: down/up from p0, sideways, then
    down/up into p1. Used instead of a diagonal so long cross-band links never
    read as a diagonal 'swoop' through other boxes."""
    x0, y0 = p0
    x1, y1 = p1
    segs = [(p0, (x0, via_y)), ((x0, via_y), (x1, via_y)), ((x1, via_y), p1)]
    for k, (a, b) in enumerate(segs):
        style = "-|>" if k == len(segs) - 1 else "-"
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=15,
                     linewidth=lw, color=color, linestyle=ls, zorder=8,
                     shrinkA=0, shrinkB=(2 if style == "-|>" else 0)))


def _band(ax, x, y, w, h, label, color, fc):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=1.0, edgecolor=color,
                 facecolor=fc, alpha=1.0, zorder=0))
    ax.add_patch(Rectangle((x, y), 3.0, h, linewidth=0, facecolor=color,
                 alpha=0.85, zorder=1))
    ax.text(x + 1.5, y + h / 2, label, ha="center", va="center", fontsize=10.5,
            color=WHITE, fontweight="bold", rotation=90, zorder=2)


def _panel(ax, x, y, w, h, title, color, light):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 linewidth=1.3, edgecolor=color, facecolor=light, zorder=3))
    ax.text(x + w / 2, y + h - 0.6, title, ha="center", va="top", fontsize=9.4,
            fontweight="bold", color=color, zorder=4)


def _person_icon(ax, cx, cy, r, color):
    ax.add_patch(Circle((cx, cy + r * 0.60), r * 0.40, facecolor=color,
                 edgecolor="none", zorder=6))
    ax.add_patch(FancyBboxPatch((cx - r * 0.58, cy - r * 0.55), r * 1.16, r * 0.95,
                 boxstyle="round,pad=0,rounding_size=0.30", facecolor=color,
                 edgecolor="none", zorder=6))


def _agent_node(ax, cx, cy, r, label, sub, color, light):
    ax.add_patch(Circle((cx, cy), r, facecolor=light, edgecolor=color,
                 linewidth=1.5, zorder=5))
    _person_icon(ax, cx, cy + r * 0.14, r * 0.58, color)
    ax.text(cx, cy - r - 0.35, label, ha="center", va="top", fontsize=8.6,
            fontweight="bold", color=color, zorder=6)
    ax.text(cx, cy - r - 1.55, sub, ha="center", va="top", fontsize=7.4,
            color=GREY, style="italic", zorder=6)


def _graph_icon(ax, x, y, w, h, color):
    rng = np.random.RandomState(3)
    pts = np.column_stack([rng.uniform(x + 0.2, x + w - 0.2, 6),
                           rng.uniform(y + 0.2, y + h - 0.2, 6)])
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (2, 5)]:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=color,
                lw=1.0, alpha=0.8, zorder=5)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.11, facecolor=color, edgecolor="none",
                     zorder=6))


def _cylinder(ax, x, y, w, h, label, color, light, fs=7.6):
    e = h * 0.24
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, facecolor=light,
                 edgecolor=color, linewidth=1.2, zorder=4))
    ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, facecolor=light,
                 edgecolor=color, linewidth=1.2, zorder=5))
    ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, facecolor=light,
                 edgecolor=color, linewidth=1.2, zorder=4))
    ax.text(x + w / 2, y + h / 2 - e * 0.15, label, ha="center", va="center",
            fontsize=fs, color=INK, zorder=6, linespacing=1.3, fontweight="bold")


def _shield(ax, cx, cy, s, color, light):
    verts = [(cx - s, cy + s * 0.75), (cx + s, cy + s * 0.75), (cx + s, cy - s * 0.15),
             (cx, cy - s * 1.05), (cx - s, cy - s * 0.15)]
    ax.add_patch(plt.Polygon(verts, closed=True, facecolor=light, edgecolor=color,
                 linewidth=1.5, zorder=5))
    ax.plot([cx - s * 0.40, cx - s * 0.06, cx + s * 0.50],
            [cy + s * 0.05, cy - s * 0.32, cy + s * 0.32],
            color=color, lw=2.0, zorder=6, solid_capstyle="round")


def _bang(ax, cx, cy, s, color, light):
    ax.add_patch(RegularPolygon((cx, cy), numVertices=8, radius=s,
                 orientation=np.pi / 8, facecolor=light, edgecolor=color,
                 linewidth=1.5, zorder=5))
    ax.text(cx, cy, "!", ha="center", va="center", fontsize=14, fontweight="bold",
            color=color, zorder=6)


def _fork(ax, cx, cy, s, color, light):
    ax.add_patch(Circle((cx, cy), s, facecolor=light, edgecolor=color,
                 linewidth=1.5, zorder=5))
    ax.plot([cx - s * 0.5, cx, cx], [cy - s * 0.30, cy, cy + s * 0.55],
            color=color, lw=2.0, zorder=6, solid_capstyle="round")
    ax.plot([cx, cx + s * 0.55], [cy, cy - s * 0.45], color=color, lw=2.0,
            zorder=6, solid_capstyle="round")


def _reliability(ax, x, y, w, h, color):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=WHITE, edgecolor=color,
                 linewidth=1.1, zorder=5))
    ax.plot([x, x + w], [y, y + h], color="#BEBEBE", lw=1.0, ls=(0, (2, 2)), zorder=6)
    xs = np.linspace(x + w * 0.10, x + w * 0.90, 5)
    ys = y + h * np.array([0.10, 0.28, 0.42, 0.68, 0.88])
    ax.plot(xs, ys, color=color, lw=1.8, marker="o", ms=3.2, zorder=7)


def _gavel(ax, cx, cy, s, color, light):
    ax.add_patch(Circle((cx, cy), s, facecolor=light, edgecolor=color,
                 linewidth=1.5, zorder=5))
    ax.add_patch(FancyBboxPatch((cx - s * 0.55, cy - s * 0.15), s * 1.1, s * 0.30,
                 boxstyle="round,pad=0,rounding_size=0.06", facecolor=color,
                 edgecolor="none", zorder=6))
    ax.add_patch(Circle((cx - s * 0.55, cy), s * 0.15, facecolor=color,
                 edgecolor="none", zorder=6))


# --------------------------------------------------------------------------- #
def render(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 10.6))
    ax.set_xlim(0, 145)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(70, 98.0, "Layered architecture of LungNoduleAgent-Enhanced",
            ha="center", va="center", fontsize=15.5, fontweight="bold", color=INK)

    band_x, band_w = 4, 126
    _band(ax, band_x, 79, band_w, 16.5, "PERCEPTION", BLUE, "#F5F7FB")
    _band(ax, band_x, 58.5, band_w, 17.5, "GROUNDING", ORANGE, "#FBF6F1")
    _band(ax, band_x, 22.5, band_w, 33, "REASONING", GREEN, "#F3F8F4")
    _band(ax, band_x, 3, band_w, 16.5, "DECISION", RED, "#FAF3F3")

    # ================= infrastructure side panel (no per-step arrows) =====
    _panel(ax, 132, 22.5, 11, 73, "", PURPLE, "#FBFAFD")
    ax.text(137.5, 93.3, "MEMORY", ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=PURPLE)
    _cylinder(ax, 133.4, 81.5, 8.2, 9.5, "Working", PURPLE, PURPLE_L, fs=8.0)
    _cylinder(ax, 133.4, 65.0, 8.2, 9.5, "Episodic", PURPLE, PURPLE_L, fs=8.0)
    _cylinder(ax, 133.4, 48.5, 8.2, 9.5, "Semantic", PURPLE, PURPLE_L, fs=8.0)
    ax.text(137.5, 44.2, "Available to\nevery stage", ha="center",
            va="top", fontsize=6.8, color=PURPLE, style="italic", linespacing=1.3)

    # ================= PERCEPTION =================
    y0 = 84.5
    _box(ax, 7, y0, 16, 9, "Axial CT slice\n+ ground truth mask\n(used for scoring\nonly)",
         fc=GREY_L, ec=GREY, fs=8.2, bold=True)
    _stack(ax, 28, y0 + 1, 15, 7, 5, "Mixture of\nexperts (x5)", BLUE_L, BLUE, fs=8.4)
    _box(ax, 51, y0 + 1, 16, 7, "IoU distance and\nDBSCAN clustering", BLUE_L, BLUE, fs=8.2)
    _box(ax, 72, y0 + 1, 16, 7, "Judging panel\n(3 weighted votes)", BLUE_L, BLUE, fs=8.2)
    _box(ax, 93, y0 + 1, 16, 7, "Refiner and\nmeasurements", BLUE_L, BLUE, fs=8.2)
    for xa in (23, 44, 67, 88):
        _arr(ax, (xa, y0 + 4.5), (xa + 5, y0 + 4.5), BLUE)

    # ================= GROUNDING =================
    y1 = 63.5
    _box(ax, 28, y1, 18, 8, "Focal crop with\ncontext margin", ORANGE_L, ORANGE, fs=8.4)
    _box(ax, 53, y1, 20, 8, "MedPrompt\nvision language model", ORANGE_L, ORANGE, fs=8.4)
    _box(ax, 80, y1, 26, 8, "Localized CT report\n(structured findings text)",
         ORANGE_L, ORANGE, fs=8.4)
    _arr(ax, (46, y1 + 4), (53, y1 + 4), ORANGE)
    _arr(ax, (73, y1 + 4), (80, y1 + 4), ORANGE)
    # measurements -> focal crop: clean right-angle route through the gap
    # between the Perception and Grounding bands (no diagonal swoop)
    _route(ax, (101, 85.5), (37, 71.5), via_y=77.5, color=BLUE)

    # ================= COLLABORATIVE REASONING =================
    _box(ax, 7, 45.5, 20, 8, "Triage router\n(easy vs. ambiguous)", GREEN_L, GREEN, fs=8.4)

    board_x, board_y, board_w, board_h = 30, 43.5, 100, 12
    _panel(ax, board_x, board_y, board_w, board_h,
          "SPECIALIST BOARD  -  role- and confidence-weighted vote", GREEN, GREEN_L)
    roles = [("Radiologist", "imaging"), ("Pathologist", "histology"),
             ("Oncologist", "malignancy risk"), ("Pulmonologist", "clinical context")]
    for i, (label, sub) in enumerate(roles):
        cx = board_x + board_w * (i + 0.5) / 4
        _agent_node(ax, cx, board_y + 5.6, 3.1, label, sub, GREEN, WHITE)
    _arr(ax, (27, 49.5), (30, 49.5), GREEN)

    # Knowledge graph sits with the router in the same left-hand column that
    # feeds the board; grouped by position rather than an extra arrow.
    _panel(ax, 7, 24, 20, 18.5, "Knowledge graph", GREY, "#F6F6F7")
    _graph_icon(ax, 9.4, 26.5, 15.5, 11.5, GREY)

    # report -> board: clean right-angle route through the inter-band gap
    _route(ax, (93, 63.5), (112, board_y + board_h), via_y=57, color=ORANGE)

    safety_y, safety_h = 30, 11
    _panel(ax, board_x, safety_y, board_w, safety_h,
          "SAFETY CHECKS  -  every claim grounded, every consensus challenged",
          RED, RED_L)
    _shield(ax, board_x + board_w * 0.30, safety_y + 4.4, 2.7, RED, WHITE)
    ax.text(board_x + board_w * 0.30, safety_y + 1.0, "Evidence verifier",
            ha="center", va="top", fontsize=8.0, fontweight="bold", color=RED)
    _bang(ax, board_x + board_w * 0.70, safety_y + 4.4, 2.9, RED, WHITE)
    ax.text(board_x + board_w * 0.70, safety_y + 1.0, "Devil's advocate",
            ha="center", va="top", fontsize=8.0, fontweight="bold", color=RED)
    ax.add_patch(FancyArrowPatch((board_x + board_w / 2, board_y),
                 (board_x + board_w / 2, safety_y + safety_h), arrowstyle="-|>",
                 mutation_scale=18, linewidth=2.2, color=GREY, zorder=8,
                 shrinkA=1, shrinkB=1))

    chair_y, chair_h = 24, 5
    _box(ax, board_x, chair_y, board_w, chair_h,
         "Chair: super-majority agreement and label stability", fc=WHITE, ec=GREEN,
         fs=9.0, bold=True)
    _gavel(ax, board_x + 6, chair_y + chair_h / 2, 2.0, GREEN, GREEN_L)
    ax.add_patch(FancyArrowPatch((board_x + board_w / 2, safety_y),
                 (board_x + board_w / 2, chair_y + chair_h), arrowstyle="-|>",
                 mutation_scale=18, linewidth=2.2, color=GREY, zorder=8,
                 shrinkA=1, shrinkB=1))

    loop_x = 131.6
    ax.add_patch(FancyArrowPatch((board_x + board_w, chair_y + chair_h / 2),
                 (loop_x, chair_y + chair_h / 2), arrowstyle="-", lw=1.4,
                 color=GREY, zorder=7))
    ax.add_patch(FancyArrowPatch((loop_x, chair_y + chair_h / 2),
                 (loop_x, board_y + board_h / 2), arrowstyle="-", lw=1.4,
                 color=GREY, zorder=7, linestyle=(0, (3, 2))))
    _arr(ax, (loop_x, board_y + board_h / 2), (board_x + board_w, board_y + board_h / 2),
         GREY, ls=(0, (3, 2)))
    ax.text(loop_x - 0.8, chair_y + chair_h + 3.0, "revise until\nconverged",
            ha="right", va="center", fontsize=7.2, color=GREY, style="italic",
            linespacing=1.3)

    # ================= DECISION AND SAFETY =================
    y3 = 5.5
    cal_x, cal_w = 7, 30
    _panel(ax, cal_x, y3, cal_w, 12.5, "Calibration and abstention gate", RED, RED_L)
    _reliability(ax, cal_x + 2, y3 + 2.6, 10, 6.3, RED)
    _fork(ax, cal_x + 21, y3 + 5.8, 3.3, RED, WHITE)
    ax.text(cal_x + 7, y3 + 1.9, "fitted temperature", ha="center", va="top",
            fontsize=6.6, color=RED, style="italic")
    ax.text(cal_x + 21, y3 + 1.9, "margin gate", ha="center", va="top",
            fontsize=6.6, color=RED, style="italic")

    guide_x = cal_x + cal_w + 6
    _box(ax, guide_x, y3 + 2.5, 22, 8, "Lung RADS\nguideline mapping", RED_L, RED, fs=8.6)
    _arr(ax, (cal_x + cal_w, y3 + 6.5), (guide_x, y3 + 6.5), RED)

    out_x = guide_x + 22 + 6
    out_w = band_x + band_w - out_x
    _box(ax, out_x, y3, out_w, 12.5, "", fc=WHITE, ec=INK, lw=1.3)
    ax.plot([out_x + 1, out_x + out_w - 1], [y3 + 6.1, y3 + 6.1], color="#CFCFCF",
            lw=1.0, zorder=5)
    ax.text(out_x + out_w / 2, y3 + 9.7,
            "Answer: label + calibrated\nprobabilities + evidence trace + guideline",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=INK,
            zorder=6, linespacing=1.4)
    ax.text(out_x + out_w / 2, y3 + 3.0,
            "Abstain: escalate case to\nhuman radiologist review",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=PURPLE,
            zorder=6, linespacing=1.4)
    _arr(ax, (guide_x + 22, y3 + 8.5), (out_x, y3 + 9.5), RED)
    # the abstain branch is already shown inside the output box itself, so no
    # second arrow is drawn from the margin gate to it (fewer, clearer arrows)

    # chair -> calibration/abstention: clean right-angle route through the gap
    _route(ax, (board_x + board_w / 2, chair_y), (cal_x + cal_w / 2, y3 + 12.5),
          via_y=21, color=GREEN)

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)




# --------------------------------------------------------------------------- #
#  Agent interaction and calibration flow (Figure 2)
# --------------------------------------------------------------------------- #
def render_flow(out_path: Path) -> None:
    """Render the agent-interaction and calibration-flow diagram.

    Complements the layered architecture figure: that one shows *what* the
    components are, this one shows *when* each runs, where the debate loop
    closes, and in what order the calibration and abstention decisions apply.
    Every connector is orthogonal and every edge is a single forward step,
    except the one feedback edge that closes the debate loop.
    """
    fig, ax = plt.subplots(figsize=(15.0, 10.0))
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(75, 97.5, "Agent interaction and calibration flow",
            ha="center", va="center", fontsize=15.5, fontweight="bold", color=INK)

    # ---------------- case in, triage out ----------------
    _box(ax, 57, 87, 36, 7.5, "localized report $R$, measurements $M$",
         fc=GREY_L, ec=GREY, fs=9.2, title="Case", title_fs=8.8)
    _box(ax, 57, 74, 36, 8.6, "is the morphology unambiguous?",
         fc=BLUE_L, ec=BLUE, fs=9.2, title="Triage router", title_fs=8.8)
    _arr(ax, (75, 87), (75, 82.6), color=GREY)

    # ---------------- easy path ----------------
    _panel(ax, 5, 34, 42, 33, "Easy path", GREEN, "#F4FAF5")
    _agent_node(ax, 26, 57.0, 3.6, "Single specialist", "", GREEN, GREEN_L)
    ax.text(26, 51.3, "1 round, no debate", ha="center", va="top",
            fontsize=7.6, color=GREY, style="italic", zorder=6)
    _box(ax, 8.5, 36.5, 35, 7.6,
         "verifier, calibration and guideline\nagent still run on this path",
         fc=WHITE, ec=GREEN, fs=8.6)

    # ---------------- debate loop ----------------
    _panel(ax, 53, 20, 92, 47,
           "Doctor Board debate loop:  rounds $t = 1 \\ldots N$,  $N = 4$",
           PURPLE, "#F7F5FC")
    roles = [("Radiologist", "$w = 1.0$", 67.0),
             ("Pathologist", "$w = 1.0$", 90.5),
             ("Oncologist", "$w = 1.2$", 114.0),
             ("Pulmonologist", "$w = 0.9$", 134.5)]
    for label, sub, cx in roles:
        _agent_node(ax, cx, 60.0, 3.1, label, "", PURPLE, PURPLE_L)
        ax.text(cx, 54.6, sub, ha="center", va="top", fontsize=7.6,
                color=GREY, zorder=6)

    _box(ax, 57.5, 44.0, 83, 7.5,
         "every cited feature graded against $R$:  supported,  unmentioned "
         "($-0.05$),  contradicted ($-0.20$)",
         fc=WHITE, ec=ORANGE, fs=8.8, title="Evidence Verifier", title_fs=8.8)
    _box(ax, 57.5, 33.5, 52, 7.5,
         "leading label $\\ell_t$,  agreement $\\alpha_t$",
         fc=WHITE, ec=BLUE, fs=9.0, title="Chair", title_fs=8.8)
    _box(ax, 113.5, 33.5, 27, 7.5, "counter-case,  $w = 0.6$",
         fc=WHITE, ec=RED, fs=8.8, title="Devil's-Advocate", title_fs=8.8)
    _box(ax, 57.5, 22.0, 83, 7.0,
         "$\\alpha_t = 1$ at $t = 1$,   or   $\\alpha_t \\geq \\tau = 0.75$ "
         "and $\\ell_t = \\ell_{t-1}$",
         fc=PURPLE_L, ec=PURPLE, fs=9.0, title="Convergence test", title_fs=8.8)

    # orthogonal connectors only
    _arr(ax, (99, 54.8), (99, 51.5), color=PURPLE)
    _arr(ax, (83.5, 44.0), (83.5, 41.0), color=PURPLE)
    _arr(ax, (109.5, 37.25), (113.5, 37.25), color=RED)
    _arr(ax, (83.5, 33.5), (83.5, 29.0), color=PURPLE)
    _arr(ax, (127, 33.5), (127, 29.0), color=RED)

    # the one feedback edge
    for a, b, style in [((57.5, 25.5), (55.2, 25.5), "-"),
                        ((55.2, 25.5), (55.2, 60.0), "-"),
                        ((55.2, 60.0), (63.6, 60.0), "-|>")]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14,
                     linewidth=1.6, color=PURPLE, zorder=8,
                     shrinkA=0, shrinkB=0))
    ax.text(54.0, 43.0, "no:  round $t+1$", rotation=90, ha="center",
            va="center", fontsize=8.4, color=PURPLE, style="italic")

    # triage branches
    _route(ax, (57, 78.3), (26, 67), via_y=71.0, color=GREEN)
    _route(ax, (93, 78.3), (99, 67), via_y=71.0, color=PURPLE)
    ax.text(40, 72.0, "unambiguous", ha="center", va="bottom", fontsize=8.6,
            color=GREEN, style="italic")
    ax.text(112, 72.0, "ambiguous", ha="center", va="bottom", fontsize=8.6,
            color=PURPLE, style="italic")

    # ---------------- calibration and abstention spine ----------------
    spine = [
        (5, "Weighted aggregation",
         "$\\bar P \\propto \\sum_i c_i\\, w_{r(i)}\\, P_i$", BLUE, BLUE_L),
        (34, "Temperature scaling",
         "$\\tilde P \\propto \\exp(\\log \\bar P / T)$\n$T$ fitted on a 30% "
         "held-out split", BLUE, BLUE_L),
        (63, "Abstention gate",
         "margin $< 0.10$  or\ntop probability $< 0.40$", ORANGE, ORANGE_L),
        (92, "False-negative guard",
         "malignant mass in the\nDevil's-Advocate $\\geq 0.60$", RED, RED_L),
        (121, "Review packet",
         "label, $\\tilde P$, abstain flag,\nLung-RADS category,\nevidence "
         "trace", GREEN, GREEN_L),
    ]
    for x, title, text, col, light in spine:
        _box(ax, x, 3.5, 24, 11.5, text, fc=light, ec=col, fs=8.6,
             title=title, title_fs=8.8)
    for x in (5, 34, 63, 92):
        _arr(ax, (x + 24, 9.2), (x + 29, 9.2), color=INK, lw=1.6)

    # both triage paths enter the shared calibration spine
    _arr(ax, (17, 34), (17, 15.0), color=GREEN)
    _route(ax, (70, 22.0), (26, 15.0), via_y=18.0, color=PURPLE)
    ax.text(81, 18.6, "yes:  stop", ha="center", va="bottom", fontsize=8.4,
            color=PURPLE, style="italic")

    ax.text(75, 0.6,
            "One feedback edge closes the debate loop; every other edge is a "
            "single forward step. Both triage paths share the calibration "
            "and abstention spine.",
            ha="center", va="bottom", fontsize=8.6, color=GREY, style="italic")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figs = Path(__file__).parent / "figures"
    render(figs / "architecture.png")
    render_flow(figs / "agent-flow.png")
    print("wrote figures/architecture.png and figures/agent-flow.png")
