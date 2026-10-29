"""Render the MOLT architecture graphic for README.md."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_name("architecture.png")
BG = "#0b1220"
PANEL = "#132033"
PANEL_ALT = "#18283b"
ACCENT = "#5eead4"
TEXT = "#e6edf5"
MUTED = "#9fb0c5"
EDGE = "#6baea8"

fig, ax = plt.subplots(figsize=(16, 8.5), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 8.5)
ax.axis("off")


def node(x, y, w, h, title, body, fill=PANEL, edge=ACCENT, body_size=8.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.14",
        linewidth=1.3, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + 0.14, y + h - 0.20, title, color=ACCENT, fontsize=8.7,
            fontweight="bold", family="DejaVu Sans", va="top")
    ax.text(x + 0.14, y + h - 0.49, body, color=TEXT, fontsize=body_size,
            family="DejaVu Sans", va="top", linespacing=1.35)


def arrow(start, end, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.6,
        color=EDGE, connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=5,
    ))

ax.text(0.45, 8.18, "MOLT  /  EXPERIMENT ARCHITECTURE", color=ACCENT,
        fontsize=11, fontweight="bold", family="DejaVu Sans", va="top")
ax.text(0.45, 7.86, "Credit-card fraud evaluation · semi-synthetic drift only",
        color=MUTED, fontsize=9.5, family="DejaVu Sans", va="top")

# Six accurate data and calibration stages.
top_y, top_h, w = 5.75, 1.5, 2.35
xs = [0.35, 3.00, 5.65, 8.30, 10.95, 13.60]
labels = [
    ("01  SOURCE", "creditcard.csv\nlocal input only"),
    ("02  TIME SPLIT", "Chronological train /\ncalibration / stream\nTime excluded"),
    ("03  MATCHED SAMPLE", "All pools: every fraud\n+ 25% legitimate\nseeded, no replacement"),
    ("04  NULL CALIBRATION", "No-drift PSI / KS limits\nleave-one-seed-out\nfull-run thresholds"),
    ("05  INJECTED SHIFT", "Abrupt / gradual harmful\nbenign mixed covariate\nseed-varied start"),
    ("06  WINDOW REPLAY", "Labels after scoring\nwindow metrics + events\npost-horizon drain"),
]
for i, (title, body) in enumerate(labels):
    node(xs[i], top_y, w, top_h, title, body, PANEL_ALT if i % 2 else PANEL, body_size=7.7)
    if i < len(xs) - 1:
        arrow((xs[i] + w, top_y + top_h / 2), (xs[i + 1], top_y + top_h / 2))

# Detector, policy, and evaluation branches.
node(1.05, 3.35, 4.05, 1.32, "DRIFT DETECTORS", "PSI / KS on features and scores\nADWIN: fraud misses + row-error baseline", PANEL_ALT)
node(5.95, 3.35, 4.15, 1.32, "MODEL POLICIES", "Static · periodic · challenger promotion\nRiver online · zero-delay reference")
node(10.95, 3.35, 4.05, 1.32, "DECISION METRICS", "Applied-threshold recall / realized FPR / FP\nRecall@1% FPR · PR-AUC · amount-unit cost", PANEL_ALT)
arrow((14.75, top_y), (3.05, 4.67), rad=0.04)
arrow((14.95, top_y), (7.95, 4.67), rad=0.02)
arrow((5.10, 4.01), (5.95, 4.01))
arrow((10.10, 4.01), (10.95, 4.01))

# Results and application.
node(5.95, 1.35, 4.15, 1.18, "PRECOMPUTED OUTPUTS", "Window Parquet · run summaries · event log\nDetector trade-offs · block-bootstrap intervals", PANEL_ALT, body_size=7.8)
node(0.35, 0.25, 4.55, 1.00, "DASHBOARD INPUTS", "Small licensed sample + saved batch model", PANEL_ALT, body_size=8.0)
node(11.05, 0.25, 4.50, 1.00, "STREAMLIT RESEARCH DASHBOARD", "Headline · drift monitor · comparison · replay", body_size=8.0)
arrow((12.95, 3.35), (9.85, 2.53), rad=0.02)
arrow((10.10, 1.94), (11.05, 0.75))
arrow((4.90, 0.75), (11.05, 0.75))

fig.savefig(OUT, dpi=170, facecolor=BG, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print(OUT)
