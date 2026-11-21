"""Shared plotting setup for the example scripts (monochrome, print-friendly)."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.3,
    "axes.prop_cycle": matplotlib.cycler(color=["#1f2a44", "#c0392b", "#5d7b8a", "#b58900", "#6c5b7b"]),
})


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("saved", os.path.relpath(path, ROOT))
