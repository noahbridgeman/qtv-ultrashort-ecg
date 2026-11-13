"""Shared thesis plotting style.

The single definition of the print-oriented Matplotlib style, the text width
and the colour palette used by every figure in this project — the scripts in
``plot_scripts/`` and the notebooks in ``notebooks/evaluation/`` alike.

Deliberately dependency-free beyond Matplotlib: importing it must not pull in
the pipeline (and with it neurokit2), so a notebook can style its plots
without loading any of the measurement code.

    from src.tests.style import set_thesis_style, save, TEXT_WIDTH_IN, BLUE, ORANGE

    set_thesis_style()
"""

from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt


# --- layout ---------------------------------------------------------------
# Width of a full-width figure, in inches, matching a typical A4 \textwidth.
# Set this to the value \the\textwidth reports in the thesis document.
TEXT_WIDTH_IN = 6.2

# Colourblind-safe qualitative palette (Wong 2011).
BLUE = "#0173B2"
ORANGE = "#DE8F05"
GREEN = "#029E73"
GREY = "#666666"

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"


def set_thesis_style(use_tex: bool = False) -> None:
    """Apply a consistent, print-oriented Matplotlib style.

    Args:
        use_tex: Render text with a local LaTeX installation. Gives an exact
            match with the thesis body font, but requires LaTeX on PATH and is
            considerably slower.
    """
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 0.9,
        "lines.solid_capstyle": "round",
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "text.usetex": use_tex,
    })


def save(fig: plt.Figure, name: str, out_dir: Path = FIGURE_DIR) -> None:
    """Write a figure as PDF and PNG into ``out_dir`` (default: figures/)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    print(f"saved {name}.pdf / .png -> {out_dir}")
