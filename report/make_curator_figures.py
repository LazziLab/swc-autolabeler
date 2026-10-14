"""Figures for curator feedback (Sec 5.7).

Reads the two experiment outputs and renders two figures:

  curator_loop_seed*.json + flag_feedback.json -> curator_loop.{pdf,png,svg}
      Figure 3, drawn at the paper's text width with 8 pt lettering. Panel a:
      the labeler as curator corrections accumulate, mean over seeds, for
      flag-guided review, random review and flag-guided review by a noisy
      curator, with the released labeler as a reference line. Panel b: the flag
      as files are reviewed with the labeler held fixed, for flag-guided review,
      random review and a flag that is never retrained. The curator
      loop refits a flag as well, but only to choose which cells to review, so
      its own scores are not plotted (see evaluation.active_learning.curator_loop).

  flag_feedback.json -> curator_flag_feedback.{png,svg}
      The flag experiment alone, kept-cell F1 P10 and average precision. Not in
      the paper, whose Figure 3b plots the same P10 curves.

    python -m report.make_curator_figures

Outputs (report/figures/):
    curator_loop.{pdf,png,svg}
    curator_flag_feedback.{png,svg}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(__file__).resolve().parent
AL = ROOT / "evaluation" / "results" / "active_learning"
FIGDIR = REPORT / "figures"

ARMS = {"flag": ("#c0392b", "Flag-guided review"),
        "random": ("#2980b9", "Random review"),
        "factory": ("#7f8c8d", "Never retrained")}
LOOP_ARMS = {"flag": ("#c0392b", "Flag-guided review", "-"),
             "random": ("#2980b9", "Random review", "-"),
             "flag_noisy": ("#d68910", "Flag-guided, noisy curator", "--")}
#: A marker per arm as well as a colour, so the curves stay apart without colour vision.
MARKERS = {"flag": "o", "random": "s", "flag_noisy": "^", "factory": "x"}
#: Figure 3 is drawn at its printed size: the text width, 8 pt sans-serif lettering,
#: fonts embedded in the PDF.
PAPER_STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
               "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
               "legend.fontsize": 8, "pdf.fonttype": 42, "lines.linewidth": 1.0}
TEXT_WIDTH_IN = 5.15


def _flag_feedback_curves(d: dict, src: Path, key: str) -> dict[str, tuple[list[int], list[float]]]:
    """Each arm's curve of ``key`` against the number of files reviewed."""
    curves = {}
    for arm in ARMS:
        rows = d.get(f"{arm}_arm")
        if not rows:
            continue
        # Review is counted in files.
        if any("n_reviewed_files" not in r for r in rows):
            raise SystemExit(f"{src.name}: no review count on the {arm} arm; "
                             f"regenerate it with evaluation.active_learning."
                             f"flag_feedback")
        curves[arm] = ([r["n_reviewed_files"] for r in rows], [r[key] for r in rows])
    return curves


def _flag_feedback_figure() -> Path | None:
    src = AL / "flag_feedback.json"
    if not src.is_file():
        print(f"  skip: {src.name} not found "
              "(run evaluation.active_learning.flag_feedback)")
        return None
    d = json.loads(src.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for key, ylabel, ax in (("kept_pcf1_p10", "Kept-cell F1 P10 @10% reject", axes[0]),
                            ("ap", "Average precision", axes[1])):
        for arm, (xs, ys) in _flag_feedback_curves(d, src, key).items():
            color, label = ARMS[arm]
            ax.plot(xs, ys, marker="o", color=color, markersize=4, label=label,
                    ls="--" if arm == "factory" else "-")
        ax.set_xlabel("Files reviewed by the curator")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Curator review sharpens the flag; the labeler is held fixed", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "curator_flag_feedback.png"
    fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out


def _curator_loop_figure() -> Path | None:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(AL.glob("curator_loop_seed*.json"))
            if p.stem.removeprefix("curator_loop_seed").isdigit()]
    if not runs:
        print("  skip: no curator_loop_seed*.json "
              "(run evaluation.active_learning.curator_loop)")
        return None
    fb_src = AL / "flag_feedback.json"
    if not fb_src.is_file():
        print(f"  skip: {fb_src.name} not found, and Figure 3's right panel is drawn "
              "from it (run evaluation.active_learning.flag_feedback)")
        return None
    runs.sort(key=lambda d: d["seed"])
    arms = [a for a in LOOP_ARMS if all(d.get(f"{a}_arm") for d in runs)]

    def mean_curve(arm: str, key: str) -> tuple[list[int], list[float]]:
        n = min(len(d[f"{arm}_arm"]) for d in runs)
        x = [runs[0][f"{arm}_arm"][i]["n_corrections"] for i in range(n)]
        y = [sum(d[f"{arm}_arm"][i][key] for d in runs) / len(runs) for i in range(n)]
        return x, y

    factory = [d["endpoint_full_pipeline"]["factory"]["neurite_macro_f1"]
               for d in runs if "factory" in (d.get("endpoint_full_pipeline") or {})]
    factory_mean = sum(factory) / len(factory) if len(factory) == len(runs) else None

    left = {arm: mean_curve(arm, "labeler_neurite_macro_f1") for arm in arms}
    right = _flag_feedback_curves(json.loads(fb_src.read_text(encoding="utf-8")), fb_src, "kept_pcf1_p10")

    # Both panels get the same y-axis span, so neither panel's movement is zoomed
    # beyond the other's. Auto-scaling each panel to its own data would blow the
    # labeler's small movement up to the size of the flag's larger one.
    left_vals = [v for _x, ys in left.values() for v in ys]
    if factory_mean is not None:
        left_vals.append(factory_mean)
    right_vals = [v for _x, ys in right.values() for v in ys]
    extents = [(min(left_vals), max(left_vals)), (min(right_vals), max(right_vals))]
    span = 1.35 * max(hi - lo for lo, hi in extents)       # headroom above the wider panel

    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.9))
        ax = axes[0]
        for arm in arms:
            color, label, ls = LOOP_ARMS[arm]
            if arm == "flag_noisy":
                label = f"{label} ({runs[0]['noise_rate']:.0%})"
            x, y = left[arm]
            ax.plot(x, y, ls=ls, marker=MARKERS[arm], color=color, markersize=3.5, label=label)
        if factory_mean is not None:
            ax.axhline(factory_mean, color=ARMS["factory"][0], ls=":", lw=1.4,
                       label="Released labeler, full training split")
        ax.set_xlabel("Curator corrections")
        ax.set_ylabel("Labeler neurite macro-F1")

        ax = axes[1]
        for arm, (xs, ys) in right.items():
            color, label = ARMS[arm]
            ax.plot(xs, ys, marker=MARKERS[arm], color=color, markersize=3.5, label=label,
                    ls="--" if arm == "factory" else "-")
        ax.set_xlabel("Files reviewed by the curator")
        ax.set_ylabel("Flag kept-set P10\n(10% rejected)")

        for ax, (lo, hi), letter in zip(axes, extents, "ab"):
            top = min(1.0, (lo + hi) / 2.0 + span / 2.0)
            ax.set_ylim(top - span, top)
            ax.grid(True, alpha=0.3, lw=0.5)
            ax.text(-0.24, 1.0, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
                    va="bottom", ha="left")
        # One legend for both panels; the two arms they share keep one colour and marker.
        handles, labels = [], []
        for ax in axes:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in labels:
                    handles.append(h)
                    labels.append(lab)
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   handlelength=2.2, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0.25, 1, 1), w_pad=1.6)
        FIGDIR.mkdir(parents=True, exist_ok=True)
        out = FIGDIR / "curator_loop.png"
        fig.savefig(out, dpi=600)
        fig.savefig(out.with_suffix(".svg"))
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
    return out


def main() -> int:
    made = [p for p in (_curator_loop_figure(), _flag_feedback_figure()) if p]
    for p in made:
        print(f"  wrote {p.relative_to(ROOT)} (+ .svg, and .pdf for Figure 3)")
    if not made:
        raise SystemExit("no experiment output found; run the experiments first")
    return 0


if __name__ == "__main__":
    sys.exit(main())
