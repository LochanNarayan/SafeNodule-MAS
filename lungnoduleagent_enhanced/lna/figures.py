"""Paper figures from a results/experiments.json payload.

Uses a non-interactive matplotlib backend so it runs headless. Each figure is
saved as PNG (300 dpi) into the given directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REGIMES = ["clean", "noisy", "borderline", "imbalanced", "adversarial"]


def _bar(ax, labels, values, errs=None, title="", ylabel="", rot=30):
    x = range(len(labels))
    ax.bar(x, values, yerr=errs, capsize=3, color="#4C72B0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=rot, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def fig_accuracy_by_regime(payload: Dict, out: Path) -> None:
    regimes = [r for r in REGIMES if r in payload["regimes"]]
    acc = [payload["regimes"][r]["full"]["mean"]["accuracy_all"] for r in regimes]
    std = [payload["regimes"][r]["full"]["std"]["accuracy_all"] for r in regimes]
    cov = [payload["regimes"][r]["full"]["mean"]["coverage"] for r in regimes]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = range(len(regimes))
    ax.bar([i - 0.2 for i in x], acc, width=0.4, yerr=std, capsize=3,
           label="accuracy (all cases)", color="#4C72B0")
    ax.bar([i + 0.2 for i in x], cov, width=0.4, label="answer coverage",
           color="#DD8452")
    ax.set_xticks(list(x)); ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Full system across evaluation regimes")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "accuracy_by_regime.png", dpi=300)
    plt.close(fig)


def fig_calibration(payload: Dict, out: Path) -> None:
    regimes = [r for r in REGIMES if r in payload["regimes"]]
    raw = [payload["regimes"][r]["full"]["mean"]["ece_raw"] for r in regimes]
    cal = [payload["regimes"][r]["full"]["mean"]["ece"] for r in regimes]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = range(len(regimes))
    ax.bar([i - 0.2 for i in x], raw, width=0.4, label="ECE before scaling",
           color="#C44E52")
    ax.bar([i + 0.2 for i in x], cal, width=0.4, label="ECE after scaling",
           color="#55A868")
    ax.set_xticks(list(x)); ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Expected calibration error: temperature scaling effect")
    ax.set_ylabel("ECE"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "calibration_ece.png", dpi=300)
    plt.close(fig)


def fig_cost(payload: Dict, out: Path) -> None:
    regimes = [r for r in REGIMES if r in payload["regimes"]]
    full = [payload["regimes"][r]["full"]["mean"]["mean_llm_calls"] for r in regimes]
    norouter = [payload["regimes"][r]["ablation"].get("- router", {})
                .get("mean", {}).get("mean_llm_calls", 0.0) for r in regimes]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = range(len(regimes))
    ax.bar([i - 0.2 for i in x], norouter, width=0.4, label="always full board",
           color="#8172B2")
    ax.bar([i + 0.2 for i in x], full, width=0.4, label="adaptive router",
           color="#4C72B0")
    ax.set_xticks(list(x)); ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Mean LLM calls per case: adaptive routing vs. always-on board")
    ax.set_ylabel("LLM calls / case"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "cost_routing.png", dpi=300)
    plt.close(fig)


def fig_ablation(payload: Dict, out: Path, regime: str = "adversarial") -> None:
    if regime not in payload["regimes"]:
        regime = next(iter(payload["regimes"]))
    abl = payload["regimes"][regime]["ablation"]
    names = list(abl.keys())
    full = abl["full"]["mean"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, key, title in zip(
            axes,
            ["accuracy_all", "false_negative_rate", "aurc"],
            ["accuracy (all)", "false-negative rate", "AURC (risk-coverage)"]):
        vals = [abl[nm]["mean"][key] for nm in names]
        colors = ["#4C72B0" if nm == "full" else "#C44E52" for nm in names]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis(); ax.set_title(title); ax.grid(axis="x", alpha=0.3)
        ax.axvline(full[key], color="#4C72B0", ls="--", lw=1)
    fig.suptitle(f"Leave-one-component-out ablation (regime = {regime})")
    fig.tight_layout(); fig.savefig(out / "ablation_adversarial.png", dpi=300)
    plt.close(fig)


def fig_verifier(payload: Dict, out: Path) -> None:
    regimes = [r for r in REGIMES if r in payload["regimes"]]
    prec = [payload["regimes"][r]["full"]["mean"]["verifier_precision"] for r in regimes]
    rec = [payload["regimes"][r]["full"]["mean"]["verifier_recall"] for r in regimes]
    flags = [payload["regimes"][r]["full"]["mean"].get("mean_hard_flags", 0.0)
             for r in regimes]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = range(len(regimes))
    ax.plot(list(x), prec, "o-", label="flag precision", color="#4C72B0")
    ax.plot(list(x), rec, "s-", label="flag recall", color="#55A868")
    ax.set_xticks(list(x)); ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax.set_title("Evidence verifier: hard-flag precision / recall vs. oracle")
    ax2 = ax.twinx()
    ax2.bar(list(x), flags, alpha=0.15, color="#C44E52")
    ax2.set_ylabel("mean hard flags / case", color="#C44E52")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "verifier_pr.png", dpi=300)
    plt.close(fig)


def make_all_figures(payload: Dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_accuracy_by_regime(payload, out_dir)
    fig_calibration(payload, out_dir)
    fig_cost(payload, out_dir)
    fig_ablation(payload, out_dir)
    fig_verifier(payload, out_dir)
