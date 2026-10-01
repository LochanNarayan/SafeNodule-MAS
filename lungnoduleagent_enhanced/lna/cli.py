"""Command-line entry point.

Usage (from the project root, i.e. the folder that contains `lna/`):

    python -m lna.cli demo                         # one case, full trace
    python -m lna.cli run --n 20 --regime noisy    # batch, full metric block
    python -m lna.cli ablation --n 40 --seeds 5    # multi-seed leave-one-out
    python -m lna.cli experiments --n 60 --seeds 10  # full paper matrix -> results/
    python -m lna.cli run --n 20 --backend anthropic --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config import Config
from .data import make_dataset
from .evaluate import (Aggregate, compute_metrics, m_acc_all, m_aurc, m_calls,
                       m_ece, m_fnr, paired_bootstrap, run_ablation_multiseed,
                       run_multiseed)
from .pipeline import LungNoduleAgent

REGIMES = ["clean", "noisy", "borderline", "imbalanced", "adversarial"]


def _cfg(args) -> Config:
    if args.regime == "clean":
        cfg = Config(backend=args.backend, model=args.model, seed=args.seed)
    else:
        cfg = Config.for_regime(args.regime, backend=args.backend,
                                model=args.model, seed=args.seed)
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default="mock", choices=["mock", "anthropic", "openai"])
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--regime", default="clean", choices=REGIMES)


def cmd_demo(args) -> None:
    cfg = _cfg(args)
    agent = LungNoduleAgent(cfg)
    cases = make_dataset(1, seed=args.seed, config=cfg)
    res = agent.diagnose(cases[0])
    case = cases[0]

    print("=" * 70)
    print(f"LungNoduleAgent-Enhanced - single case trace  (regime={cfg.regime})")
    print("=" * 70)
    print("\n[1] NODULE SPOTTER")
    print(f"    experts={cfg.n_experts}  detection IoU vs truth = {res.detection_iou:.3f}")
    print(f"    measurements: long={case.measurements.long_diameter_mm}mm "
          f"short={case.measurements.short_diameter_mm}mm vol={case.measurements.volume_mm3}mm^3")
    print("\n[2] SIMULATED RADIOLOGIST")
    print(f"    report: {res.report}")
    if cfg.feature_noise or cfg.report_dropout:
        print(f"    (oracle density/margin = {case.features.density}/{case.features.margin};"
              f" perceived = {case.perceived.density}/{case.perceived.margin})")
    print(f"\n[0] TRIAGE ROUTER: {res.routing} case"
          + ("  [UNDER-TRIAGED]" if res.under_triaged else ""))
    print(f"\n[3] DOCTOR BOARD ({res.rounds_used} round(s), {res.n_llm_calls} LLM calls)")
    for line in res.notes:
        print(f"    note: {line}")
    print("\n    EVIDENCE VERIFIER trace:")
    for t in res.evidence_trace:
        print(f"      {t}")
    print(f"\n    GUIDELINE agent: {res.guideline}")
    print("\n    calibrated probs: " +
          ", ".join(f"{k}={v:.2f}" for k, v in res.calibrated_probs.items()))
    print("    raw probs:        " +
          ", ".join(f"{k}={v:.2f}" for k, v in res.raw_probs.items()))
    print("\n" + "-" * 70)
    verdict = "ABSTAINED (escalate to human)" if res.abstained else \
              ("CORRECT" if res.correct else "INCORRECT")
    print(f"FINAL: {res.predicted_label}  (true={res.true_label})  "
          f"conf={res.confidence:.2f}  -> {verdict}")
    print("-" * 70)


def cmd_run(args) -> None:
    cfg = _cfg(args)
    agent = LungNoduleAgent(cfg)
    cases = make_dataset(args.n, seed=args.seed, config=cfg)
    frac = cfg.calibration_split_frac
    if frac and frac > 0 and cfg.use_calibration:
        k = max(2, int(round(len(cases) * frac)))
        t = agent.fit_calibration([c for c in cases[:k]])
        print(f"(fitted calibration temperature T={t:.2f} on {k} held-out cases)")
        cases = cases[k:]
    results = agent.diagnose_many(cases)
    for r in results:
        print(r.summary())
    print("\n" + "=" * 70)
    print(f"METRICS (regime={cfg.regime}, n={len(results)}):")
    print("  ", compute_metrics(results))
    print("=" * 70)


def cmd_ablation(args) -> None:
    cfg = _cfg(args)
    seeds = list(range(1, args.seeds + 1))
    table = run_ablation_multiseed(cfg, args.n, seeds)
    print(f"\nLeave-one-component-out ablation  (regime={cfg.regime}, "
          f"n={args.n} cases x {len(seeds)} seeds)")
    hdr = ["variant", "acc(all)", "macroF1", "cover", "FNR", "AURC", "ECE",
           "vP", "vR", "calls"]
    print("=" * 92)
    print(f"{hdr[0]:<20}" + "".join(f"{h:>9}" for h in hdr[1:]))
    print("-" * 92)
    for name, agg in table.items():
        m = agg.mean
        print(f"{name:<20}"
              f"{m['accuracy_all']:>9.3f}{m['macro_f1']:>9.3f}{m['coverage']:>9.3f}"
              f"{m['false_negative_rate']:>9.3f}{m['aurc']:>9.3f}{m['ece']:>9.3f}"
              f"{m['verifier_precision']:>9.2f}{m['verifier_recall']:>9.2f}"
              f"{m['mean_llm_calls']:>9.1f}")
    print("=" * 92)


def cmd_experiments(args) -> None:
    """Full paper matrix: per-regime multi-seed metrics, per-regime ablation,
    and paired significance tests. Writes JSON + CSV to results/."""
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.seeds + 1))
    payload: dict = {"n": args.n, "seeds": seeds, "regimes": {}}

    for regime in REGIMES:
        base = Config() if regime == "clean" else Config.for_regime(regime)
        print(f"\n### regime = {regime} "
              f"(n={args.n} x {len(seeds)} seeds) ###")
        main = run_multiseed(base, args.n, seeds)
        print("  full system:", main)
        abl = run_ablation_multiseed(base, args.n, seeds)
        for name, agg in abl.items():
            print(f"    {name:<20} acc(all)={agg.mean['accuracy_all']:.3f} "
                  f"FNR={agg.mean['false_negative_rate']:.3f} "
                  f"AURC={agg.mean['aurc']:.3f} ECE={agg.mean['ece']:.3f} "
                  f"calls={agg.mean['mean_llm_calls']:.1f}")

        # significance: full vs {no board, no router, no calibration} on this regime
        import copy as _c
        sig = {}
        tests = {
            "board_vs_single": (base, _mk(base, board_roles=base.board_roles[:1]),
                                m_acc_all, "accuracy_all"),
            "router_cost": (base, _mk(base, use_router=False), m_calls, "mean_llm_calls"),
            "calibration_ece": (base, _mk(base, use_calibration=False), m_ece, "ece"),
            "devil_fnr": (base, _mk(base, use_devils_advocate=False), m_fnr,
                          "false_negative_rate"),
        }
        for tname, (ca, cb, fn, _lbl) in tests.items():
            sig[tname] = paired_bootstrap(ca, cb, args.n, seeds, fn,
                                          n_boot=args.boot)
            d = sig[tname]
            print(f"    [sig] {tname:<18} delta={d['delta']:+.3f} "
                  f"95%CI[{d['lo']:+.3f},{d['hi']:+.3f}] p={d['p_value']:.3f}")

        payload["regimes"][regime] = {
            "full": {"mean": main.mean, "std": main.std, "per_seed": main.per_seed},
            "ablation": {k: {"mean": v.mean, "std": v.std} for k, v in abl.items()},
            "significance": sig,
        }

    (out / "experiments.json").write_text(json.dumps(payload, indent=2))
    _write_csv(out / "main_by_regime.csv", payload)
    _write_ablation_csv(out / "ablation_by_regime.csv", payload)
    print(f"\nwrote {out/'experiments.json'}, {out/'main_by_regime.csv'}, "
          f"{out/'ablation_by_regime.csv'}")

    if not args.no_figures:
        try:
            from .figures import make_all_figures
            make_all_figures(payload, out / "figures")
            print(f"wrote figures to {out/'figures'}")
        except Exception as e:  # pragma: no cover
            print(f"(figures skipped: {e})")


def _mk(base: Config, **kw) -> Config:
    import copy as _c
    c = _c.deepcopy(base)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _write_csv(path: Path, payload: dict) -> None:
    keys = ["accuracy", "accuracy_all", "macro_f1", "coverage",
            "false_negative_rate", "aurc", "ece", "ece_raw", "brier", "brier_raw",
            "over_confidence", "verifier_precision", "verifier_recall",
            "mean_llm_calls", "under_triage_rate", "mean_rounds",
            "mean_detection_iou"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["regime"] + keys + [f"{k}_std" for k in keys])
        for regime, blob in payload["regimes"].items():
            m, s = blob["full"]["mean"], blob["full"]["std"]
            w.writerow([regime] + [f"{m[k]:.4f}" for k in keys]
                       + [f"{s[k]:.4f}" for k in keys])


def _write_ablation_csv(path: Path, payload: dict) -> None:
    keys = ["accuracy_all", "macro_f1", "coverage", "false_negative_rate",
            "aurc", "ece", "verifier_precision", "verifier_recall",
            "mean_llm_calls", "mean_rounds"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["regime", "variant"] + keys)
        for regime, blob in payload["regimes"].items():
            for variant, mv in blob["ablation"].items():
                m = mv["mean"]
                w.writerow([regime, variant] + [f"{m[k]:.4f}" for k in keys])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lna", description="LungNoduleAgent-Enhanced")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="run one case with a full trace")
    _add_common(p_demo); p_demo.set_defaults(func=cmd_demo)

    p_run = sub.add_parser("run", help="run a batch and print the full metric block")
    _add_common(p_run); p_run.set_defaults(func=cmd_run)

    p_abl = sub.add_parser("ablation", help="multi-seed leave-one-component-out")
    _add_common(p_abl); p_abl.add_argument("--seeds", type=int, default=5)
    p_abl.set_defaults(func=cmd_ablation)

    p_exp = sub.add_parser("experiments", help="full paper matrix -> results/")
    _add_common(p_exp)
    p_exp.add_argument("--seeds", type=int, default=10)
    p_exp.add_argument("--boot", type=int, default=2000)
    p_exp.add_argument("--out", default="results")
    p_exp.add_argument("--no-figures", action="store_true")
    p_exp.set_defaults(func=cmd_experiments)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
