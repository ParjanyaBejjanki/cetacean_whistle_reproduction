"""Command line interface:  python whistle.py <command> ...   (see README.md)"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import audio
from .config import PROJECT_ROOT, load_detector_config, load_pipeline_config, project_path
from .pipeline import STAGE_ORDER, RunOptions, run_pipeline


def _select_files(cfg, args) -> list[Path]:
    raw = project_path(cfg, "raw_dir")
    allf = audio.list_raw_files(raw)
    if args.files:
        names = {Path(f).name for f in args.files}
        sel = [p for p in allf if p.name in names or p.stem in {Path(f).stem for f in args.files}]
        if len(sel) != len(names):
            raise SystemExit(f"could not resolve all --files in {raw}")
        return sel
    which = args.set
    if which == "all":
        return allf
    k = {"val1": cfg["validation"]["phase1"], "val5": cfg["validation"]["phase2"], "val10": cfg["validation"]["phase3"]}[which]
    return audio.select_validation_files(allf, k)


def cmd_run(args) -> int:
    cfg = load_pipeline_config(args.config, args.override)
    det_cfg = load_detector_config()
    files = _select_files(cfg, args)
    opts = RunOptions(run_id=args.run_id, files=files, first=args.first, last=args.last, storage_mode=args.storage,
                      from_run=Path(args.from_run) if args.from_run else None, device=args.device,
                      verify_hash=not args.no_verify_hash, fail_fast=args.fail_fast, note=args.note or "")
    summary = run_pipeline(cfg, opts, det_cfg=det_cfg)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("files",)}, indent=2))
    return 0 if summary["failed"] == 0 else 2


def cmd_validate_run(args) -> int:
    from .checks import verify_run

    cfg = load_pipeline_config(args.config)
    r = verify_run(project_path(cfg, "runs_dir") / args.run_id)
    print(json.dumps({k: v for k, v in r.items() if k != "per_file"}, indent=2))
    return 0 if r["pass"] else 2


def cmd_report(args) -> int:
    """Compact per-file table for a finished run (reads logs/stage_stats.jsonl; writes nothing)."""
    cfg = load_pipeline_config(args.config)
    d = project_path(cfg, "runs_dir") / args.run_id
    print(f"{'file':34s} {'s/file':>6s} {'frames':>6s} {'%>tau':>6s} {'tracks':>6s} {'rej':>4s} {'vec':>4s} {'imgs':>4s} {'s8rej':>5s}")
    for line in open(d / "logs" / "stage_stats.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["status"] != "OK":
            print(f"{r['stem']:34s} FAILED {r.get('error')}")
            continue
        g = r["diag"]
        print(f"{r['stem']:34s} {sum(v for k, v in r['timings_s'].items() if k != 'verify_sha256'):6.1f} {g['s4']['out_shape'][0] if 's4' in g else '-':>6} "
              f"{100 * g['s5']['frac_nodes_above_tau'] if 's5' in g else float('nan'):6.2f} {g.get('s6', {}).get('n_tracks', '-'):>6} {g.get('s6', {}).get('n_rejected', '-'):>4} "
              f"{g.get('s7', {}).get('n_vectors', '-'):>4} {g.get('s9', {}).get('n_images', '-'):>4} {g.get('s8', {}).get('n_rejected', '-'):>5}")
    return 0


def cmd_list_sets(args) -> int:
    cfg = load_pipeline_config(args.config)
    allf = audio.list_raw_files(project_path(cfg, "raw_dir"))
    for name, k in (("val1", cfg["validation"]["phase1"]), ("val5", cfg["validation"]["phase2"]), ("val10", cfg["validation"]["phase3"])):
        print(name, [p.name for p in audio.select_validation_files(allf, k)])
    print("all", len(allf), "files")
    return 0


def cmd_build_backgrounds(args) -> int:
    from .synth.backgrounds import build_all_banks

    cfg = load_pipeline_config(args.config)
    dcfg = load_detector_config()
    build_all_banks(cfg, dcfg)
    return 0


def cmd_train(args) -> int:
    from .train import train_detector

    cfg = load_pipeline_config(args.config)
    dcfg = load_detector_config(overrides=args.override)
    train_detector(cfg, dcfg, version=args.version)
    return 0


def cmd_eval_synth(args) -> int:
    from .evaluate import evaluate_on_synthetic

    cfg = load_pipeline_config(args.config, args.override)
    dcfg = load_detector_config()
    evaluate_on_synthetic(cfg, dcfg, version=args.version, split=args.split, calibrate=args.calibrate)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="whistle", description=__doc__)
    ap.add_argument("--config", default=None, help="pipeline config TOML (default configs/pipeline.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline (or a stage range) on a file set")
    r.add_argument("--run-id", required=True, help="new run folder name under runs/ (never overwritten)")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--set", choices=["val1", "val5", "val10", "all"], default="val1")
    g.add_argument("--files", nargs="+", help="raw file names or stems")
    r.add_argument("--first", choices=STAGE_ORDER, default="s1")
    r.add_argument("--last", choices=STAGE_ORDER, default="s9")
    r.add_argument("--storage", choices=["debug", "normal", "minimal"], default=None)
    r.add_argument("--from-run", default=None, help="previous run holding the intermediate before --first")
    r.add_argument("--device", default=None, help="auto|cpu|cuda for S5")
    r.add_argument("--override", action="append", default=[], help="config override key=value (recorded in the run)")
    r.add_argument("--no-verify-hash", action="store_true")
    r.add_argument("--fail-fast", action="store_true")
    r.add_argument("--note", default="")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("validate-run", help="automated lineage/output/consistency checks of a finished run")
    v.add_argument("--run-id", required=True)
    v.set_defaults(fn=cmd_validate_run)

    rp = sub.add_parser("report", help="compact per-file table for a finished run")
    rp.add_argument("--run-id", required=True)
    rp.set_defaults(fn=cmd_report)

    s = sub.add_parser("list-sets", help="show the deterministic validation file sets")
    s.set_defaults(fn=cmd_list_sets)

    b = sub.add_parser("build-backgrounds", help="extract real-background banks for synthetic training data")
    b.set_defaults(fn=cmd_build_backgrounds)

    t = sub.add_parser("train", help="train the S5 detector on synthetic data")
    t.add_argument("--version", default="ridgenet-v1")
    t.add_argument("--override", action="append", default=[], help="detector config override")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval-synth", help="evaluate/calibrate on SYNTHETIC validation/test data")
    e.add_argument("--version", default="ridgenet-v1")
    e.add_argument("--split", choices=["val", "test"], default="test")
    e.add_argument("--calibrate", action="store_true", help="write the operating threshold (use split=val)")
    e.add_argument("--override", action="append", default=[])
    e.set_defaults(fn=cmd_eval_synth)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
