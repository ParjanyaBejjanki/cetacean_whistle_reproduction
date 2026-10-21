"""Command line interface:  python whistle.py <command> ...   (see README.md)"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import audio
from .config import PROJECT_ROOT, load_detector_config, load_pipeline_config, project_path
from .pipeline import STAGE_ORDER, RunOptions, run_pipeline


def _validation_sets(cfg) -> dict[str, list[Path]]:
    """val1/val5/val10: evenly spaced files of encounter 191 (synth_raw_dir, unchanged so they stay disjoint from the
    synthetic backgrounds); per_encounter: the midpoint file of every configured encounter."""
    v191 = audio.list_raw_files(project_path(cfg, "synth_raw_dir"))
    sets = {name: audio.select_validation_files(v191, cfg["validation"][ph])
            for name, ph in (("val1", "phase1"), ("val5", "phase2"), ("val10", "phase3"))}
    sets["per_encounter"] = [audio.select_validation_files(fs, 1)[0] for fs in audio.files_by_encounter(cfg).values()]
    return sets


def _select_files(cfg, args) -> list[Path]:
    if args.files:
        try:
            return [audio.find_raw_file(cfg, f) for f in args.files]
        except FileNotFoundError as e:
            raise SystemExit(str(e)) from e
    return _validation_sets(cfg)[args.set]


def cmd_run(args) -> int:
    cfg = load_pipeline_config(args.config, args.override)
    det_cfg = load_detector_config()
    files = _select_files(cfg, args)
    if not args.no_verify_hash and (missing := audio.missing_from_manifest(cfg, files)):
        raise SystemExit(f"{len(missing)} files are not in {project_path(cfg, 'manifest_csv')} (e.g. {missing[:3]}), so their "
                         "SHA-256 cannot be verified: run `python whistle.py build-manifest` first (or pass --no-verify-hash)")
    opts = RunOptions(run_id=args.run_id, files=files, first=args.first, last=args.last, storage_mode=args.storage,
                      from_run=Path(args.from_run) if args.from_run else None, device=args.device,
                      verify_hash=not args.no_verify_hash, fail_fast=args.fail_fast, note=args.note or "")
    summary = run_pipeline(cfg, opts, det_cfg=det_cfg, progress=print)
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
    for name, files in _validation_sets(cfg).items():
        print(name, [p.name for p in files])
    for eid, files in audio.files_by_encounter(cfg).items():
        print(f"encounter_{eid}", len(files), "files")
    print("all", len(audio.list_all_raw_files(cfg)), "files")
    return 0


def cmd_build_manifest(args) -> int:
    from .manifest import build_manifest

    build_manifest(load_pipeline_config(args.config))
    return 0


def cmd_build_dataset(args) -> int:
    from .dataset import build_dataset

    cfg = load_pipeline_config(args.config, args.override)
    info = build_dataset(cfg, load_detector_config(), encounters=args.encounters, device=args.device)
    print(json.dumps(info["totals"], indent=2))
    return 0 if info["totals"]["files_failed_unexpected"] == 0 else 2


def cmd_validate_dataset(args) -> int:
    from .dataset import validate_dataset

    cfg = load_pipeline_config(args.config)
    r = validate_dataset(project_path(cfg, "dataset_dir"), cfg)
    print(json.dumps(r, indent=2))
    return 0 if r["pass"] else 2


def cmd_build_backgrounds(args) -> int:
    from .synth.backgrounds import build_all_banks

    cfg = load_pipeline_config(args.config)
    dcfg = load_detector_config()
    build_all_banks(cfg, dcfg)
    return 0


def cmd_build_ridge_masks(args) -> int:
    from .synth.backgrounds import build_ridge_masks

    cfg = load_pipeline_config(args.config)
    dcfg = load_detector_config()
    build_ridge_masks(cfg, dcfg, "train")
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


def cmd_compare_synth(args) -> int:
    from .evaluate import compare_on_synthetic

    cfg = load_pipeline_config(args.config)
    dcfg = load_detector_config()
    bench = PROJECT_ROOT / args.bench_spec if args.bench_spec else None
    compare_on_synthetic(cfg, dcfg, args.versions, args.split, bench)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="whistle", description=__doc__)
    ap.add_argument("--config", default=None, help="pipeline config TOML (default configs/pipeline.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline (or a stage range) on a file set")
    r.add_argument("--run-id", required=True, help="new check-run folder under preprocessing_checks/ (never overwritten)")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--set", choices=["val1", "val5", "val10", "per_encounter"], default="val1",
                   help="check-run file set (the full data is processed by build-dataset, not by run)")
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

    s = sub.add_parser("list-sets", help="show the deterministic validation file sets and files per encounter")
    s.set_defaults(fn=cmd_list_sets)

    m = sub.add_parser("build-manifest", help="hash every raw file into metadata/manifest_all.csv (once; never overwritten)")
    m.set_defaults(fn=cmd_build_manifest)

    bd = sub.add_parser("build-dataset", help="FINAL dataset: preprocessed/encounter_<id>/ (skips complete encounters)")
    bd.add_argument("--encounters", nargs="+", type=int, default=None, help="encounter ids (default: all configured)")
    bd.add_argument("--device", default=None, help="auto|cpu|cuda for S5")
    bd.add_argument("--override", action="append", default=[], help="config override key=value (changes the config hash)")
    bd.set_defaults(fn=cmd_build_dataset)

    vd = sub.add_parser("validate-dataset", help="automated checks of preprocessed/")
    vd.set_defaults(fn=cmd_validate_dataset)

    b = sub.add_parser("build-backgrounds", help="extract real-background banks for synthetic training data")
    b.set_defaults(fn=cmd_build_backgrounds)

    rm = sub.add_parser("build-ridge-masks", help="cache label-free masks of real tonal structure in the training backgrounds")
    rm.set_defaults(fn=cmd_build_ridge_masks)

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

    c = sub.add_parser("compare-synth", help="paired comparison of detector versions on SYNTHETIC data (first = baseline)")
    c.add_argument("--versions", nargs="+", required=True)
    c.add_argument("--split", choices=["val", "test"], default="val")
    c.add_argument("--bench-spec", default=None, help="stored S4 .npy (relative to project) for latency measurement")
    c.set_defaults(fn=cmd_compare_synth)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
