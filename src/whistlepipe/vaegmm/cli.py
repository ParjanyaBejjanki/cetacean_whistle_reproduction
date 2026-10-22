"""VAE-GMM commands of ``python whistle.py`` (registered by whistlepipe.cli). See resources/TRAINING_SPECIFICATION.md."""
from __future__ import annotations

import json

from .config import load_vaegmm_config, resolve

# Experiment matrix of specification §3.8: identical configurations are trained once and reused.
EXPERIMENTS = {
    "V1": {"run_id": "V1_L64_b1e-4_critic", "latent_dim": 64, "beta": 1e-4, "use_critic": True, "for": "main; Tables 1-6, Figs. 7-12"},
    "V2": {"run_id": "V2_L32_b1e-4_critic", "latent_dim": 32, "beta": 1e-4, "use_critic": True, "for": "Table 1"},
    "V3": {"run_id": "V3_L128_b1e-4_critic", "latent_dim": 128, "beta": 1e-4, "use_critic": True, "for": "Table 1"},
    "V4": {"run_id": "V4_L256_b1e-4_critic", "latent_dim": 256, "beta": 1e-4, "use_critic": True, "for": "Table 1"},
    "V5": {"run_id": "V5_L64_b1e-2_critic", "latent_dim": 64, "beta": 1e-2, "use_critic": True, "for": "Table 2"},
    "V6": {"run_id": "V6_L64_b1e-6_critic", "latent_dim": 64, "beta": 1e-6, "use_critic": True, "for": "Table 2"},
    "V7": {"run_id": "V7_L64_b1_nocritic", "latent_dim": 64, "beta": 1.0, "use_critic": False, "for": "Table 4 (1) standard VAE"},
    "V8": {"run_id": "V8_L64_b1e-4_nocritic", "latent_dim": 64, "beta": 1e-4, "use_critic": False, "for": "Table 4 (3)"},
}


def _cfg(args):
    return load_vaegmm_config(args.vaegmm_config, args.override)


def cmd_vae_experiments(args) -> int:
    for k, e in EXPERIMENTS.items():
        print(f"{k}: {json.dumps(e)}")
        print(f"    python whistle.py train-vae --experiment {k}")
    return 0


def cmd_vae_gates(args) -> int:
    from .gates import GATES, TRAINING_GATES

    cfg = _cfg(args)
    names = list(GATES) if args.gate == "all" else (["data", "numerics", "features"] if args.gate == "no-training" else [args.gate])
    results = {n: GATES[n](cfg) for n in names if n not in TRAINING_GATES or args.allow_training}
    skipped = [n for n in names if n in TRAINING_GATES and not args.allow_training]
    if skipped:
        print(f"skipped (they train; pass --allow-training): {skipped}")
    return 0 if all(r["passed"] for r in results.values()) else 1


def cmd_train_vae(args) -> int:
    from .train import run_spec, train_vae

    cfg = _cfg(args)
    e = EXPERIMENTS[args.experiment]
    spec = run_spec(cfg, e["run_id"], e["latent_dim"], e["beta"], e["use_critic"])
    print(json.dumps({"spec": spec, "deviations": cfg.to_dict()["deviations"]}, indent=1), flush=True)
    train_vae(cfg, spec, resume=args.resume)
    return 0


def _run_id(args) -> str:
    return EXPERIMENTS[args.experiment]["run_id"] if args.experiment in EXPERIMENTS else args.experiment


def cmd_fit_gmm(args) -> int:
    from .evaluate import fit_gmm

    cfg = _cfg(args)
    for K in (args.K or [cfg["paper"]["n_components"]]):
        fit_gmm(cfg, _run_id(args), K, args.reg_rel, args.seed, args.tag)
    return 0


def cmd_vae_eval(args) -> int:
    from . import evaluate as ev

    cfg = _cfg(args)
    rid = _run_id(args)
    K = args.K[0] if args.K else None
    if args.what == "recon":
        ev.eval_recon(cfg, rid, args.force)
    elif args.what == "gen":
        ev.eval_generation(cfg, rid, args.source, K or (cfg["paper"]["n_components"] if args.source == "gmm" else None),
                           args.tag, args.ldm_run, args.force)
    elif args.what == "cluster":
        ev.eval_clustering(cfg, rid, args.K or cfg["paper"]["n_components_fig11"], args.force)
    elif args.what == "components":
        ev.analyze_components(cfg, rid, K)
    elif args.what == "conditional":
        ev.eval_conditional(cfg, rid, K, args.force)
    elif args.what == "cost":
        ev.eval_cost(cfg, rid, K, args.force)
    elif args.what == "figures":
        ev.make_figures(cfg, rid, K)
    return 0


def cmd_train_ldm(args) -> int:
    from .evaluate import train_ldm

    cfg = _cfg(args)
    for T in args.T:
        train_ldm(cfg, _run_id(args), T, f"LDM_T{T}_on_{_run_id(args)}", args.resume)
    return 0


def cmd_fetch_eval_weights(args) -> int:
    from .fid import WEIGHT_URLS, download_weights

    if not args.approved:
        print("Downloads (only with --approved, after the project owner's approval):")
        for k, u in WEIGHT_URLS.items():
            print(f"  {k}: {u}")
        return 1
    print(json.dumps(download_weights(resolve("models/eval_weights")), indent=1))
    return 0


def add_commands(sub) -> None:
    def common(p):
        p.add_argument("--vaegmm-config", default=None, help="VAE-GMM config TOML (default configs/vaegmm.toml)")
        p.add_argument("--override", action="append", default=[], help="config override key=value (recorded)")
        return p

    p = common(sub.add_parser("vae-experiments", help="list the VAE experiment matrix V1-V8 (specification §3.8)"))
    p.set_defaults(fn=cmd_vae_experiments)

    p = common(sub.add_parser("vae-gates", help="validation gates G1-G6 (training gates need --allow-training)"))
    p.add_argument("--gate", default="no-training", choices=["data", "numerics", "smoke", "overfit", "timing", "features", "no-training", "all"])
    p.add_argument("--allow-training", action="store_true", help="also run the gates that train (smoke, overfit, timing)")
    p.set_defaults(fn=cmd_vae_gates)

    p = common(sub.add_parser("train-vae", help="train one VAE configuration of the experiment matrix (300 epochs)"))
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    p.add_argument("--resume", action="store_true", help="continue an interrupted run from checkpoints/last.pt")
    p.set_defaults(fn=cmd_train_vae)

    p = common(sub.add_parser("fit-gmm", help="encode training images (unrotated means) and fit the full-covariance GMM"))
    p.add_argument("--experiment", required=True, help="V1..V8 or a run id")
    p.add_argument("--K", type=int, nargs="+", default=None)
    p.add_argument("--reg-rel", type=float, default=None, help="ridge sensitivity check only (default: config)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tag", default="")
    p.set_defaults(fn=cmd_fit_gmm)

    p = common(sub.add_parser("vae-eval", help="evaluation of a finished run: recon (Table 1), gen (Tables 2/4/5), cluster "
                                             "(Fig. 11), components, conditional (Table 3, Fig. 12), cost (Table 6), figures"))
    p.add_argument("what", choices=["recon", "gen", "cluster", "components", "conditional", "cost", "figures"])
    p.add_argument("--experiment", required=True, help="V1..V8 or a run id")
    p.add_argument("--K", type=int, nargs="+", default=None)
    p.add_argument("--source", default="gmm", choices=["gmm", "prior", "ldm"])
    p.add_argument("--tag", default="")
    p.add_argument("--ldm-run", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_vae_eval)

    p = common(sub.add_parser("train-ldm", help="LDM baseline (Table 5) on the latents of a finished VAE run"))
    p.add_argument("--experiment", default="V1")
    p.add_argument("--T", type=int, nargs="+", default=[250, 500, 1000])
    p.add_argument("--resume", action="store_true")
    p.set_defaults(fn=cmd_train_ldm)

    p = common(sub.add_parser("fetch-eval-weights", help="download Inception-V3 / VGG-16 weights (needs --approved)"))
    p.add_argument("--approved", action="store_true")
    p.set_defaults(fn=cmd_fetch_eval_weights)
