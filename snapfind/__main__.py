import argparse
from pathlib import Path
import subprocess
import sys

from .config import ROOT, load_config, repo_path


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="config file (default: configs/default.yaml)")
    common.add_argument("--out", help="output folder (default: results/baseline)")
    common.add_argument("--split", help="split.csv to read (default: paths.split in the config)")
    common.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set search.k=20")
    common.add_argument("--force", action="store_true",
                        help="overwrite an existing split.csv or features.npy that differs")

    parser = argparse.ArgumentParser(prog="snapfind")
    stages = parser.add_subparsers(dest="stage")
    stages.required = True  # Python 3.6 på Jetson

    split = stages.add_parser("split", parents=[common], help="write split.csv")
    split.add_argument("--limit", type=int, help="use whole car models until about this many images")

    stages.add_parser("features", parents=[common], help="extract features.npy")

    evaluate = stages.add_parser("evaluate", parents=[common], help="search, score, write results")
    evaluate.add_argument("--variants", nargs="+", default=["A", "B"])
    evaluate.add_argument("--feature-dir", action="append", default=[], metavar="VARIANT=PATH",
                          help="folder with a variant's features.npy, e.g. A=results/baseline_test")

    costs = stages.add_parser("costs", parents=[common], help="encoding time and JPEG size")
    costs.add_argument("--device-label", default="unlabelled", help="e.g. macbook-cpu or aau-gpu")

    everything = stages.add_parser("all", parents=[common], help="split, features, evaluate, costs")
    everything.add_argument("--limit", type=int)
    everything.add_argument("--device-label", default="unlabelled")
    return parser


def run_all(args, out):
    shared = ["--out", out]
    if args.config:
        shared += ["--config", args.config]
    for item in args.set:
        shared += ["--set", item]
    if args.force:
        shared.append("--force")
    split_path = str(Path(out) / "split.csv")

    stages = [
        ["split"] + (["--limit", str(args.limit)] if args.limit else []),
        ["features", "--split", split_path],
        ["evaluate", "--split", split_path, "--feature-dir", f"A={out}"],
        ["costs", "--split", split_path, "--device-label", args.device_label],
    ]
    for stage in stages:
        print(f"\n== {stage[0]} ==", flush=True)
        subprocess.run([sys.executable, "-m", "snapfind", *stage, *shared], check=True, cwd=ROOT)


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    out = args.out or str(Path(cfg["paths"]["results"]) / "baseline")
    out_dir = repo_path(out)
    split_path = repo_path(args.split or cfg["paths"]["split"])

    if args.stage == "all":
        run_all(args, out)
    elif args.stage == "split":
        from . import split
        split.run(cfg, out_dir, limit=args.limit, force=args.force)
    elif args.stage == "features":
        from . import features
        features.run(cfg, out_dir, split_path, force=args.force)
    elif args.stage == "evaluate":
        from . import evaluate
        feature_dirs = dict(item.split("=", 1) for item in args.feature_dir)
        evaluate.run(cfg, out_dir, split_path, args.variants, feature_dirs)
    elif args.stage == "costs":
        from . import costs
        costs.run(cfg, out_dir, split_path, args.device_label)


if __name__ == "__main__":
    main()
