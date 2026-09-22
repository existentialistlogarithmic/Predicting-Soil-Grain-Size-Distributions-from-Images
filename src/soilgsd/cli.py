"""Command line entry point: ``python -m soilgsd <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .constants import LOG_WIDTHS, MAX_SCORE, SUPPORTS_MM
from .models import MODEL_REGISTRY
from .pipeline import load_settings, make_submission, prepare_features, run_cv
from .validate import validate_submission_file

DEFAULT_CONFIG = Path("configs/default.yaml")


def _settings(args: argparse.Namespace):
    config = Path(args.config) if args.config else (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    return load_settings(
        config,
        data_root=getattr(args, "data_root", None),
        artifacts=getattr(args, "artifacts", None),
        model=getattr(args, "model", None),
        n_splits=getattr(args, "n_splits", None),
    )


def _cmd_describe(args: argparse.Namespace) -> int:
    print("Predicting Soil Grain Size Distributions from Images")
    print(f"  supports (mm): {', '.join(f'{value:g}' for value in SUPPORTS_MM)}")
    print(f"  metric:        mean log-diameter-weighted EMD, range [0, {MAX_SCORE:g}], lower is better")
    print(f"  interval weights: {', '.join(f'{w:.4f}' for w in LOG_WIDTHS)} (sum {LOG_WIDTHS.sum():g})")
    print(f"  models:        {', '.join(sorted(MODEL_REGISTRY))} (or blend:a+b)")
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from .synthetic import generate_dataset

    root = generate_dataset(
        args.out, n_train=args.n_train, n_test=args.n_test, seed=args.seed
    )
    print(f"wrote a synthetic competition folder to {root}")
    print("it mimics the official layout; scores on it mean nothing for the leaderboard")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    settings = _settings(args)
    for split in args.splits:
        frame = prepare_features(settings, split=split, refresh=args.refresh)
        print(f"{split}: {len(frame)} samples x {frame.shape[1] - 1} features")
    return 0


def _cmd_cv(args: argparse.Namespace) -> int:
    settings = _settings(args)
    results = []
    for name in args.models:
        result = run_cv(settings, model=name, refresh=args.refresh)
        print(result.render())
        results.append(result)
    if len(results) > 1:
        best = min(results, key=lambda item: item.score)
        print(f"\nbest: {best.model_name} at {best.score:.4f}")
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    settings = _settings(args)
    path = make_submission(settings, model=args.model, out_path=args.out, refresh=args.refresh)
    print(f"wrote {path}")
    print("upload with: kaggle competitions submit -c soil-grain-size-from-photos "
          f'-f "{path}" -m "<message>"')
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    report = validate_submission_file(args.submission, args.template)
    print(report.render())
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soilgsd", description=__doc__)
    parser.add_argument("--config", help=f"YAML settings file (default: {DEFAULT_CONFIG})")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--data-root", help="folder holding the official competition files")
        sub.add_argument("--artifacts", help="folder for caches, reports and submissions")
        sub.add_argument("--refresh", action="store_true", help="ignore the feature cache")

    describe = subparsers.add_parser("describe", help="print the competition's fixed facts")
    describe.set_defaults(func=_cmd_describe)

    synth = subparsers.add_parser(
        "make-synthetic", help="generate a fake dataset so the pipeline can be run offline"
    )
    synth.add_argument("--out", default="data/synthetic")
    synth.add_argument("--n-train", type=int, default=24)
    synth.add_argument("--n-test", type=int, default=8)
    synth.add_argument("--seed", type=int, default=0)
    synth.set_defaults(func=_cmd_synth)

    features = subparsers.add_parser("features", help="extract and cache photo features")
    add_common(features)
    features.add_argument("--splits", nargs="+", default=["train", "test"])
    features.set_defaults(func=_cmd_features)

    cv = subparsers.add_parser("cv", help="grouped cross-validation against the competition metric")
    add_common(cv)
    cv.add_argument("--models", nargs="+", default=["constant", "ridge", "knn", "gbt"])
    cv.add_argument("--n-splits", type=int)
    cv.set_defaults(func=_cmd_cv)

    submit = subparsers.add_parser("submit", help="fit on all data and write a submission csv")
    add_common(submit)
    submit.add_argument("--model")
    submit.add_argument("--out")
    submit.set_defaults(func=_cmd_submit)

    validate = subparsers.add_parser("validate", help="check a submission csv before uploading")
    validate.add_argument("submission")
    validate.add_argument("--template", help="path to sample_submission.csv")
    validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
