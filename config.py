"""
Configuration for the public EviFus-CRC release.

Design principles:
1. Use relative paths by default so the repository can be moved across machines.
2. Keep the parser aligned with the feature-level fusion code path.
3. Treat explicit feature paths in the CSV as the primary data source.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def str2bool(v):
    """Robust bool parser for command-line flags."""
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for EviFus-CRC.

    This public release focuses on feature-level multimodal fusion using
    pre-extracted CT and pathology embeddings.
    """
    parser = argparse.ArgumentParser(
        description="EviFus-CRC: uncertainty-aware multimodal survival modeling"
    )

    # ------------------------------------------------------------------
    # Experiment
    # ------------------------------------------------------------------
    parser.add_argument("--experiment-name", type=str, default="EviFusCRC")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    parser.add_argument(
        "--split-csv",
        type=str,
        default="./example_split.csv",
        help="CSV file containing patient-level split information and labels.",
    )
    parser.add_argument(
        "--ct-feature-root",
        type=str,
        default="./features/ct",
        help="Fallback directory for CT features. Used only when explicit CT paths are not provided in the CSV.",
    )
    parser.add_argument(
        "--pathology-feature-root",
        type=str,
        default="./features/pathology",
        help="Fallback directory for pathology features. Used only when explicit pathology paths are not provided in the CSV.",
    )

    parser.add_argument("--patient-id-col", type=str, default="patient_name")
    parser.add_argument("--duration-col", type=str, default="duration")
    parser.add_argument("--event-col", type=str, default="event")
    parser.add_argument("--split-col", type=str, default="split")
    parser.add_argument("--train-tag", type=str, default="train")
    parser.add_argument("--val-tag", type=str, default="val")
    parser.add_argument("--test-tag", type=str, default="test")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", type=str2bool, default=True)

    # ------------------------------------------------------------------
    # Feature / model settings
    # ------------------------------------------------------------------
    parser.add_argument(
        "--ct-feature-dim",
        type=int,
        default=512,
        help="Input dimension of pre-extracted CT embeddings.",
    )
    parser.add_argument(
        "--pathology-feature-dim",
        type=int,
        default=512,
        help="Input dimension of pre-extracted pathology embeddings.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Shared hidden dimension for modality encoders and fusion.",
    )
    parser.add_argument(
        "--num-risk-classes",
        type=int,
        default=2,
        help="Number of evidential risk classes (default: low vs high risk).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Dropout rate used in modality encoders / heads.",
    )

    # ------------------------------------------------------------------
    # Missing-modality robustness
    # ------------------------------------------------------------------
    parser.add_argument(
        "--use-compensation",
        type=str2bool,
        default=True,
        help="Enable uncertainty-constrained compensation for missing modalities.",
    )
    parser.add_argument(
        "--modality-dropout-prob",
        type=float,
        default=0.20,
        help="Probability of randomly dropping one modality during training.",
    )
    parser.add_argument(
        "--missing-value",
        type=float,
        default=0.0,
        help="Fill value used when a modality is missing.",
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "adam", "adamw"],
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="warmup_cosine",
        choices=["none", "warmup", "cosine", "warmup_cosine"],
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=200,
        help="Early stopping patience based on validation C-index.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps for gradient accumulation.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=5.0,
        help="Gradient clipping threshold.",
    )
    parser.add_argument(
        "--use-amp",
        type=str2bool,
        default=True,
        help="Use mixed precision training.",
    )

    # ------------------------------------------------------------------
    # Survival loss
    # ------------------------------------------------------------------
    parser.add_argument(
        "--num-time-bins",
        type=int,
        default=-1,
        help="Number of discrete survival time bins. Use -1 to infer from training data.",
    )
    parser.add_argument(
        "--survival-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the survival ranking loss term.",
    )
    parser.add_argument(
        "--uncertainty-regularization-weight",
        type=float,
        default=0.01,
        help="Weight for uncertainty regularization.",
    )

    # ------------------------------------------------------------------
    # Evaluation / export
    # ------------------------------------------------------------------
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Path to model checkpoint for evaluation.",
    )
    parser.add_argument(
        "--save-predictions",
        type=str2bool,
        default=False,
        help="Whether to export patient-level predictions.",
    )
    parser.add_argument(
        "--prediction-filename",
        type=str,
        default="predictions.csv",
    )

    return parser


def finalize_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    Post-process parsed arguments.

    - Convert paths to stringified absolute paths
    - Build run directory
    - Validate key arguments
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.experiment_name}_seed{args.seed}_{timestamp}"

    output_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    args.output_dir = str(output_dir)
    args.run_dir = str(run_dir)

    args.split_csv = str(Path(args.split_csv).expanduser().resolve())
    args.ct_feature_root = str(Path(args.ct_feature_root).expanduser().resolve())
    args.pathology_feature_root = str(Path(args.pathology_feature_root).expanduser().resolve())

    if args.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if args.hidden_dim <= 0:
        raise ValueError("hidden_dim must be a positive integer.")
    if args.ct_feature_dim <= 0 or args.pathology_feature_dim <= 0:
        raise ValueError("Feature dimensions must be positive integers.")
    if not (0.0 <= args.modality_dropout_prob <= 1.0):
        raise ValueError("modality_dropout_prob must be in [0, 1].")
    if args.num_risk_classes < 2:
        raise ValueError("num_risk_classes must be >= 2.")

    return args


def get_config() -> argparse.Namespace:
    """Parse, validate and finalize runtime configuration."""
    parser = build_parser()
    args = parser.parse_args()
    args = finalize_config(args)
    return args


def save_config(args: argparse.Namespace, filename: str = "config.json") -> None:
    save_path = Path(args.run_dir) / filename
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


def print_config(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("EviFus-CRC configuration")
    print("=" * 80)
    for key, value in sorted(vars(args).items()):
        print(f"{key:35s}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    cfg = get_config()
    print_config(cfg)
    save_config(cfg)
