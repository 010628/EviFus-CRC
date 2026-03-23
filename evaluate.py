
"""
Evaluation script for EviFus-CRC.

Besides standard risk prediction, this script exports the uncertainty- and
trust-related quantities used in the manuscript:
- u_fused
- ct_uncertainty / pathology_uncertainty
- trust_ct / trust_pathology
- conflict
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import get_config, print_config
from dataset import build_datasets
from model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def concordance_index(
    risk: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
) -> float:
    risk = np.asarray(risk, dtype=np.float64).reshape(-1)
    duration = np.asarray(duration, dtype=np.float64).reshape(-1)
    event = np.asarray(event, dtype=np.float64).reshape(-1)

    concordant = 0.0
    tied = 0.0
    permissible = 0.0

    n = len(risk)
    for i in range(n):
        for j in range(i + 1, n):
            if duration[i] == duration[j]:
                if event[i] == 1 and event[j] == 1:
                    permissible += 1
                    if risk[i] == risk[j]:
                        tied += 1
                    elif risk[i] > risk[j] and duration[i] < duration[j]:
                        concordant += 1
                    elif risk[j] > risk[i] and duration[j] < duration[i]:
                        concordant += 1
                continue

            if duration[i] < duration[j] and event[i] == 1:
                permissible += 1
                if risk[i] == risk[j]:
                    tied += 1
                elif risk[i] > risk[j]:
                    concordant += 1

            elif duration[j] < duration[i] and event[j] == 1:
                permissible += 1
                if risk[i] == risk[j]:
                    tied += 1
                elif risk[j] > risk[i]:
                    concordant += 1

    if permissible == 0:
        return 0.5
    return float((concordant + 0.5 * tied) / permissible)


def build_dataloaders_for_evaluation(config):
    train_set, val_set, test_set = build_datasets(config)

    common_loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
        drop_last=False,
    )

    val_loader = DataLoader(val_set, **common_loader_kwargs)
    test_loader = DataLoader(test_set, **common_loader_kwargs)
    return val_set, test_set, val_loader, test_loader


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    """Load a training checkpoint for standalone evaluation."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    return checkpoint


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()

    all_patient_ids: List[str] = []
    all_duration: List[float] = []
    all_event: List[float] = []
    all_label: List[int] = []

    all_risk: List[float] = []
    all_u_fused: List[float] = []
    all_conflict: List[float] = []

    all_ct_uncertainty: List[float] = []
    all_pathology_uncertainty: List[float] = []
    all_trust_ct: List[float] = []
    all_trust_pathology: List[float] = []

    all_ct_present: List[int] = []
    all_pathology_present: List[int] = []

    for batch in dataloader:
        patient_ids = batch["patient_id"]
        batch = move_batch_to_device(batch, device)

        outputs = model(batch)

        # These exported tensors support the uncertainty-guided analyses in the paper.

        risk = outputs["risk"].detach().cpu().numpy().reshape(-1)
        u_fused = outputs["u_fused"].detach().cpu().numpy().reshape(-1)
        conflict = outputs["conflict"].detach().cpu().numpy().reshape(-1)

        ct_uncertainty = outputs["ct_uncertainty"].detach().cpu().numpy().reshape(-1)
        pathology_uncertainty = outputs["pathology_uncertainty"].detach().cpu().numpy().reshape(-1)
        trust_ct = outputs["trust_ct"].detach().cpu().numpy().reshape(-1)
        trust_pathology = outputs["trust_pathology"].detach().cpu().numpy().reshape(-1)

        duration = batch["duration"].detach().cpu().numpy().reshape(-1)
        event = batch["event"].detach().cpu().numpy().reshape(-1)
        label = batch["label"].detach().cpu().numpy().reshape(-1)

        presence_mask = batch["presence_mask"].detach().cpu().numpy()
        ct_present = presence_mask[:, 0].astype(int)
        pathology_present = presence_mask[:, 1].astype(int)

        all_patient_ids.extend([str(x) for x in patient_ids])
        all_duration.extend(duration.tolist())
        all_event.extend(event.tolist())
        all_label.extend(label.tolist())

        all_risk.extend(risk.tolist())
        all_u_fused.extend(u_fused.tolist())
        all_conflict.extend(conflict.tolist())

        all_ct_uncertainty.extend(ct_uncertainty.tolist())
        all_pathology_uncertainty.extend(pathology_uncertainty.tolist())
        all_trust_ct.extend(trust_ct.tolist())
        all_trust_pathology.extend(trust_pathology.tolist())

        all_ct_present.extend(ct_present.tolist())
        all_pathology_present.extend(pathology_present.tolist())

    results_df = pd.DataFrame(
        {
            "patient_id": all_patient_ids,
            "duration": all_duration,
            "event": all_event,
            "label": all_label,
            "predicted_risk": all_risk,
            "u_fused": all_u_fused,
            "conflict": all_conflict,
            "ct_uncertainty": all_ct_uncertainty,
            "pathology_uncertainty": all_pathology_uncertainty,
            "trust_ct": all_trust_ct,
            "trust_pathology": all_trust_pathology,
            "ct_present": all_ct_present,
            "pathology_present": all_pathology_present,
        }
    )

    metrics = {
        "c_index": concordance_index(
            risk=results_df["predicted_risk"].values,
            duration=results_df["duration"].values,
            event=results_df["event"].values,
        ),
        "num_cases": int(len(results_df)),
        "num_events": int(results_df["event"].sum()),
    }

    return metrics, results_df


def save_predictions(results_df: pd.DataFrame, save_path: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_path, index=False)


def save_metrics(metrics: Dict[str, float], save_path: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    config = get_config()
    print_config(config)

    if not config.checkpoint_path:
        raise ValueError("Please provide --checkpoint-path for evaluation.")

    set_seed(config.seed)

    device = torch.device(
        f"cuda:{config.gpu_id}" if (config.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    val_set, test_set, val_loader, test_loader = build_dataloaders_for_evaluation(config)

    model = build_model(config).to(device)
    checkpoint = load_checkpoint(config.checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nRunning validation inference...")
    val_metrics, val_df = run_inference(
        model=model,
        dataloader=val_loader,
        device=device,
    )

    print(
        f"[Validation] cases={val_metrics['num_cases']} | "
        f"events={val_metrics['num_events']} | "
        f"C-index={val_metrics['c_index']:.4f}"
    )

    print("\nRunning test inference...")
    test_metrics, test_df = run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    print(
        f"[Test] cases={test_metrics['num_cases']} | "
        f"events={test_metrics['num_events']} | "
        f"C-index={test_metrics['c_index']:.4f}"
    )

    run_dir = Path(config.output_dir) / "evaluation"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(val_metrics, run_dir / "val_metrics.json")
    save_metrics(test_metrics, run_dir / "test_metrics.json")

    if config.save_predictions:
        save_predictions(val_df, run_dir / f"val_{config.prediction_filename}")
        save_predictions(test_df, run_dir / f"test_{config.prediction_filename}")
        print(f"\nPrediction files saved to: {run_dir}")

    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()
