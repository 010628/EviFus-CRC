"""
Training entry point for EviFus-CRC.

This script implements the full feature-level training loop, including:
- modality dropout for missing-modality robustness,
- composite objective (risk + ranking + uncertainty regularization),
- early stopping based on validation C-index.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import get_config, print_config, save_config
from dataset import build_datasets
from model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def save_time_bins(run_dir: str, train_set) -> None:
    path = Path(run_dir) / "time_bins.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"time_bin_edges": train_set.get_time_bin_edges()}, f, indent=2)


def build_dataloaders(config):
    train_set, val_set, test_set = build_datasets(config)

    common_loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=False,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )
    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def apply_modality_dropout(

    batch: Dict[str, torch.Tensor],
    dropout_prob: float,
    missing_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    if dropout_prob <= 0:
        return batch

    ct = batch["ct"].clone()
    pathology = batch["pathology"].clone()
    presence_mask = batch["presence_mask"].clone()

    batch_size = ct.size(0)
    device = ct.device

    # Innovation note:
    # simulate realistic incomplete-input scenarios only when both modalities
    # are originally present; this improves robustness under missing modalities.
    both_present = (presence_mask[:, 0] > 0) & (presence_mask[:, 1] > 0)
    apply_drop = (torch.rand(batch_size, device=device) < dropout_prob) & both_present

    if not torch.any(apply_drop):
        batch["ct"] = ct
        batch["pathology"] = pathology
        batch["presence_mask"] = presence_mask
        return batch

    drop_choice = torch.randint(low=0, high=2, size=(batch_size,), device=device)

    drop_ct = apply_drop & (drop_choice == 0)
    drop_path = apply_drop & (drop_choice == 1)

    if torch.any(drop_ct):
        ct[drop_ct] = missing_value
        presence_mask[drop_ct, 0] = 0.0

    if torch.any(drop_path):
        pathology[drop_path] = missing_value
        presence_mask[drop_path, 1] = 0.0

    batch["ct"] = ct
    batch["pathology"] = pathology
    batch["presence_mask"] = presence_mask
    return batch


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


class EviFusCriterion(nn.Module):
    """
    Composite objective for multimodal survival modeling.
    """
    def __init__(
        self,
        survival_loss_weight: float = 1.0,
        uncertainty_regularization_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.survival_loss_weight = survival_loss_weight
        self.uncertainty_regularization_weight = uncertainty_regularization_weight

    def pairwise_ranking_loss(
        self,
        risk: torch.Tensor,
        duration: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        risk = risk.view(-1)
        duration = duration.view(-1)
        event = event.view(-1)

        if risk.numel() <= 1:
            return risk.new_tensor(0.0)

        risk_i = risk.unsqueeze(1)
        risk_j = risk.unsqueeze(0)

        time_i = duration.unsqueeze(1)
        time_j = duration.unsqueeze(0)
        event_i = event.unsqueeze(1)

        comparable = (time_i < time_j) & (event_i > 0.5)
        if not torch.any(comparable):
            return risk.new_tensor(0.0)

        # Pairwise ranking: a patient with an earlier observed event should
        # receive a higher predicted risk.
        score_diff = risk_i - risk_j
        loss = F.softplus(-score_diff)[comparable].mean()
        return loss

    def uncertainty_regularization(
        self,
        risk: torch.Tensor,
        label: torch.Tensor,
        u_fused: torch.Tensor,
    ) -> torch.Tensor:
        risk = risk.view(-1)
        label = label.float().view(-1)
        uncertainty = u_fused.view(-1)

        # Innovation note:
        # penalize overconfident errors rather than uncertainty itself.
        prediction_error = torch.abs(risk - label).detach()
        overconfidence = 1.0 - uncertainty
        return (overconfidence * prediction_error).mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        risk = outputs["risk"]
        u_fused = outputs["u_fused"]

        label = batch["label"]
        duration = batch["duration"]
        event = batch["event"]

        # BCELoss is unsafe under autocast when the input is already a probability.
        # Compute this term explicitly in float32.
        device_type = risk.device.type
        # BCELoss on probabilities is kept in float32 for mixed-precision stability.
        with torch.amp.autocast(device_type=device_type, enabled=False):
            risk_loss = F.binary_cross_entropy(risk.float().view(-1),label.float().view(-1),)
        ranking_loss = self.pairwise_ranking_loss(risk, duration, event)
        uncertainty_loss = self.uncertainty_regularization(risk, label, u_fused)

        total_loss = (
            risk_loss
            + self.survival_loss_weight * ranking_loss
            + self.uncertainty_regularization_weight * uncertainty_loss
        )

        return {
            "loss": total_loss,
            "risk_loss": risk_loss.detach(),
            "ranking_loss": ranking_loss.detach(),
            "uncertainty_loss": uncertainty_loss.detach(),
        }


def build_optimizer(config, model: nn.Module):
    if config.optimizer == "adam":
        return Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_scheduler(config, optimizer):
    if config.scheduler == "none":
        return None

    max_epochs = max(int(config.max_epochs), 1)
    warmup_epochs = max(int(config.warmup_epochs), 0)

    def lr_lambda(epoch: int) -> float:
        if config.scheduler == "warmup":
            if warmup_epochs == 0:
                return 1.0
            return min((epoch + 1) / warmup_epochs, 1.0)

        if config.scheduler == "cosine":
            progress = epoch / max(max_epochs - 1, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        if config.scheduler == "warmup_cosine":
            if epoch < warmup_epochs and warmup_epochs > 0:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs - 1, 1)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scaler,
    device: torch.device,
    config,
    training: bool,
) -> Dict[str, float]:
    model.train() if training else model.eval()

    loss_meter = AverageMeter()
    risk_loss_meter = AverageMeter()
    ranking_loss_meter = AverageMeter()
    uncertainty_loss_meter = AverageMeter()

    all_risk = []
    all_duration = []
    all_event = []

    accumulation_steps = max(int(config.gradient_accumulation_steps), 1)

    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        batch = move_batch_to_device(batch, device)

        if training and config.modality_dropout_prob > 0:
            # Modality dropout is part of the robustness training strategy
            # described in the manuscript.

            batch = apply_modality_dropout(
                batch=batch,
                dropout_prob=config.modality_dropout_prob,
                missing_value=config.missing_value,
            )

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type,enabled=(config.use_amp and device.type == "cuda"),):
                outputs = model(batch)
                loss_dict = criterion(outputs, batch)
                loss = loss_dict["loss"] / accumulation_steps

            if training:
                scaler.scale(loss).backward()

                should_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(dataloader))
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = batch["ct"].size(0)
        loss_meter.update(loss_dict["loss"].item(), batch_size)
        risk_loss_meter.update(loss_dict["risk_loss"].item(), batch_size)
        ranking_loss_meter.update(loss_dict["ranking_loss"].item(), batch_size)
        uncertainty_loss_meter.update(loss_dict["uncertainty_loss"].item(), batch_size)

        all_risk.append(outputs["risk"].detach().cpu().numpy().reshape(-1))
        all_duration.append(batch["duration"].detach().cpu().numpy().reshape(-1))
        all_event.append(batch["event"].detach().cpu().numpy().reshape(-1))

    all_risk = np.concatenate(all_risk, axis=0)
    all_duration = np.concatenate(all_duration, axis=0)
    all_event = np.concatenate(all_event, axis=0)

    c_index = concordance_index(
        risk=all_risk,
        duration=all_duration,
        event=all_event,
    )

    return {
        "loss": loss_meter.avg,
        "risk_loss": risk_loss_meter.avg,
        "ranking_loss": ranking_loss_meter.avg,
        "uncertainty_loss": uncertainty_loss_meter.avg,
        "c_index": c_index,
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_cindex: float,
    config,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_cindex": best_val_cindex,
        "config": vars(config),
    }
    torch.save(checkpoint, path)


def main() -> None:
    config = get_config()
    print_config(config)
    save_config(config)

    set_seed(config.seed)

    device = torch.device(
        f"cuda:{config.gpu_id}" if (config.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    train_set, val_set, test_set, train_loader, val_loader, test_loader = build_dataloaders(config)
    save_time_bins(config.run_dir, train_set)

    model = build_model(config).to(device)
    criterion = EviFusCriterion(
        survival_loss_weight=config.survival_loss_weight,
        uncertainty_regularization_weight=config.uncertainty_regularization_weight,
    ).to(device)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler(device = device.type,enabled = (config.use_amp and device.type == "cuda"),)

    best_val_cindex = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    best_ckpt_path = str(Path(config.run_dir) / "best_model.pt")
    last_ckpt_path = str(Path(config.run_dir) / "last_model.pt")

    print("\nStart training...\n")

    for epoch in range(1, config.max_epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            training=True,
        )

        val_metrics = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            training=False,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{config.max_epochs:03d}] | "
            f"LR {current_lr:.2e} | "
            f"Train loss {train_metrics['loss']:.4f} | "
            f"Train C-index {train_metrics['c_index']:.4f} | "
            f"Val loss {val_metrics['loss']:.4f} | "
            f"Val C-index {val_metrics['c_index']:.4f}"
        )

        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_cindex=best_val_cindex,
            config=config,
        )

        if val_metrics["c_index"] > best_val_cindex:
            best_val_cindex = val_metrics["c_index"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_cindex=best_val_cindex,
                config=config,
            )
            print(f"  -> New best model saved at epoch {epoch} (val C-index = {best_val_cindex:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    print(f"\nTraining finished. Best validation C-index: {best_val_cindex:.4f} at epoch {best_epoch}")

    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        config=config,
        training=False,
    )

    print(
        f"\n[Test] Loss {test_metrics['loss']:.4f} | "
        f"C-index {test_metrics['c_index']:.4f}"
    )

    summary_path = Path(config.run_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_cindex": best_val_cindex,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )

    print(f"\nArtifacts saved to: {config.run_dir}")


if __name__ == "__main__":
    main()
