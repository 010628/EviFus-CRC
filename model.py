"""
Core model definition for EviFus-CRC.

The main methodological contributions implemented here are:
1. modality-specific evidential heads (BBA1 / BBA2),
2. two-stage Dempster-Shafer fusion,
3. uncertainty-guided missing-modality compensation.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_float_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)
    return mask.float()


def _safe_uniform_belief(
    batch_size: int,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    belief = torch.zeros(batch_size, num_classes, device=device, dtype=dtype)
    uncertainty = torch.ones(batch_size, 1, device=device, dtype=dtype)
    return belief, uncertainty


def _masked_zero_features(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = _as_float_mask(mask)
    return features * mask


class SoftmaxBBAHead(nn.Module):
    """
    BBA1: normalized belief masses via softmax.

    Innovation note:
    This branch provides a stable probability-style belief assignment that is
    later fused with the evidence-style branch (BBA2) inside each modality.
    """
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, presence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = _as_float_mask(presence_mask)
        logits = self.net(x)
        belief = torch.softmax(logits, dim=-1) * mask
        uncertainty = 1.0 - belief.sum(dim=-1, keepdim=True)
        uncertainty = uncertainty.clamp(min=1e-6, max=1.0)
        return belief, uncertainty


class EvidenceBBAHead(nn.Module):
    """
    BBA2: non-negative evidence mapped to Dirichlet-style belief masses.
    """
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, presence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = _as_float_mask(presence_mask)
        # Innovation note:
        # map raw network output to non-negative evidence and then derive
        # belief mass plus epistemic uncertainty under a Dirichlet-style form.
        evidence = F.softplus(self.net(x)) * mask
        evidence_sum = evidence.sum(dim=-1, keepdim=True)

        uncertainty = self.num_classes / (evidence_sum + self.num_classes + 1e-6)
        belief = evidence / (evidence_sum + self.num_classes + 1e-6)

        belief = belief * mask
        uncertainty = torch.where(mask > 0, uncertainty, torch.ones_like(uncertainty))
        return belief, uncertainty


class DempsterShaferFusion(nn.Module):
    """
    Dempster-Shafer fusion for singleton class beliefs.
    """
    def __init__(self, num_classes: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def combine(
        self,
        belief_a: torch.Tensor,
        uncertainty_a: torch.Tensor,
        belief_b: torch.Tensor,
        uncertainty_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pairwise = torch.matmul(belief_a.unsqueeze(2), belief_b.unsqueeze(1))
        same_class = torch.diagonal(pairwise, dim1=1, dim2=2).sum(dim=-1, keepdim=True)
        total_interaction = pairwise.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
        conflict = (total_interaction - same_class).clamp(min=0.0)

        denominator = (1.0 - conflict).clamp(min=self.eps)

        # Innovation note:
        # Dempster-Shafer fusion explicitly combines belief masses while
        # carrying uncertainty and conflict across modalities.
        fused_belief = (
            belief_a * belief_b
            + belief_a * uncertainty_b
            + belief_b * uncertainty_a
        ) / denominator

        fused_uncertainty = (uncertainty_a * uncertainty_b) / denominator
        fused_uncertainty = fused_uncertainty.clamp(min=self.eps, max=1.0)

        belief_sum = fused_belief.sum(dim=-1, keepdim=True)
        overflow = (belief_sum + fused_uncertainty - 1.0).clamp(min=0.0)
        if torch.any(overflow > 0):
            fused_belief = fused_belief / (belief_sum + self.eps) * (1.0 - fused_uncertainty)

        return fused_belief, fused_uncertainty, conflict

    def fuse_intra_modality(
        self,
        bba1_belief: torch.Tensor,
        bba1_uncertainty: torch.Tensor,
        bba2_belief: torch.Tensor,
        bba2_uncertainty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.combine(
            belief_a=bba1_belief,
            uncertainty_a=bba1_uncertainty,
            belief_b=bba2_belief,
            uncertainty_b=bba2_uncertainty,
        )

    def fuse_inter_modality(
        self,
        modality_outputs: Dict[str, Dict[str, torch.Tensor]],
        modality_presence: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = list(modality_outputs.keys())
        batch_size = modality_outputs[keys[0]]["belief"].shape[0]
        device = modality_outputs[keys[0]]["belief"].device
        dtype = modality_outputs[keys[0]]["belief"].dtype

        fused_belief, fused_uncertainty = _safe_uniform_belief(
            batch_size=batch_size,
            num_classes=self.num_classes,
            device=device,
            dtype=dtype,
        )
        total_conflict = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        has_any_modality = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        for key in keys:
            present = _as_float_mask(modality_presence[key])
            belief = modality_outputs[key]["belief"]
            uncertainty = modality_outputs[key]["uncertainty"]

            first_modality_mask = (has_any_modality == 0) & (present > 0)
            later_modality_mask = (has_any_modality > 0) & (present > 0)

            if first_modality_mask.any():
                idx = first_modality_mask.squeeze(-1)
                fused_belief[idx] = belief[idx]
                fused_uncertainty[idx] = uncertainty[idx]
                has_any_modality[idx] = 1.0

            if later_modality_mask.any():
                idx = later_modality_mask.squeeze(-1)
                fb, fu, fc = self.combine(
                    belief_a=fused_belief[idx],
                    uncertainty_a=fused_uncertainty[idx],
                    belief_b=belief[idx],
                    uncertainty_b=uncertainty[idx],
                )
                fused_belief[idx] = fb
                fused_uncertainty[idx] = fu
                total_conflict[idx] = total_conflict[idx] + fc
                has_any_modality[idx] = 1.0

        return fused_belief, fused_uncertainty, total_conflict


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, presence_mask: torch.Tensor) -> torch.Tensor:
        x = _masked_zero_features(x, presence_mask)
        return self.net(x)


class CompensationModule(nn.Module):
    """
    Generate low-confidence pseudo-evidence for a missing modality based on the
    observed modality, while constraining its influence by uncertainty.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        observed_feature: torch.Tensor,
        observed_uncertainty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([observed_feature, observed_uncertainty], dim=-1)
        pseudo_feature = self.projector(x)

        # Innovation note:
        # the compensation weight is uncertainty-constrained: if the observed
        # modality is itself unreliable, the pseudo feature should have limited influence.
        confidence = 1.0 - observed_uncertainty
        learned_gate = self.gate(x)
        compensation_scale = (confidence * learned_gate).clamp(min=0.0, max=1.0)
        return pseudo_feature, compensation_scale


class RiskMapper(nn.Module):
    def __init__(self, num_classes: int = 2, positive_class_index: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.positive_class_index = positive_class_index

    def forward(self, belief: torch.Tensor, uncertainty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert evidential output to pignistic probability and then use the
        # positive-risk class as the final continuous risk score.
        pignistic = belief + uncertainty / self.num_classes
        risk = pignistic[:, self.positive_class_index].unsqueeze(-1)
        return risk, pignistic


class OptionalCalibrationModule(nn.Module):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = enabled

    def forward(
        self,
        pathology_belief: torch.Tensor,
        pathology_uncertainty: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return pathology_belief, pathology_uncertainty


class EviFusCRC(nn.Module):
    """
    EviFus-CRC: uncertainty-aware multimodal evidential fusion for CRC prognosis.

    Inputs are assumed to be pre-extracted patient-level embeddings:
        - CT embedding:        [B, ct_feature_dim]
        - Pathology embedding: [B, pathology_feature_dim]
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_risk_classes

        self.ct_encoder = ModalityEncoder(
            in_dim=config.ct_feature_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.pathology_encoder = ModalityEncoder(
            in_dim=config.pathology_feature_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        self.ct_bba1 = SoftmaxBBAHead(config.hidden_dim, self.num_classes, dropout=config.dropout)
        self.ct_bba2 = EvidenceBBAHead(config.hidden_dim, self.num_classes, dropout=config.dropout)

        self.pathology_bba1 = SoftmaxBBAHead(config.hidden_dim, self.num_classes, dropout=config.dropout)
        self.pathology_bba2 = EvidenceBBAHead(config.hidden_dim, self.num_classes, dropout=config.dropout)

        self.ds_fusion = DempsterShaferFusion(num_classes=self.num_classes)
        self.risk_mapper = RiskMapper(num_classes=self.num_classes, positive_class_index=1)

        self.use_compensation = bool(config.use_compensation)
        if self.use_compensation:
            self.ct_to_path_compensation = CompensationModule(config.hidden_dim, dropout=config.dropout)
            self.path_to_ct_compensation = CompensationModule(config.hidden_dim, dropout=config.dropout)

        self.calibration = OptionalCalibrationModule(enabled=False)

    def _encode_ct(self, ct: torch.Tensor, ct_present: torch.Tensor) -> Dict[str, torch.Tensor]:
        ct_hidden = self.ct_encoder(ct, ct_present)
        ct_b1, ct_u1 = self.ct_bba1(ct_hidden, ct_present)
        ct_b2, ct_u2 = self.ct_bba2(ct_hidden, ct_present)
        ct_belief, ct_uncertainty, _ = self.ds_fusion.fuse_intra_modality(
            ct_b1, ct_u1, ct_b2, ct_u2
        )
        return {
            "hidden": ct_hidden,
            "belief": ct_belief,
            "uncertainty": ct_uncertainty,
            "bba1_belief": ct_b1,
            "bba1_uncertainty": ct_u1,
            "bba2_belief": ct_b2,
            "bba2_uncertainty": ct_u2,
        }

    def _encode_pathology(
        self,
        pathology: torch.Tensor,
        pathology_present: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        pathology_hidden = self.pathology_encoder(pathology, pathology_present)
        path_b1, path_u1 = self.pathology_bba1(pathology_hidden, pathology_present)
        path_b2, path_u2 = self.pathology_bba2(pathology_hidden, pathology_present)
        path_belief, path_uncertainty, _ = self.ds_fusion.fuse_intra_modality(
            path_b1, path_u1, path_b2, path_u2
        )
        path_belief, path_uncertainty = self.calibration(
            pathology_belief=path_belief,
            pathology_uncertainty=path_uncertainty,
            metadata=metadata,
        )
        return {
            "hidden": pathology_hidden,
            "belief": path_belief,
            "uncertainty": path_uncertainty,
            "bba1_belief": path_b1,
            "bba1_uncertainty": path_u1,
            "bba2_belief": path_b2,
            "bba2_uncertainty": path_u2,
        }

    def _apply_missing_modality_compensation(
        self,
        ct_output: Dict[str, torch.Tensor],
        pathology_output: Dict[str, torch.Tensor],
        ct_present: torch.Tensor,
        pathology_present: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if not self.use_compensation:
            return ct_output, pathology_output, ct_present, pathology_present

        ct_present_mask = _as_float_mask(ct_present)
        path_present_mask = _as_float_mask(pathology_present)

        effective_ct_present = ct_present.clone().float()
        effective_path_present = pathology_present.clone().float()

        ct_missing = (ct_present_mask == 0)
        path_missing = (path_present_mask == 0)

        # pathology -> CT compensation
        if ct_missing.any() and (~path_missing).any():
            pseudo_ct_hidden, scale = self.path_to_ct_compensation(
                observed_feature=pathology_output["hidden"],
                observed_uncertainty=pathology_output["uncertainty"],
            )
            pseudo_b1, pseudo_u1 = self.ct_bba1(pseudo_ct_hidden, torch.ones_like(ct_present))
            pseudo_b2, pseudo_u2 = self.ct_bba2(pseudo_ct_hidden, torch.ones_like(ct_present))
            pseudo_belief, _, _ = self.ds_fusion.fuse_intra_modality(
                pseudo_b1, pseudo_u1, pseudo_b2, pseudo_u2
            )
            pseudo_belief = pseudo_belief * scale
            pseudo_uncertainty = 1.0 - pseudo_belief.sum(dim=-1, keepdim=True)
            pseudo_uncertainty = pseudo_uncertainty.clamp(min=1e-6, max=1.0)

            ct_output["belief"] = torch.where(ct_missing, pseudo_belief, ct_output["belief"])
            ct_output["uncertainty"] = torch.where(ct_missing, pseudo_uncertainty, ct_output["uncertainty"])
            ct_output["hidden"] = torch.where(ct_missing, pseudo_ct_hidden, ct_output["hidden"])
            effective_ct_present = torch.where(ct_missing.squeeze(-1), torch.ones_like(effective_ct_present), effective_ct_present)

        # CT -> pathology compensation
        if path_missing.any() and (~ct_missing).any():
            pseudo_path_hidden, scale = self.ct_to_path_compensation(
                observed_feature=ct_output["hidden"],
                observed_uncertainty=ct_output["uncertainty"],
            )
            pseudo_b1, pseudo_u1 = self.pathology_bba1(pseudo_path_hidden, torch.ones_like(pathology_present))
            pseudo_b2, pseudo_u2 = self.pathology_bba2(pseudo_path_hidden, torch.ones_like(pathology_present))
            pseudo_belief, _, _ = self.ds_fusion.fuse_intra_modality(
                pseudo_b1, pseudo_u1, pseudo_b2, pseudo_u2
            )
            pseudo_belief = pseudo_belief * scale
            pseudo_uncertainty = 1.0 - pseudo_belief.sum(dim=-1, keepdim=True)
            pseudo_uncertainty = pseudo_uncertainty.clamp(min=1e-6, max=1.0)

            pathology_output["belief"] = torch.where(path_missing, pseudo_belief, pathology_output["belief"])
            pathology_output["uncertainty"] = torch.where(path_missing, pseudo_uncertainty, pathology_output["uncertainty"])
            pathology_output["hidden"] = torch.where(path_missing, pseudo_path_hidden, pathology_output["hidden"])
            effective_path_present = torch.where(path_missing.squeeze(-1), torch.ones_like(effective_path_present), effective_path_present)

        return ct_output, pathology_output, effective_ct_present, effective_path_present

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        ct = batch["ct"]
        pathology = batch["pathology"]
        presence_mask = batch["presence_mask"].float()

        ct_present = presence_mask[:, 0]
        pathology_present = presence_mask[:, 1]

        ct_output = self._encode_ct(ct, ct_present)
        pathology_output = self._encode_pathology(pathology, pathology_present, metadata=metadata)

        # Innovation note:
        # when one modality is missing, generate low-confidence pseudo evidence
        # from the observed modality instead of naively dropping the patient.
        ct_output, pathology_output, effective_ct_present, effective_path_present = self._apply_missing_modality_compensation(
            ct_output=ct_output,
            pathology_output=pathology_output,
            ct_present=ct_present,
            pathology_present=pathology_present,
        )

        fused_belief, u_fused, conflict = self.ds_fusion.fuse_inter_modality(
            modality_outputs={
                "ct": {
                    "belief": ct_output["belief"],
                    "uncertainty": ct_output["uncertainty"],
                },
                "pathology": {
                    "belief": pathology_output["belief"],
                    "uncertainty": pathology_output["uncertainty"],
                },
            },
            modality_presence={
                "ct": effective_ct_present,
                "pathology": effective_path_present,
            },
        )

        risk, pignistic = self.risk_mapper(fused_belief, u_fused)

        # Trust attribution used in the manuscript:
        # lower uncertainty -> higher modality trust.
        trust_ct = 1.0 - ct_output["uncertainty"]
        trust_pathology = 1.0 - pathology_output["uncertainty"]

        return {
            "risk": risk,
            "pignistic": pignistic,
            "fused_belief": fused_belief,
            "u_fused": u_fused,
            "conflict": conflict,

            # Backward-compatible aliases
            "evidence": fused_belief,
            "uncertainty": u_fused,

            "ct_uncertainty": ct_output["uncertainty"],
            "pathology_uncertainty": pathology_output["uncertainty"],
            "trust_ct": trust_ct,
            "trust_pathology": trust_pathology,
            "ct_belief": ct_output["belief"],
            "pathology_belief": pathology_output["belief"],
            "presence_mask": presence_mask,
        }


def build_model(config) -> EviFusCRC:
    return EviFusCRC(config)
