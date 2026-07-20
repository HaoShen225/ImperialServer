from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


def one_hot_mask(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def soft_dice_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 3,
    include_bg: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    target = one_hot_mask(y, num_classes)
    classes = range(num_classes) if include_bg else range(1, num_classes)
    losses = []
    for cls in classes:
        pc = prob[:, cls]
        tc = target[:, cls]
        inter = (pc * tc).sum(dim=(1, 2))
        denom = pc.sum(dim=(1, 2)) + tc.sum(dim=(1, 2))
        losses.append(1.0 - (2.0 * inter + eps) / (denom + eps))
    return torch.cat([v.reshape(-1) for v in losses]).mean() if losses else logits.sum() * 0.0


def segmentation_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int = 3,
    dice_weight: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    return F.cross_entropy(logits, mask) + float(dice_weight) * soft_dice_loss(
        logits,
        mask,
        num_classes=num_classes,
        include_bg=False,
        eps=eps,
    )


def L2_disperssion_loss(
    features: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int = 3,
    margin: float = 0.0,
) -> torch.Tensor:
    """Repel batch class prototypes in normalized dec1 space."""
    if features.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask[:, None].float(), size=features.shape[-2:], mode="nearest")[:, 0].long()

    z = F.normalize(features, p=2, dim=1, eps=1e-8)
    flat_z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    flat_y = mask.reshape(-1)

    prototypes: List[torch.Tensor] = []
    for cls in range(int(num_classes)):
        cls_z = flat_z[flat_y == cls]
        if cls_z.numel() == 0:
            continue
        mu = F.normalize(cls_z.mean(dim=0), p=2, dim=0, eps=1e-8)
        prototypes.append(mu)

    if len(prototypes) < 2:
        return features.sum() * 0.0

    losses = []
    for i in range(len(prototypes)):
        for j in range(i + 1, len(prototypes)):
            cos_ij = torch.sum(prototypes[i] * prototypes[j])
            losses.append(F.relu(cos_ij - float(margin)).pow(2))
    return torch.stack(losses).mean() if losses else features.sum() * 0.0


dec1_dispersion_loss = L2_disperssion_loss


def backbone_training_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dec1_features: torch.Tensor,
    num_classes: int = 3,
    dice_weight: float = 0.5,
    lambda_disp: float = 0.05,
    disp_margin: float = 0.0,
) -> Dict[str, torch.Tensor]:
    seg = segmentation_loss(logits, mask, num_classes=num_classes, dice_weight=dice_weight)
    disp = L2_disperssion_loss(dec1_features, mask, num_classes=num_classes, margin=disp_margin)
    total = seg + float(lambda_disp) * disp
    return {"loss": total, "seg_loss": seg, "disp_loss": disp}


def sadg_grad_l2_norm(
    params: Iterable[torch.nn.Parameter],
    device: str | torch.device | None = None,
) -> torch.Tensor:
    params = list(params)
    total: torch.Tensor | None = None
    for p in params:
        if p.grad is None:
            continue
        value = p.grad.detach().pow(2).sum()
        total = value if total is None else total + value
    if total is not None:
        return torch.sqrt(total)
    if device is None:
        device = params[0].device if params else torch.device("cpu")
    return torch.tensor(0.0, device=torch.device(device))


def add_sadg_perturbation(
    params: Iterable[torch.nn.Parameter],
    grad_norm: torch.Tensor,
    *,
    rho: float,
    eps: float,
) -> List[tuple[torch.nn.Parameter, torch.Tensor]]:
    scale = float(rho) / (grad_norm + float(eps))
    perturbations: List[tuple[torch.nn.Parameter, torch.Tensor]] = []
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            e_w = p.grad.detach() * scale
            p.add_(e_w)
            perturbations.append((p, e_w))
    return perturbations


def restore_sadg_perturbation(
    perturbations: Sequence[tuple[torch.nn.Parameter, torch.Tensor]],
) -> None:
    with torch.no_grad():
        for p, e_w in perturbations:
            p.sub_(e_w)


def apply_sadg_perturbation_from_loss(
    params: Iterable[torch.nn.Parameter],
    epsilon_loss: torch.Tensor,
    *,
    rho: float,
    eps: float,
    device: str | torch.device | None = None,
) -> Dict[str, Any]:
    params = list(params)
    epsilon_loss.backward()
    grad_norm = sadg_grad_l2_norm(params, device=device)
    perturbations = add_sadg_perturbation(params, grad_norm, rho=float(rho), eps=float(eps))
    perturb_norm = float(rho) if float(grad_norm.detach().cpu()) > 0.0 else 0.0
    return {
        "perturbations": perturbations,
        "grad_norm": grad_norm,
        "perturb_norm": perturb_norm,
    }
