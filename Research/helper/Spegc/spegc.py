"""SPEGC utilities for continual test-time medical image segmentation.

This is a PyTorch/U-Net adaptation of the official SPEGC implementation.  It
keeps the semantic-prompt-enhanced graph clustering objective intact while
replacing detector proposals with reliable foreground pixels from segmentation
pseudo-labels.
"""

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ModelState = Dict[str, torch.Tensor]
OptimizerState = Dict[str, Any]
PoolItem = Tuple[torch.Tensor, torch.Tensor]


def model_features_and_logits(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(features, logits)`` from a supported segmentation output."""
    if isinstance(output, (tuple, list)):
        if len(output) < 2:
            raise ValueError("SPEGC requires a model output containing both features and logits.")
        features, logits = output[-2], output[-1]
    elif isinstance(output, Mapping):
        feature_key = "features" if "features" in output else "feature"
        if feature_key not in output or "logits" not in output:
            raise KeyError("A mapping output must contain 'features' (or 'feature') and 'logits'.")
        features, logits = output[feature_key], output["logits"]
    else:
        raise TypeError("SPEGC requires a (features, logits) sequence or mapping model output.")

    if not isinstance(features, torch.Tensor) or not isinstance(logits, torch.Tensor):
        raise TypeError("Extracted features and logits must both be torch.Tensor objects.")
    if features.ndim != 4 or logits.ndim != 4:
        raise ValueError("SPEGC expects [B, C, H, W] features and logits.")
    if features.shape[0] != logits.shape[0]:
        raise ValueError("Feature and logit batch sizes differ.")
    if features.shape[-2:] != logits.shape[-2:]:
        logits = F.interpolate(logits, size=features.shape[-2:], mode="bilinear", align_corners=False)
    return features, logits


def mc_dropout_uncertainty(
    features: torch.Tensor,
    passes: int = 4,
    dropout_probability: float = 0.1,
) -> torch.Tensor:
    """Estimate a per-location uncertainty map with feature MC Dropout."""
    if features.ndim != 4 or not features.is_floating_point():
        raise ValueError("MC Dropout expects floating [B, C, H, W] features.")
    if int(passes) < 2:
        raise ValueError("At least two MC Dropout passes are required to estimate variance.")
    if not 0.0 <= float(dropout_probability) < 1.0:
        raise ValueError("Dropout probability must satisfy 0 <= p < 1.")
    stochastic = torch.stack(
        [F.dropout2d(features, p=float(dropout_probability), training=True) for _ in range(int(passes))],
        dim=0,
    )
    return stochastic.var(dim=0).mean(dim=1)


def sample_reliable_foreground_nodes(
    features: torch.Tensor,
    logits: torch.Tensor,
    uncertainty: torch.Tensor,
    keep_ratio: float = 0.5,
    sample_dist: int = 10,
    background_index: int = 0,
    max_nodes: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, int]]]:
    """Build one sparse reliable foreground graph per batch item.

    Reliable locations are the lowest-uncertainty ``keep_ratio`` fraction of
    foreground pseudo-labels.  The final strided sampling follows the official
    ``sample_dist`` implementation and therefore yields approximately that many
    nodes when many candidates are present.
    """
    if features.ndim != 4 or logits.ndim != 4 or uncertainty.ndim != 3:
        raise ValueError("Expected features/logits [B,C,H,W] and uncertainty [B,H,W].")
    if features.shape[0] != logits.shape[0] or features.shape[0] != uncertainty.shape[0]:
        raise ValueError("Features, logits, and uncertainty must have the same batch size.")
    if features.shape[-2:] != logits.shape[-2:] or features.shape[-2:] != uncertainty.shape[-2:]:
        raise ValueError("Features, logits, and uncertainty must have the same spatial size.")
    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError("keep_ratio must satisfy 0 < keep_ratio <= 1.")
    if int(sample_dist) < 1:
        raise ValueError("sample_dist must be at least 1.")
    if max_nodes is not None and int(max_nodes) < 1:
        raise ValueError("max_nodes must be positive when specified.")

    pseudo_labels = logits.detach().argmax(dim=1)
    nodes_batch: List[torch.Tensor] = []
    labels_batch: List[torch.Tensor] = []
    stats: List[Dict[str, int]] = []
    for batch_index in range(features.shape[0]):
        flat_labels = pseudo_labels[batch_index].reshape(-1)
        foreground_indices = torch.nonzero(flat_labels != int(background_index), as_tuple=False).flatten()
        foreground_count = int(foreground_indices.numel())
        if foreground_count == 0:
            nodes = features.new_empty((0, features.shape[1]))
            labels = flat_labels.new_empty((0,))
            reliable_count = 0
        else:
            flat_uncertainty = uncertainty[batch_index].reshape(-1)
            foreground_uncertainty = flat_uncertainty.index_select(0, foreground_indices)
            reliable_count = max(1, int(foreground_count * float(keep_ratio)))
            reliable_order = torch.argsort(foreground_uncertainty)[:reliable_count]
            reliable_indices = foreground_indices.index_select(0, reliable_order)

            step = reliable_count // int(sample_dist)
            if step > 1:
                reliable_indices = reliable_indices[::step]
            if max_nodes is not None:
                reliable_indices = reliable_indices[: int(max_nodes)]

            flat_features = features[batch_index].permute(1, 2, 0).reshape(-1, features.shape[1])
            nodes = flat_features.index_select(0, reliable_indices)
            labels = flat_labels.index_select(0, reliable_indices)

        nodes_batch.append(nodes)
        labels_batch.append(labels)
        stats.append(
            {
                "foreground_pixels": foreground_count,
                "reliable_pixels": reliable_count,
                "sampled_nodes": int(nodes.shape[0]),
            }
        )
    return nodes_batch, labels_batch, stats


class GraphFeaturePool:
    """FIFO history of detached graphs used to form an online pseudo-batch."""

    def __init__(self, capacity: int = 3, min_size: int = 1) -> None:
        if int(capacity) < 1:
            raise ValueError("Graph pool capacity must be at least 1.")
        if not 0 <= int(min_size) <= int(capacity):
            raise ValueError("min_size must satisfy 0 <= min_size <= capacity.")
        self.capacity = int(capacity)
        self.min_size = int(min_size)
        self._items: List[PoolItem] = []

    def __len__(self) -> int:
        return len(self._items)

    @property
    def ready(self) -> bool:
        return len(self) >= self.min_size

    def update(self, nodes: torch.Tensor, labels: torch.Tensor) -> None:
        if nodes.ndim != 2 or labels.ndim != 1 or nodes.shape[0] != labels.shape[0]:
            raise ValueError("Pool items must be nodes [N,D] and labels [N].")
        if nodes.shape[0] == 0:
            return
        if len(self._items) >= self.capacity:
            self._items.pop(0)
        self._items.append((nodes.detach().clone(), labels.detach().clone()))

    def pseudo_batch(
        self,
        current_nodes: Sequence[torch.Tensor],
        current_labels: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if len(current_nodes) != len(current_labels):
            raise ValueError("Current node and label graph counts differ.")
        valid = [
            (nodes, labels)
            for nodes, labels in zip(current_nodes, current_labels)
            if nodes.ndim == 2 and nodes.shape[0] > 0
        ]
        nodes = [item[0] for item in valid] + [item[0] for item in self._items]
        labels = [item[1] for item in valid] + [item[1] for item in self._items]
        return nodes, labels

    def clear(self) -> None:
        self._items.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "min_size": self.min_size,
            "items": [(nodes.clone(), labels.clone()) for nodes, labels in self._items],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity or int(state["min_size"]) != self.min_size:
            raise ValueError("Graph pool configuration differs from the saved state.")
        items = state.get("items", [])
        if len(items) > self.capacity:
            raise ValueError("Saved graph pool exceeds its configured capacity.")
        self._items = [(nodes.detach().clone(), labels.detach().clone()) for nodes, labels in items]


class SemanticPromptGraphClustering(nn.Module):
    """Semantic Prompt Feature Enhancement and graph clustering objective."""

    def __init__(
        self,
        feature_dim: int = 64,
        num_centroids: int = 48,
        target_clusters: int = 48,
        num_prompts: int = 8,
        density_temperature: float = 0.1,
        sinkhorn_temperature: float = 0.05,
        commonality_weight: float = 0.2,
        sinkhorn_iterations: int = 20,
    ) -> None:
        super().__init__()
        if min(int(feature_dim), int(num_centroids), int(target_clusters), int(num_prompts)) < 1:
            raise ValueError("Feature, centroid, cluster, and prompt counts must be positive.")
        if float(density_temperature) <= 0.0 or float(sinkhorn_temperature) <= 0.0:
            raise ValueError("SPEGC temperatures must be positive.")
        if int(sinkhorn_iterations) < 1:
            raise ValueError("sinkhorn_iterations must be at least 1.")
        self.feature_dim = int(feature_dim)
        self.target_clusters = int(target_clusters)
        self.density_temperature = float(density_temperature)
        self.sinkhorn_temperature = float(sinkhorn_temperature)
        self.commonality_weight = float(commonality_weight)
        self.sinkhorn_iterations = int(sinkhorn_iterations)

        self.commonality_prompts = nn.Parameter(torch.randn(num_prompts, feature_dim) * 0.01)
        self.hierarchy_prompts = nn.Parameter(torch.randn(num_prompts, feature_dim) * 0.01)
        self.pooling_context = nn.Parameter(torch.randn(feature_dim) * 0.01)
        self.query_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.centroids = nn.Parameter(torch.randn(num_centroids, feature_dim) + 1.0 / num_centroids)
        nn.init.normal_(self.query_projection.weight, std=0.01)
        nn.init.normal_(self.key_projection.weight, std=0.01)

    def run_sinkhorn(self, cost: torch.Tensor, selected_edges: int) -> torch.Tensor:
        """Solve the official two-column entropic OT edge-selection problem."""
        if cost.ndim != 2 or cost.shape[1] != 2 or cost.shape[0] < 1:
            raise ValueError("Sinkhorn cost must have shape [E, 2] with E >= 1.")
        edge_count = int(cost.shape[0])
        selected_edges = max(0, min(int(selected_edges), edge_count))
        gamma = torch.exp(-cost / self.sinkhorn_temperature).clamp_min(1e-12)
        column_marginal = cost.new_tensor([edge_count - selected_edges, selected_edges]).view(2, 1)
        for _ in range(self.sinkhorn_iterations):
            gamma = gamma / gamma.sum(dim=1, keepdim=True).clamp_min(1e-12)
            column_sums = gamma.sum(dim=0, keepdim=True).T
            gamma = gamma * (column_marginal / column_sums.clamp_min(1e-12)).T
        return gamma

    def forward(
        self,
        nodes_batch: Sequence[torch.Tensor],
        centroids: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Any:
        valid_nodes = [nodes for nodes in nodes_batch if nodes.ndim == 2 and nodes.shape[0] > 0]
        centers = self.centroids if centroids is None else centroids
        if centers.ndim != 2 or centers.shape[1] != self.feature_dim:
            raise ValueError(f"Centroids must have shape [Z, {self.feature_dim}].")
        if any(nodes.shape[1] != self.feature_dim for nodes in valid_nodes):
            raise ValueError(f"Every node graph must have feature dimension {self.feature_dim}.")
        if len(valid_nodes) <= 1:
            zero = centers.sum() * 0.0
            details = {
                "graph_loss": zero,
                "commonality_loss": zero,
                "total_nodes": sum(int(nodes.shape[0]) for nodes in valid_nodes),
                "candidate_edges": 0,
                "edge_budget": 0,
                "selected_self_edge_mass": 0.0,
            }
            return (zero, details) if return_details else zero

        enhanced_nodes: List[torch.Tensor] = []
        retrieved_commonality: List[torch.Tensor] = []
        commonality_norm = F.normalize(self.commonality_prompts, p=2, dim=-1)
        hierarchy_norm = F.normalize(self.hierarchy_prompts, p=2, dim=-1)
        for nodes in valid_nodes:
            attention = F.softmax(nodes @ self.pooling_context, dim=0)
            query = attention @ nodes
            query_norm = F.normalize(query, p=2, dim=-1)
            commonality_weights = F.relu(-(commonality_norm @ query_norm))
            hierarchy_weights = F.softmax(hierarchy_norm @ query_norm, dim=0)
            commonality_prompt = commonality_weights @ self.commonality_prompts
            hierarchy_prompt = hierarchy_weights @ self.hierarchy_prompts
            retrieved_commonality.append(commonality_prompt)
            enhanced_nodes.append(nodes + commonality_prompt.unsqueeze(0) + hierarchy_prompt.unsqueeze(0))

        all_nodes = torch.cat(enhanced_nodes, dim=0)
        total_nodes = int(all_nodes.shape[0])
        if total_nodes <= self.target_clusters:
            zero = centers.sum() * 0.0
            details = {
                "graph_loss": zero,
                "commonality_loss": zero,
                "total_nodes": total_nodes,
                "candidate_edges": total_nodes * max(0, total_nodes - 1),
                "edge_budget": 0,
                "selected_self_edge_mass": 0.0,
            }
            return (zero, details) if return_details else zero

        query = self.query_projection(all_nodes)
        key = self.key_projection(all_nodes)
        similarity = (query @ key.T) / (self.feature_dim ** 0.5)
        positive_similarity = F.relu(similarity)
        density = positive_similarity.sum(dim=1)
        density_difference = density.unsqueeze(0) - density.unsqueeze(1)
        refined = positive_similarity * torch.sigmoid(density_difference / self.density_temperature)

        off_diagonal = ~torch.eye(total_nodes, dtype=torch.bool, device=refined.device)
        affinities = refined[off_diagonal]
        minimum, maximum = affinities.min(), affinities.max()
        cost = torch.stack((affinities - minimum, maximum - affinities), dim=1)
        affinity_range = maximum - minimum
        cost = torch.where(affinity_range > 1e-8, cost / affinity_range.clamp_min(1e-12), cost)
        selected_edges = total_nodes - self.target_clusters
        selected_vector = self.run_sinkhorn(cost, selected_edges)[:, 1]
        selected_affinity = refined.new_zeros((total_nodes, total_nodes))
        selected_affinity[off_diagonal] = selected_vector

        assignments = F.softmax(all_nodes @ centers.T, dim=1)
        detached_assignments = assignments.detach()
        log_assignments = assignments.clamp_min(1e-12).log()
        log_detached = detached_assignments.clamp_min(1e-12).log()
        negative_entropy = (assignments * log_assignments).sum(dim=1)
        cross_term = assignments @ log_detached.T
        pairwise_kl = negative_entropy.unsqueeze(0) - cross_term.T
        graph_loss = (selected_affinity * pairwise_kl).sum() / total_nodes

        prompt_batch = F.normalize(torch.stack(retrieved_commonality, dim=0), p=2, dim=1)
        commonality_loss = (1.0 - prompt_batch @ prompt_batch.T).mean()
        loss = graph_loss + self.commonality_weight * commonality_loss
        details = {
            "graph_loss": graph_loss,
            "commonality_loss": commonality_loss,
            "total_nodes": total_nodes,
            "candidate_edges": int(affinities.numel()),
            "edge_budget": selected_edges,
            "selected_self_edge_mass": float(selected_affinity.diagonal().detach().sum().cpu()),
        }
        return (loss, details) if return_details else loss


@torch.enable_grad()
def forward_and_adapt_spegc(
    images: torch.Tensor,
    model: nn.Module,
    graph_module: SemanticPromptGraphClustering,
    optimizer: torch.optim.Optimizer,
    graph_pool: GraphFeaturePool,
    *,
    detach_backbone: bool = True,
    mc_passes: int = 4,
    dropout_probability: float = 0.1,
    keep_ratio: float = 0.5,
    sample_dist: int = 10,
    background_index: int = 0,
    max_nodes: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run one online SPEGC step and return pre-update segmentation logits."""
    features, logits = model_features_and_logits(model(images))
    graph_features = features.detach() if detach_backbone else features
    uncertainty = mc_dropout_uncertainty(graph_features, mc_passes, dropout_probability)
    current_nodes, current_labels, sampling_stats = sample_reliable_foreground_nodes(
        graph_features,
        logits,
        uncertainty,
        keep_ratio,
        sample_dist,
        background_index,
        max_nodes,
    )
    valid_nodes = [nodes for nodes in current_nodes if nodes.shape[0] > 0]
    valid_labels = [labels for nodes, labels in zip(current_nodes, current_labels) if nodes.shape[0] > 0]
    pool_was_ready = graph_pool.ready
    pseudo_nodes, _ = graph_pool.pseudo_batch(valid_nodes, valid_labels)
    total_nodes = sum(int(nodes.shape[0]) for nodes in pseudo_nodes)
    total_pixels = int(uncertainty.numel())

    stats: Dict[str, Any] = {
        "updated": False,
        "skip_reason": None,
        "pool_size_before": len(graph_pool),
        "pool_size_after": None,
        "graph_count": len(pseudo_nodes),
        "current_graph_count": len(valid_nodes),
        "total_nodes": total_nodes,
        "candidate_edges": total_nodes * max(0, total_nodes - 1),
        "edge_budget": max(0, total_nodes - graph_module.target_clusters),
        "sampled_nodes": sum(item["sampled_nodes"] for item in sampling_stats),
        "foreground_pixels": sum(item["foreground_pixels"] for item in sampling_stats),
        "foreground_ratio": (
            sum(item["foreground_pixels"] for item in sampling_stats) / total_pixels
            if total_pixels else 0.0
        ),
        "reliable_pixels": sum(item["reliable_pixels"] for item in sampling_stats),
        "graph_loss": None,
        "commonality_loss": None,
        "total_loss": None,
        "gradient_norm": None,
        "selected_self_edge_mass": 0.0,
        "detach_backbone": bool(detach_backbone),
    }

    optimizer.zero_grad(set_to_none=True)
    if not valid_nodes:
        stats["skip_reason"] = "empty_current_graph"
    elif not pool_was_ready:
        stats["skip_reason"] = "pool_not_ready"
    elif len(pseudo_nodes) <= 1:
        stats["skip_reason"] = "insufficient_graphs"
    elif total_nodes <= graph_module.target_clusters:
        stats["skip_reason"] = "insufficient_nodes"
    else:
        loss, details = graph_module(pseudo_nodes, return_details=True)
        stats["total_nodes"] = int(details["total_nodes"])
        stats["candidate_edges"] = int(details["candidate_edges"])
        stats["edge_budget"] = int(details["edge_budget"])
        stats["selected_self_edge_mass"] = float(details["selected_self_edge_mass"])
        if not bool(torch.isfinite(loss)):
            stats["skip_reason"] = "nonfinite_loss"
        else:
            loss.backward()
            gradients = [
                parameter.grad
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            if not gradients:
                stats["skip_reason"] = "no_gradients"
            elif not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                stats["skip_reason"] = "nonfinite_gradient"
            else:
                squared_norm = sum(gradient.detach().float().square().sum() for gradient in gradients)
                stats["gradient_norm"] = float(torch.sqrt(squared_norm).cpu())
                optimizer.step()
                stats["updated"] = True
        stats["total_loss"] = float(loss.detach().cpu())
        stats["graph_loss"] = float(details["graph_loss"].detach().cpu())
        stats["commonality_loss"] = float(details["commonality_loss"].detach().cpu())
        if not stats["updated"]:
            optimizer.zero_grad(set_to_none=True)

    for nodes, labels in zip(valid_nodes, valid_labels):
        graph_pool.update(nodes, labels)
    stats["pool_size_after"] = len(graph_pool)
    return logits, stats


class SPEGC(nn.Module):
    """Online SPEGC wrapper with continual/episodic state management."""

    def __init__(
        self,
        model: nn.Module,
        graph_module: SemanticPromptGraphClustering,
        optimizer: torch.optim.Optimizer,
        *,
        steps: int = 1,
        episodic: bool = False,
        detach_backbone: bool = True,
        pool_size: int = 3,
        min_pool_size: int = 1,
        mc_passes: int = 4,
        dropout_probability: float = 0.1,
        keep_ratio: float = 0.5,
        sample_dist: int = 10,
        background_index: int = 0,
        max_nodes: Optional[int] = None,
    ) -> None:
        super().__init__()
        if int(steps) < 1:
            raise ValueError("SPEGC steps must be at least 1.")
        self.model = model
        self.graph_module = graph_module
        self.optimizer = optimizer
        self.steps = int(steps)
        self.episodic = bool(episodic)
        self.detach_backbone = bool(detach_backbone)
        self.mc_passes = int(mc_passes)
        self.dropout_probability = float(dropout_probability)
        self.keep_ratio = float(keep_ratio)
        self.sample_dist = int(sample_dist)
        self.background_index = int(background_index)
        self.max_nodes = max_nodes
        self.graph_pool = GraphFeaturePool(pool_size, min_pool_size)
        self.last_stats: Dict[str, Any] = {}

        self.model_state: ModelState = deepcopy(model.state_dict())
        self.graph_state: ModelState = deepcopy(graph_module.state_dict())
        self.optimizer_state: OptimizerState = deepcopy(optimizer.state_dict())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()
        logits: Optional[torch.Tensor] = None
        for _ in range(self.steps):
            logits, self.last_stats = forward_and_adapt_spegc(
                images,
                self.model,
                self.graph_module,
                self.optimizer,
                self.graph_pool,
                detach_backbone=self.detach_backbone,
                mc_passes=self.mc_passes,
                dropout_probability=self.dropout_probability,
                keep_ratio=self.keep_ratio,
                sample_dist=self.sample_dist,
                background_index=self.background_index,
                max_nodes=self.max_nodes,
            )
        if logits is None:
            raise RuntimeError("SPEGC did not execute an adaptation step.")
        return logits

    def reset(self) -> None:
        """Restore initial model, graph module, optimizer, and empty history."""
        self.model.load_state_dict(self.model_state, strict=True)
        self.graph_module.load_state_dict(self.graph_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.graph_pool.clear()
        self.last_stats = {}


__all__: Sequence[str] = (
    "GraphFeaturePool",
    "SPEGC",
    "SemanticPromptGraphClustering",
    "forward_and_adapt_spegc",
    "mc_dropout_uncertainty",
    "model_features_and_logits",
    "sample_reliable_foreground_nodes",
)
