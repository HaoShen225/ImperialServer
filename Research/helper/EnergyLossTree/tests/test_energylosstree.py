"""Tests for global and windowed EnergyLossTree pseudo-label propagation."""

import math
import unittest
from typing import List, Optional, Sequence, Tuple

import torch

from Research.helper.EnergyLossTree import (
    DEFAULT_SPATIAL_TEMPERATURE,
    DualTreePseudoLabels,
    make_pseudo_label_weights,
    propagate_dual_tree_pseudo_labels,
    windowed_tree_propagation,
)
from Research.helper.EnergyLossTree import energylosstree as implementation


def _probabilities(
    batch: int = 1,
    classes: int = 4,
    height: int = 7,
    width: int = 9,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(17)
    logits = torch.randn(
        batch,
        classes,
        height,
        width,
        generator=generator,
        device=device,
    )
    return logits.softmax(dim=1)


def _require_cuda_extension() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is unavailable.")
    try:
        implementation._load_tree_filter_cuda()
    except RuntimeError as error:
        raise unittest.SkipTest(str(error))


class EnergyLossTreeCpuTests(unittest.TestCase):
    def test_default_spatial_temperature_is_sixteen_pixels(self) -> None:
        self.assertEqual(DEFAULT_SPATIAL_TEMPERATURE, 16.0)

    def test_default_weights_are_float32_ones(self) -> None:
        probabilities = _probabilities(batch=2, height=5, width=6)
        weights = make_pseudo_label_weights(probabilities)
        self.assertEqual(weights.shape, (2, 1, 5, 6))
        self.assertEqual(weights.dtype, torch.float32)
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))

    def test_explicit_weight_validation_and_detach(self) -> None:
        probabilities = _probabilities(height=3, width=4)
        raw = torch.linspace(0.0, 1.0, 12, requires_grad=True).reshape(1, 3, 4)
        weights = make_pseudo_label_weights(probabilities, raw)
        self.assertEqual(weights.shape, (1, 1, 3, 4))
        self.assertFalse(weights.requires_grad)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            make_pseudo_label_weights(probabilities, -torch.ones(1, 3, 4))
        with self.assertRaisesRegex(ValueError, "matching"):
            make_pseudo_label_weights(probabilities, torch.ones(1, 2, 4))

    def test_probability_input_is_not_modified_on_cpu_error(self) -> None:
        probabilities = _probabilities(height=4, width=4)
        original = probabilities.clone()
        guidance = torch.randn(1, 3, 4, 4)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            windowed_tree_propagation(
                probabilities,
                guidance,
                window_size=4,
                stride=4,
            )
        self.assertTrue(torch.equal(probabilities, original))

    def test_optional_window_and_temperature_validation_precede_cuda(self) -> None:
        probabilities = _probabilities(height=4, width=4)
        guidance = torch.randn(1, 3, 4, 4)
        for window_size, stride in ((None, 2), (4, None)):
            with self.subTest(window_size=window_size, stride=stride):
                with self.assertRaisesRegex(ValueError, "both be provided or both be None"):
                    windowed_tree_propagation(
                        probabilities,
                        guidance,
                        window_size=window_size,
                        stride=stride,
                    )

        for temperature in (0.0, -1.0, math.inf, math.nan, True):
            with self.subTest(spatial_temperature=temperature):
                with self.assertRaisesRegex(ValueError, "spatial_temperature"):
                    windowed_tree_propagation(
                        probabilities,
                        guidance,
                        spatial_temperature=temperature,
                    )

    def test_tree_edge_affinity_combines_feature_and_path_distance(self) -> None:
        flat_guidance = torch.tensor([[[0.0, 2.0, 2.0, 2.0]]])
        sorted_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        sorted_parent = torch.tensor([[0, 0, 1, 2]], dtype=torch.int32)
        affinity = implementation._tree_edge_affinities(
            flat_guidance,
            sorted_index,
            sorted_parent,
            width=4,
            sigma=2.0,
            spatial_temperature=4.0,
        )
        expected = torch.tensor(
            [[1.0, math.exp(-2.0 - 0.25), math.exp(-0.25), math.exp(-0.25)]]
        )
        torch.testing.assert_close(affinity, expected)
        self.assertAlmostEqual(
            float(affinity[0, 1:].prod()),
            math.exp(-2.0 - 3.0 / 4.0),
            places=6,
        )

        feature_only = implementation._tree_edge_affinities(
            flat_guidance,
            sorted_index,
            sorted_parent,
            width=4,
            sigma=2.0,
            spatial_temperature=None,
        )
        torch.testing.assert_close(
            feature_only,
            torch.tensor([[1.0, math.exp(-2.0), 1.0, 1.0]]),
        )

    def test_compiled_boruvka_mst_forward(self) -> None:
        try:
            extension = implementation._load_tree_filter_cuda()
        except RuntimeError as error:
            raise unittest.SkipTest(str(error))
        edges = torch.tensor(
            [[[0, 1], [0, 2], [1, 3], [2, 3]]],
            dtype=torch.int32,
        )
        weights = torch.tensor([[1.0, 4.0, 2.0, 3.0]], dtype=torch.float32)
        tree = extension.mst_forward(edges, weights, 4)
        actual = {tuple(sorted(edge)) for edge in tree[0].tolist()}
        self.assertEqual(actual, {(0, 1), (1, 3), (2, 3)})


class EnergyLossTreeCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_cuda_extension()

    def test_window_argument_validation(self) -> None:
        probabilities = _probabilities(height=4, width=4, device="cuda")
        guidance = torch.randn(1, 2, 4, 4, device="cuda")
        cases = [
            (0, 1, "positive"),
            (4, 5, "cannot exceed"),
            ((4, 4), (2, 5), "cannot exceed"),
        ]
        for window_size, stride, message in cases:
            with self.subTest(window_size=window_size, stride=stride):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    windowed_tree_propagation(
                        probabilities,
                        guidance,
                        window_size=window_size,
                        stride=stride,
                    )

    def test_omitted_and_all_one_weights_are_equivalent(self) -> None:
        probabilities = _probabilities(height=6, width=7, device="cuda")
        guidance = torch.randn(1, 3, 4, 5, device="cuda")
        implicit = windowed_tree_propagation(
            probabilities,
            guidance,
            window_size=(4, 5),
            stride=(2, 3),
            sigma=0.7,
        )
        explicit = windowed_tree_propagation(
            probabilities,
            guidance,
            window_size=(4, 5),
            stride=(2, 3),
            pseudo_label_weights=torch.ones(1, 1, 6, 7, device="cuda"),
            sigma=0.7,
        )
        torch.testing.assert_close(implicit, explicit, rtol=1e-5, atol=1e-6)

    def test_default_global_mode_matches_image_sized_window(self) -> None:
        probabilities = _probabilities(height=4, width=5, device="cuda")
        guidance = torch.randn(1, 3, 4, 5, device="cuda")
        global_output = windowed_tree_propagation(
            probabilities,
            guidance,
            sigma=0.7,
            spatial_temperature=2.0,
        )
        explicit_window = windowed_tree_propagation(
            probabilities,
            guidance,
            window_size=(4, 5),
            stride=(4, 5),
            sigma=0.7,
            spatial_temperature=2.0,
        )
        torch.testing.assert_close(global_output, explicit_window, rtol=1e-5, atol=1e-6)

    def test_temperature_controls_long_range_influence(self) -> None:
        probabilities = torch.tensor(
            [[[[0.0, 1.0, 1.0, 1.0, 1.0]], [[1.0, 0.0, 0.0, 0.0, 0.0]]]],
            device="cuda",
        )
        guidance = torch.zeros(1, 1, 1, 5, device="cuda")
        strong_decay = windowed_tree_propagation(
            probabilities,
            guidance,
            spatial_temperature=0.5,
        )
        weak_decay = windowed_tree_propagation(
            probabilities,
            guidance,
            spatial_temperature=100.0,
        )
        self.assertLess(
            float(strong_decay[0, 1, 0, -1]),
            float(weak_decay[0, 1, 0, -1]),
        )

    def test_single_weighted_seed_propagates_through_constant_tree(self) -> None:
        probabilities = _probabilities(classes=3, height=3, width=4, device="cuda")
        guidance = torch.zeros(1, 2, 3, 4, device="cuda")
        weights = torch.zeros(1, 1, 3, 4, device="cuda")
        weights[0, 0, 1, 2] = 1.0
        output = windowed_tree_propagation(
            probabilities,
            guidance,
            window_size=(3, 4),
            stride=(3, 4),
            pseudo_label_weights=weights,
            sigma=1.0,
        )
        expected = probabilities[:, :, 1:2, 2:3].expand_as(output)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)

    def test_zero_weight_window_falls_back_to_source_probabilities(self) -> None:
        probabilities = _probabilities(classes=3, height=3, width=4, device="cuda")
        guidance = torch.randn(1, 2, 3, 4, device="cuda")
        output = windowed_tree_propagation(
            probabilities,
            guidance,
            window_size=(3, 4),
            stride=(3, 4),
            pseudo_label_weights=torch.zeros(1, 1, 3, 4, device="cuda"),
            sigma=1.0,
        )
        torch.testing.assert_close(output, probabilities, rtol=1e-5, atol=1e-6)

    def test_tail_windows_small_images_and_probability_simplex(self) -> None:
        probabilities = _probabilities(batch=2, height=7, width=9, device="cuda")
        guidance = torch.randn(2, 5, 3, 4, device="cuda")
        output = windowed_tree_propagation(
            probabilities,
            guidance,
            window_size=(12, 6),
            stride=(7, 4),
            sigma=0.5,
            window_batch_size=3,
        )
        self.assertEqual(output.shape, probabilities.shape)
        self.assertEqual(output.dtype, torch.float32)
        self.assertFalse(output.requires_grad)
        self.assertTrue(bool(torch.isfinite(output).all()))
        torch.testing.assert_close(
            output.sum(dim=1),
            torch.ones_like(output[:, 0]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_default_full_image_256_smoke(self) -> None:
        probabilities = _probabilities(height=256, width=256, device="cuda")
        guidance = torch.randn(1, 2, 64, 64, device="cuda")
        output = windowed_tree_propagation(probabilities, guidance)
        self.assertEqual(output.shape, probabilities.shape)
        self.assertFalse(output.requires_grad)
        self.assertTrue(bool(torch.isfinite(output).all()))
        torch.testing.assert_close(
            output.sum(dim=1),
            torch.ones_like(output[:, 0]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_dual_trees_are_parallel_and_independent(self) -> None:
        probabilities = _probabilities(height=5, width=6, device="cuda")
        shallow = torch.randn(1, 2, 5, 6, device="cuda")
        deep = torch.randn(1, 4, 3, 3, device="cuda")
        first = propagate_dual_tree_pseudo_labels(
            probabilities,
            shallow,
            deep,
            window_size=(4, 4),
            stride=(2, 2),
            shallow_sigma=0.3,
            deep_sigma=0.3,
        )
        swapped = propagate_dual_tree_pseudo_labels(
            probabilities,
            deep,
            shallow,
            window_size=(4, 4),
            stride=(2, 2),
            shallow_sigma=0.3,
            deep_sigma=0.3,
        )
        self.assertIsInstance(first, DualTreePseudoLabels)
        torch.testing.assert_close(first.shallow, swapped.deep, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(first.deep, swapped.shallow, rtol=1e-5, atol=1e-6)

    def test_cuda_filter_matches_small_boruvka_reference(self) -> None:
        probabilities = torch.tensor(
            [
                [
                    [[0.70, 0.10, 0.25], [0.15, 0.55, 0.20]],
                    [[0.20, 0.65, 0.25], [0.25, 0.15, 0.50]],
                    [[0.10, 0.25, 0.50], [0.60, 0.30, 0.30]],
                ]
            ],
            dtype=torch.float32,
        )
        guidance = torch.tensor(
            [[[[0.00, 0.21, 0.77], [0.08, 0.52, 1.41]]]],
            dtype=torch.float32,
        )
        sigma = 0.73
        for temperature in (DEFAULT_SPATIAL_TEMPERATURE, None):
            with self.subTest(spatial_temperature=temperature):
                expected = _reference_tree_filter(
                    probabilities,
                    guidance,
                    sigma,
                    spatial_temperature=temperature,
                )
                actual = windowed_tree_propagation(
                    probabilities.cuda(),
                    guidance.cuda(),
                    window_size=(2, 3),
                    stride=(2, 3),
                    sigma=sigma,
                    spatial_temperature=temperature,
                ).cpu()
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def _boruvka_tree(
    vertex_count: int,
    edges: Sequence[Tuple[int, int, float]],
) -> List[Tuple[int, int, float]]:
    parent = list(range(vertex_count))
    rank = [0] * vertex_count

    def find(vertex: int) -> int:
        if parent[vertex] != vertex:
            parent[vertex] = find(parent[vertex])
        return parent[vertex]

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    tree: List[Tuple[int, int, float]] = []
    tree_count = vertex_count
    while tree_count > 1:
        cheapest = [-1] * vertex_count
        for edge_index, (source, target, weight) in enumerate(edges):
            left, right = find(source), find(target)
            if left == right:
                continue
            if cheapest[left] == -1 or edges[cheapest[left]][2] > weight:
                cheapest[left] = edge_index
            if cheapest[right] == -1 or edges[cheapest[right]][2] > weight:
                cheapest[right] = edge_index
        for edge_index in cheapest:
            if edge_index == -1:
                continue
            source, target, weight = edges[edge_index]
            left, right = find(source), find(target)
            if left == right:
                continue
            tree.append((source, target, weight))
            union(left, right)
            tree_count -= 1
    return tree


def _reference_tree_filter(
    probabilities: torch.Tensor,
    guidance: torch.Tensor,
    sigma: float,
    *,
    spatial_temperature: Optional[float] = DEFAULT_SPATIAL_TEMPERATURE,
) -> torch.Tensor:
    _, classes, height, width = probabilities.shape
    vertex_count = height * width
    flat_guidance = guidance[0].reshape(guidance.shape[1], vertex_count)
    edges: List[Tuple[int, int, float]] = []
    for row in range(height - 1):
        for column in range(width):
            source = row * width + column
            target = (row + 1) * width + column
            distance = float((flat_guidance[:, source] - flat_guidance[:, target]).square().sum())
            edges.append((source, target, distance))
    for row in range(height):
        for column in range(width - 1):
            source = row * width + column
            target = row * width + column + 1
            distance = float((flat_guidance[:, source] - flat_guidance[:, target]).square().sum())
            edges.append((source, target, distance))

    tree = _boruvka_tree(vertex_count, edges)
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(vertex_count)]
    for source, target, distance in tree:
        spatial_distance = math.hypot(
            source // width - target // width,
            source % width - target % width,
        )
        spatial_exponent = (
            0.0
            if spatial_temperature is None
            else spatial_distance / spatial_temperature
        )
        affinity = math.exp(-distance / sigma - spatial_exponent)
        adjacency[source].append((target, affinity))
        adjacency[target].append((source, affinity))

    path_affinity = torch.zeros(vertex_count, vertex_count, dtype=torch.float64)
    for source in range(vertex_count):
        stack = [(source, -1, 1.0)]
        while stack:
            current, parent, affinity = stack.pop()
            path_affinity[source, current] = affinity
            for target, edge_affinity in adjacency[current]:
                if target != parent:
                    stack.append((target, current, affinity * edge_affinity))

    flat_probabilities = probabilities[0].reshape(classes, vertex_count).double()
    filtered = flat_probabilities @ path_affinity.T
    filtered = filtered / path_affinity.sum(dim=1).unsqueeze(0)
    return filtered.reshape(1, classes, height, width).float()


if __name__ == "__main__":
    unittest.main()
