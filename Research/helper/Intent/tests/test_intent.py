"""Tests for four-class InTEnt single-image adaptation utilities."""

import math
import unittest

import torch
import torch.nn as nn

from Research.helper.Intent import (
    InTEnt,
    balanced_fg_bg_entropy,
    capture_bn_stats,
    categorical_entropy,
    entropy_integration_weights,
    estimate_test_bn_stats,
    integrate_probabilities,
    interpolate_bn_stats,
    load_bn_stats,
    model_logits,
)


class TinySegmenter(nn.Module):
    """Small tuple-output segmenter with deterministic BatchNorm behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.head = nn.Conv2d(1, 4, kernel_size=1, bias=True)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.tensor([1.0, -1.0, 0.5, -0.5]).reshape(4, 1, 1, 1)
            )
            self.head.bias.copy_(torch.tensor([0.2, -0.1, 0.3, -0.2]))

    def forward(self, image: torch.Tensor):
        features = self.bn(image)
        return features, self.head(features)


class FailingSegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.bn(image)
        raise RuntimeError("deliberate forward failure")


def _stats_equal(first, second) -> bool:
    return all(
        torch.equal(first[name][0], second[name][0])
        and torch.equal(first[name][1], second[name][1])
        for name in first
    )


class InTEntUtilityTests(unittest.TestCase):
    def test_model_logits_accepts_supported_outputs(self) -> None:
        logits = torch.randn(1, 4, 3, 3)
        self.assertIs(model_logits(logits), logits)
        self.assertIs(model_logits((torch.zeros(1), logits)), logits)
        self.assertIs(model_logits([torch.zeros(1), logits]), logits)
        self.assertIs(model_logits({"logits": logits}), logits)

        with self.assertRaisesRegex(ValueError, "empty"):
            model_logits(())
        with self.assertRaisesRegex(KeyError, "logits"):
            model_logits({"prediction": logits})
        with self.assertRaisesRegex(TypeError, "Unsupported"):
            model_logits("not a model output")

    def test_capture_load_and_interpolate_statistics(self) -> None:
        model = TinySegmenter()
        with torch.no_grad():
            model.bn.running_mean.fill_(1.5)
            model.bn.running_var.fill_(2.0)
        source = capture_bn_stats(model)
        target = {
            "bn": (
                torch.tensor([5.5]),
                torch.tensor([6.0]),
            )
        }

        mixed = interpolate_bn_stats(source, target, test_fraction=0.25)
        torch.testing.assert_close(mixed["bn"][0], torch.tensor([2.5]))
        torch.testing.assert_close(mixed["bn"][1], torch.tensor([3.0]))

        load_bn_stats(model, target)
        torch.testing.assert_close(model.bn.running_mean, torch.tensor([5.5]))
        torch.testing.assert_close(model.bn.running_var, torch.tensor([6.0]))

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            interpolate_bn_stats(source, target, test_fraction=1.1)
        with self.assertRaisesRegex(ValueError, "layer names"):
            interpolate_bn_stats(source, {}, test_fraction=0.5)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_bn_stats(model, {})

    def test_estimate_test_statistics_restores_all_bn_state(self) -> None:
        model = TinySegmenter().eval()
        with torch.no_grad():
            model.bn.running_mean.fill_(0.75)
            model.bn.running_var.fill_(1.25)
            model.bn.num_batches_tracked.fill_(7)
        model.bn.momentum = 0.25

        original_statistics = capture_bn_stats(model)
        original_batches = model.bn.num_batches_tracked.clone()
        image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        target = estimate_test_bn_stats(model, image)

        torch.testing.assert_close(target["bn"][0], torch.tensor([2.5]))
        torch.testing.assert_close(
            target["bn"][1],
            torch.tensor([5.0 / 3.0]),
        )
        self.assertTrue(_stats_equal(original_statistics, capture_bn_stats(model)))
        self.assertTrue(torch.equal(model.bn.num_batches_tracked, original_batches))
        self.assertEqual(model.bn.momentum, 0.25)
        self.assertFalse(model.training)
        self.assertFalse(model.bn.training)

    def test_estimation_restores_state_when_forward_fails(self) -> None:
        model = FailingSegmenter().train()
        model.bn.eval()
        with torch.no_grad():
            model.bn.running_mean.fill_(0.5)
            model.bn.running_var.fill_(1.5)
            model.bn.num_batches_tracked.fill_(3)
        original_statistics = capture_bn_stats(model)
        original_batches = model.bn.num_batches_tracked.clone()

        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            estimate_test_bn_stats(model, torch.randn(1, 1, 3, 3))

        self.assertTrue(_stats_equal(original_statistics, capture_bn_stats(model)))
        self.assertTrue(torch.equal(model.bn.num_batches_tracked, original_batches))
        self.assertTrue(model.training)
        self.assertFalse(model.bn.training)

    def test_categorical_and_balanced_entropy(self) -> None:
        probabilities = torch.tensor(
            [
                [
                    [[1.00, 0.70, 0.10, 0.05]],
                    [[0.00, 0.10, 0.70, 0.05]],
                    [[0.00, 0.10, 0.10, 0.85]],
                    [[0.00, 0.10, 0.10, 0.05]],
                ]
            ],
            dtype=torch.float32,
        )
        entropy = categorical_entropy(probabilities)
        h_70 = -(0.7 * math.log(0.7) + 3.0 * 0.1 * math.log(0.1))
        h_85 = -(0.85 * math.log(0.85) + 3.0 * 0.05 * math.log(0.05))
        expected_entropy = torch.tensor([[[0.0, h_70, h_70, h_85]]])
        torch.testing.assert_close(entropy, expected_entropy, rtol=1e-5, atol=1e-6)

        expected_balanced = 0.5 * ((0.0 + h_70) / 2.0 + (h_70 + h_85) / 2.0)
        torch.testing.assert_close(
            balanced_fg_bg_entropy(probabilities),
            torch.tensor([expected_balanced]),
            rtol=1e-5,
            atol=1e-6,
        )

        background_only = probabilities[:, :, :, :2]
        torch.testing.assert_close(
            balanced_fg_bg_entropy(background_only),
            categorical_entropy(background_only).mean(dim=(1, 2)),
        )

        half_precision = torch.tensor(
            [[[[1.0]], [[0.0]], [[0.0]], [[0.0]]]],
            dtype=torch.float16,
        )
        self.assertTrue(bool(torch.isfinite(categorical_entropy(half_precision)).all()))

        with self.assertRaisesRegex(ValueError, "non-negative"):
            categorical_entropy(torch.tensor([[[[-0.1]], [[1.1]]]]))
        with self.assertRaisesRegex(ValueError, "background_index"):
            balanced_fg_bg_entropy(probabilities, background_index=4)

    def test_entropy_weights_prefer_low_entropy_and_handle_ties(self) -> None:
        entropies = torch.tensor([[0.2], [0.7], [1.2]])
        weights = entropy_integration_weights(entropies)
        self.assertGreater(weights[0, 0].item(), weights[1, 0].item())
        self.assertGreater(weights[1, 0].item(), weights[2, 0].item())
        torch.testing.assert_close(weights.sum(dim=0), torch.ones(1))

        tied = entropy_integration_weights(torch.ones(4, 2))
        torch.testing.assert_close(tied, torch.full((4, 2), 0.25))

    def test_integrate_probabilities_validates_simplexes(self) -> None:
        branches = torch.tensor(
            [
                [0.70, 0.10, 0.10, 0.10],
                [0.10, 0.20, 0.30, 0.40],
            ],
            dtype=torch.float32,
        ).reshape(2, 1, 4, 1, 1)
        weights = torch.tensor([[0.25], [0.75]])
        integrated = integrate_probabilities(branches, weights)
        expected = 0.25 * branches[0] + 0.75 * branches[1]
        torch.testing.assert_close(integrated, expected)
        torch.testing.assert_close(integrated.sum(dim=1), torch.ones(1, 1, 1))

        invalid_probabilities = branches.clone()
        invalid_probabilities[0, 0, 0, 0, 0] += 0.2
        with self.assertRaisesRegex(ValueError, "sum to one"):
            integrate_probabilities(invalid_probabilities, weights)
        with self.assertRaisesRegex(ValueError, "weights must sum"):
            integrate_probabilities(branches, torch.tensor([[0.25], [0.25]]))


class InTEntWrapperTests(unittest.TestCase):
    def test_default_six_branch_forward_is_episodic(self) -> None:
        model = TinySegmenter().eval()
        with torch.no_grad():
            model.bn.running_mean.fill_(0.2)
            model.bn.running_var.fill_(1.4)
            model.bn.num_batches_tracked.fill_(11)
        source_statistics = capture_bn_stats(model)
        source_batches = model.bn.num_batches_tracked.clone()
        adapter = InTEnt(model)
        image = torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)

        result = adapter.forward_with_details(image)

        self.assertEqual(result.probabilities.shape, (1, 4, 4, 4))
        self.assertEqual(result.branch_probabilities.shape, (6, 1, 4, 4, 4))
        self.assertEqual(result.entropies.shape, (6, 1))
        self.assertEqual(result.weights.shape, (6, 1))
        torch.testing.assert_close(
            result.test_fractions,
            torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        )
        torch.testing.assert_close(result.weights.sum(dim=0), torch.ones(1))
        torch.testing.assert_close(
            result.probabilities.sum(dim=1),
            torch.ones(1, 4, 4),
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertTrue(_stats_equal(source_statistics, capture_bn_stats(model)))
        self.assertTrue(torch.equal(model.bn.num_batches_tracked, source_batches))
        self.assertFalse(model.training)
        self.assertFalse(model.bn.training)

        torch.testing.assert_close(adapter(image), result.probabilities)
        self.assertTrue(_stats_equal(source_statistics, capture_bn_stats(model)))

    def test_wrapper_rejects_non_single_batches_and_models_without_bn(self) -> None:
        adapter = InTEnt(TinySegmenter())
        with self.assertRaisesRegex(ValueError, "single-image"):
            adapter(torch.randn(2, 1, 4, 4))
        with self.assertRaisesRegex(RuntimeError, "BatchNorm"):
            InTEnt(nn.Conv2d(1, 4, kernel_size=1))

    def test_constructor_validates_configuration(self) -> None:
        model = TinySegmenter()
        with self.assertRaisesRegex(ValueError, "duplicates"):
            InTEnt(model, test_fractions=(0.0, 0.0))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            InTEnt(model, test_fractions=(-0.1, 0.5))
        with self.assertRaisesRegex(ValueError, "background_index"):
            InTEnt(model, background_index=-1)


if __name__ == "__main__":
    unittest.main()
