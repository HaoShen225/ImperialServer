"""CPU tests for the global learnable BN-statistic interpolation experiment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RESEARCH_ROOT = REPOSITORY_ROOT / "Research"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from myExperiment1_GlobalLearnableInterpolation import (  # noqa: E402
    GlobalInterpolationState,
    capture_target_statistics,
    first_argmax,
    first_argmin,
    make_grid,
    optimize_slice,
    present_class_macro_dice,
    probabilities_for_current_state,
    replace_batch_norms,
)


class TinySegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.head = nn.Conv2d(1, 4, kernel_size=1)
        with torch.no_grad():
            self.bn.running_mean.fill_(1.0)
            self.bn.running_var.fill_(4.0)
            self.bn.weight.fill_(1.25)
            self.bn.bias.fill_(-0.2)
            self.head.weight.copy_(
                torch.tensor([1.0, -1.0, 0.5, -0.5]).reshape(4, 1, 1, 1)
            )
            self.head.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.1]))

    def forward(self, image: torch.Tensor):
        features = self.bn(image)
        return features, self.head(features)


class GlobalInterpolationTests(unittest.TestCase):
    def make_wrapped(self):
        model = TinySegmenter().eval()
        state = GlobalInterpolationState(device=torch.device("cpu"))
        names = replace_batch_norms(model, state)
        model.eval()
        model.requires_grad_(False)
        return model, state, names

    def test_endpoints_match_source_and_target_normalization(self) -> None:
        model, state, names = self.make_wrapped()
        image = torch.tensor([[[[1.0, 2.0], [3.0, 6.0]]]])
        capture_target_statistics(model, state, image, names)

        state.use_fixed(0.0)
        source_features = model.bn(image)
        expected_source = (image - 1.0) / torch.sqrt(torch.tensor(4.0 + 1e-5))
        expected_source = 1.25 * expected_source - 0.2
        torch.testing.assert_close(source_features, expected_source)

        target_mean = image.mean(dim=(0, 2, 3))
        target_var = image.var(dim=(0, 2, 3), unbiased=False)
        state.use_fixed(1.0)
        target_features = model.bn(image)
        expected_target = (
            image - target_mean.reshape(1, 1, 1, 1)
        ) / torch.sqrt(target_var.reshape(1, 1, 1, 1) + 1e-5)
        expected_target = 1.25 * expected_target - 0.2
        torch.testing.assert_close(target_features, expected_target)

    def test_only_global_a_receives_a_finite_gradient(self) -> None:
        model, state, names = self.make_wrapped()
        image = torch.tensor([[[[0.5, 1.5], [4.0, 8.0]]]])
        capture_target_statistics(model, state, image, names)
        state.reset_parameter(0.5)
        state.use_learned()
        probabilities = probabilities_for_current_state(model, image)
        loss = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1).mean()
        loss.backward()

        self.assertIsNotNone(state.a.grad)
        self.assertTrue(bool(torch.isfinite(state.a.grad)))
        self.assertGreater(abs(float(state.a.grad.item())), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_optimizer_reforward_uses_updated_fraction(self) -> None:
        model, state, names = self.make_wrapped()
        image = torch.tensor([[[[0.25, 1.0], [3.0, 7.0]]]])
        capture_target_statistics(model, state, image, names)
        result = optimize_slice(
            model,
            state,
            image,
            objective="tent",
            initial_s=0.5,
            learning_rate=0.1,
            adaptation_steps=2,
            background_index=0,
            entropy_eps=1e-12,
        )
        self.assertTrue(torch.isfinite(result.probabilities).all())
        self.assertNotEqual(result.final_s, result.initial_s)
        self.assertTrue(0.0 < result.final_s < 1.0)
        self.assertLessEqual(result.final_loss, result.initial_loss + 1e-8)

    def test_grid_and_present_class_dice_rules(self) -> None:
        self.assertEqual(make_grid(0.5), (0.0, 0.5, 1.0))
        self.assertEqual(first_argmin([1.0, 1.0, 2.0]), 0)
        self.assertEqual(first_argmax([0.5, 0.5, 0.1]), 0)

        prediction = torch.tensor([[1, 0], [0, 0]])
        target = torch.tensor([[1, 0], [2, 0]])
        value, count = present_class_macro_dice(prediction, target)
        self.assertEqual(count, 2)
        self.assertGreater(value, 0.49)
        self.assertLess(value, 0.51)

        empty = torch.zeros((2, 2), dtype=torch.long)
        value, count = present_class_macro_dice(empty, empty)
        self.assertEqual(count, 0)
        self.assertTrue(torch.isnan(torch.tensor(value)))


if __name__ == "__main__":
    unittest.main()
