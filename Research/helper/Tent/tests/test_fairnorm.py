"""CPU tests for the Batch-size-1 FairNorm baseline."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RESEARCH_ROOT = REPOSITORY_ROOT / "Research"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from sota_FairNorm import (  # noqa: E402
    BATCH_SIZE,
    METHOD,
    NORMALIZATION,
    SUMMARY_FIELDS,
    configure_fairnorm,
    fairnorm_logits,
    rebuild_summary,
    task_coordinates,
)


class TinySegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout2d(p=0.75)
        self.head = nn.Conv2d(1, 4, kernel_size=1)
        with torch.no_grad():
            self.bn.running_mean.fill_(3.0)
            self.bn.running_var.fill_(4.0)
            self.bn.weight.fill_(1.5)
            self.bn.bias.fill_(-0.25)
            self.head.weight.copy_(
                torch.tensor([1.0, -1.0, 0.5, -0.5]).reshape(4, 1, 1, 1)
            )
            self.head.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.1]))

    def forward(self, images: torch.Tensor):
        features = self.dropout(self.bn(images))
        return features, self.head(features)


class FairNormTests(unittest.TestCase):
    def make_model(self) -> TinySegmenter:
        model = TinySegmenter()
        names = configure_fairnorm(model)
        self.assertEqual(names, ("bn",))
        return model

    def test_configuration_freezes_model_and_only_bn_uses_batch_mode(self) -> None:
        model = self.make_model()
        self.assertFalse(model.training)
        self.assertTrue(model.bn.training)
        self.assertFalse(model.dropout.training)
        self.assertFalse(model.bn.track_running_stats)
        self.assertIsNone(model.bn.running_mean)
        self.assertIsNone(model.bn.running_var)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_singleton_bn_matches_current_slice_statistics(self) -> None:
        model = self.make_model()
        image = torch.tensor([[[[0.0, 1.0], [4.0, 7.0]]]])
        actual = model.bn(image)
        mean = image.mean(dim=(0, 2, 3), keepdim=True)
        variance = image.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        expected = (image - mean) / torch.sqrt(variance + model.bn.eps)
        expected = expected * 1.5 - 0.25
        torch.testing.assert_close(actual, expected)

    def test_predictions_are_finite_and_have_no_cross_slice_state(self) -> None:
        model = self.make_model()
        first = torch.tensor([[[[0.0, 1.0], [4.0, 7.0]]]])
        second = torch.tensor([[[[-5.0, 2.0], [3.0, 20.0]]]])
        counter_before = model.bn.num_batches_tracked.detach().clone()
        first_logits = fairnorm_logits(model, first)
        second_logits = fairnorm_logits(model, second)
        repeated_logits = fairnorm_logits(model, first)
        self.assertTrue(torch.isfinite(first_logits).all())
        self.assertTrue(torch.isfinite(second_logits).all())
        torch.testing.assert_close(first_logits, repeated_logits)
        torch.testing.assert_close(model.bn.num_batches_tracked, counter_before)

    def test_non_singleton_batch_is_rejected(self) -> None:
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, r"\[1,C,H,W\]"):
            fairnorm_logits(model, torch.zeros((2, 1, 4, 4)))

    def test_task_mapping_and_partial_aggregation(self) -> None:
        self.assertEqual(BATCH_SIZE, 1)
        self.assertEqual(task_coordinates(0), (1, 0, "A"))
        self.assertEqual(task_coordinates(23), (2, 0, "D"))
        self.assertEqual(task_coordinates(59), (3, 4, "D"))
        with self.assertRaises(ValueError):
            task_coordinates(60)

        with tempfile.TemporaryDirectory(prefix="fairnorm-test-") as temporary:
            root = Path(temporary)
            for task_id in (0, 59):
                task_dir = root / "shards" / f"task_{task_id}"
                task_dir.mkdir(parents=True)
                summary = {field: "" for field in SUMMARY_FIELDS}
                summary.update(
                    {
                        "method": METHOD,
                        "task_id": task_id,
                        "batch_size": BATCH_SIZE,
                        "normalization": NORMALIZATION,
                    }
                )
                (task_dir / "completion.json").write_text(
                    json.dumps({"status": "complete", "summary": summary}),
                    encoding="utf-8",
                )

            status = rebuild_summary(root, require_complete=False)
            self.assertEqual(status["completed_tasks"], 2)
            self.assertFalse(status["complete"])
            self.assertNotIn(0, status["missing_task_ids"])
            self.assertNotIn(59, status["missing_task_ids"])
            with (root / "run_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["task_id"]) for row in rows], [0, 59])
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                rebuild_summary(root, require_complete=True)


if __name__ == "__main__":
    unittest.main()
