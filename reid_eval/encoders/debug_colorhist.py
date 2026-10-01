"""Tiny deterministic encoder (4x4 colour grid) for pipeline tests. Not a ReID model."""
import numpy as np
import torch

from . import register
from .base import BaseEncoder


class ColorGridEncoder(BaseEncoder):
    input_size = (32, 32)

    def to_tensor(self, rgb):
        return torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)

    def encode(self, batch):
        return torch.nn.functional.adaptive_avg_pool2d(batch, 4).flatten(1)

    def describe(self):
        return {"model_name": self.name, "repo": None, "repo_commit": None, "checkpoint": None, "checkpoint_sha256": None,
                "config": None, "feature": "4x4 average colour grid (debug only)", "input_size": list(self.input_size),
                "normalisation": {"mean": [0, 0, 0], "std": [1, 1, 1], "color": "RGB", "scale": "0-1"},
                "precision": "fp32", "embedding_dim": 48, "device": "cpu", "notes": ["debug encoder, not a ReID model"]}


@register("debug_colorgrid")
def _factory(**kw):
    return ColorGridEncoder()
