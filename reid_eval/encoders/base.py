from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from ..preprocess import to_rgb_resized


@runtime_checkable
class Encoder(Protocol):
    name: str
    input_size: tuple  # (h, w) the checkpoint expects

    def preprocess(self, img_bgr_224: np.ndarray, pad_x: int, pad_y: int, mode: str) -> torch.Tensor: ...

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, ...) preprocessed images -> (B, D) raw features, NO L2 normalisation."""
        ...

    def describe(self) -> dict:
        """Everything the manifest needs (repo, commit, checkpoint, feature, normalisation, ...)."""
        ...


class BaseEncoder:
    """Shared preprocessing: mode handling + resize; subclasses define `to_tensor` (scaling/normalisation)."""
    name = "base"
    input_size = (256, 256)
    device = torch.device("cpu")

    def to_tensor(self, rgb_uint8: np.ndarray) -> torch.Tensor:  # (H, W, 3) uint8 -> (3, H, W) float
        raise NotImplementedError

    def preprocess(self, img_bgr_224, pad_x, pad_y, mode):
        return self.to_tensor(to_rgb_resized(img_bgr_224, pad_x, pad_y, mode, self.input_size))
