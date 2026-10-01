"""FastReID wrapper: the repo's own model, its test-time feature, its test config values.

Test-time feature of `Baseline`: `EmbeddingHead.forward` in eval mode returns the BN-neck feature
(`bn_feat`) whatever MODEL.HEADS.NECK_FEAT says (that option only selects the *training* feature).
This is exactly what `fastreid.evaluation.ReidEvaluator` collects, so we call `model(batch)` unchanged.

FastReID normalises inside the model (`Baseline.preprocess_image`, mean/std on the 0-255 scale), so
`preprocess` returns RGB float 0-255 and the model does the rest; the manifest records the mean/std.
"""
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import torch

from . import register
from .base import BaseEncoder
from ..common import REPO_ROOT, git_commit, sha256_file

PRESETS = {
    "fastreid_veriwild_r50ibn": "configs/VERIWild/bagtricks_R50-ibn.yml",
    "fastreid_veri_sbs_r50ibn": "configs/VeRi/sbs_R50-ibn.yml",
}


class FastReIDEncoder(BaseEncoder):
    def __init__(self, name, config, weights, device=None, trust_checkpoint=False):
        from fastreid.config import get_cfg
        from fastreid.modeling.meta_arch import build_model

        self.name = name
        self.config_path = Path(config) if Path(config).is_absolute() else REPO_ROOT / config
        self.weights_path = Path(weights)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.notes = []

        cfg = get_cfg()
        cfg.merge_from_file(str(self.config_path))
        state = self._load_state(self.weights_path, trust_checkpoint)
        self._adapt_legacy_state(state, cfg)
        # classifier size depends on the training set: read it from the checkpoint so the load can be strict
        cfg.MODEL.HEADS.NUM_CLASSES = int(state["heads.weight"].shape[0]) if "heads.weight" in state else 0
        cfg.MODEL.BACKBONE.PRETRAIN = False  # weights come from our checkpoint; never download ImageNet weights
        cfg.MODEL.DEVICE = str(self.device)
        cfg.MODEL.WEIGHTS = ""
        self.cfg = cfg
        self.input_size = tuple(int(x) for x in cfg.INPUT.SIZE_TEST)

        model = build_model(cfg)
        res = model.load_state_dict(state, strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(f"checkpoint {self.weights_path.name} does not match {self.config_path.name}: "
                               f"missing={res.missing_keys[:8]} unexpected={res.unexpected_keys[:8]}")
        self.model = model.to(self.device).eval()
        self.mean, self.std = list(cfg.MODEL.PIXEL_MEAN), list(cfg.MODEL.PIXEL_STD)
        with torch.no_grad():
            self._dim = int(self.encode(torch.zeros(2, 3, *self.input_size)).shape[1])

    def _adapt_legacy_state(self, state, cfg):
        """Model-zoo checkpoints come from older FastReID versions. Two harmless differences, handled explicitly:
        * the classifier is called `heads.classifier.weight` (now `heads.weight`); it is only used in training, never at test time;
        * `pixel_mean` / `pixel_std` are stored in the checkpoint (now non-persistent buffers). They are NOT silently ignored:
          they must equal the config's values, otherwise the model was trained with another normalisation.
        Anything else that does not match is still an error (strict load below).
        """
        if "heads.weight" not in state and "heads.classifier.weight" in state:
            state["heads.weight"] = state.pop("heads.classifier.weight")
            self.notes.append("legacy checkpoint: heads.classifier.weight loaded as heads.weight (classifier unused at test time)")
        for key, want in (("pixel_mean", cfg.MODEL.PIXEL_MEAN), ("pixel_std", cfg.MODEL.PIXEL_STD)):
            if key in state:
                got = state.pop(key).flatten().float()
                if not torch.allclose(got, torch.tensor(list(want), dtype=torch.float32), atol=1e-2):
                    raise RuntimeError(f"checkpoint {key}={got.tolist()} differs from the config's MODEL.{key.upper()}={list(want)}")
                self.notes.append(f"legacy checkpoint: stored {key} equals the config value (checked, then dropped)")

    @staticmethod
    def _load_state(path, trust):
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=not trust)  # pickle is unsafe: see --trust-checkpoint
        except TypeError:  # torch < 1.13 has no weights_only argument
            ckpt = torch.load(str(path), map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    def to_tensor(self, rgb):
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()  # 0-255, like fastreid ToTensor

    @torch.no_grad()
    def encode(self, batch):
        # copy: Baseline.preprocess_image normalises in place and the caller may reuse the batch (flip TTA)
        return self.model(batch.to(self.device, copy=True)).float().cpu()

    def _config_name(self):
        try:  # Path.is_relative_to needs Python >= 3.9
            return str(self.config_path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.config_path)

    def describe(self):
        return {
            "model_name": self.name, "repo": "https://github.com/JDAI-CV/fast-reid", "repo_commit": git_commit(),
            "checkpoint": self.weights_path.name, "checkpoint_sha256": sha256_file(self.weights_path),
            "checkpoint_source": "record the download URL and licence here (VeRi/VehicleID/VERI-Wild weights are research-only)",
            "config": self._config_name(),
            "feature": "bn_feat (test-time output of EmbeddingHead, the feature ReidEvaluator uses)",
            "input_size": list(self.input_size),
            "normalisation": {"mean": [m / 255 for m in self.mean], "std": [s / 255 for s in self.std], "color": "RGB",
                              "applied_in": "model (Baseline.preprocess_image, on the 0-255 scale)"},
            "precision": "fp32", "embedding_dim": self._dim,
            "device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else platform.processor() or "cpu",
            "torch": torch.__version__,
            "notes": ["no camera/view inputs (FastReID does not use them)",
                      "checkpoint loaded strictly; MODEL.HEADS.NUM_CLASSES read from the checkpoint",
                      "MODEL.BACKBONE.PRETRAIN=False; no AMP; resize = PIL bicubic as in fastreid test transforms"] + self.notes,
        }


def _make(name):
    def factory(weights=None, device=None, config=None, trust_checkpoint=False, **kw):
        if not weights:
            raise ValueError(f"{name} needs --weights <checkpoint.pth>")
        return FastReIDEncoder(name, config or PRESETS[name], weights, device, trust_checkpoint)
    return factory


for _n in PRESETS:
    register(_n)(_make(_n))
