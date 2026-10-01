"""CLIP-ReID (https://github.com/Syliz517/CLIP-ReID) wrapper, NON-SIE checkpoints.

NOT TESTED in this repo's CI (needs the CLIP-ReID code and a checkpoint trained without SIE).
It wraps the repo's own `make_model` and calls `model(img)` exactly as its `processor_clipreid.do_inference`
does, with camera/view labels left at None. Set the repo path with --clipreid-repo or $CLIPREID_REPO.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from . import register
from .base import BaseEncoder
from ..common import git_commit, sha256_file

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class CLIPReIDEncoder(BaseEncoder):
    def __init__(self, name, weights, config=None, clipreid_repo=None, num_classes=576, device=None, **_):
        repo = clipreid_repo or os.environ.get("CLIPREID_REPO")
        if not repo or not Path(repo).is_dir():
            raise RuntimeError("CLIP-ReID repo not found: pass --clipreid-repo or set CLIPREID_REPO")
        if not weights:
            raise ValueError(f"{name} needs --weights <checkpoint.pth>")
        sys.path.insert(0, str(repo))
        from config import cfg  # CLIP-ReID's own yacs config
        from model.make_model_clipreid import make_model

        self.name, self.repo, self.weights_path = name, Path(repo), Path(weights)
        self.config_path = Path(config) if config else self.repo / "configs/vehicle/vit_clipreid.yml"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg.merge_from_file(str(self.config_path))
        # no camera / viewpoint ids exist for our data: SIE must be off (needs a checkpoint trained without it)
        cfg.merge_from_list(["MODEL.SIE_CAMERA", False, "MODEL.SIE_VIEW", False, "TEST.WEIGHT", str(self.weights_path)])
        cfg.freeze()
        self.cfg = cfg
        self.input_size = tuple(int(x) for x in cfg.INPUT.SIZE_TEST)
        model = make_model(cfg, num_class=num_classes, camera_num=1, view_num=1)  # num_class must match the checkpoint
        model.load_param(str(self.weights_path))
        self.model = model.to(self.device).eval()
        self.neck_feat = cfg.TEST.NECK_FEAT
        with torch.no_grad():
            self._dim = int(self.encode(torch.zeros(2, 3, *self.input_size)).shape[1])

    def to_tensor(self, rgb):
        t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
        return (t - torch.tensor(CLIP_MEAN).view(3, 1, 1)) / torch.tensor(CLIP_STD).view(3, 1, 1)

    @torch.no_grad()
    def encode(self, batch):
        return self.model(batch.to(self.device), cam_label=None, view_label=None).float().cpu()

    def describe(self):
        return {
            "model_name": self.name, "repo": "https://github.com/Syliz517/CLIP-ReID", "repo_commit": git_commit(self.repo),
            "checkpoint": self.weights_path.name, "checkpoint_sha256": sha256_file(self.weights_path),
            "checkpoint_source": "record the download URL and licence here",
            "config": str(self.config_path), "feature": f"model(img) in eval mode, TEST.NECK_FEAT={self.neck_feat}",
            "input_size": list(self.input_size),
            "normalisation": {"mean": CLIP_MEAN, "std": CLIP_STD, "color": "RGB", "applied_in": "encoder.preprocess"},
            "precision": "fp32", "embedding_dim": self._dim,
            "device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else platform.processor() or "cpu",
            "torch": torch.__version__,
            "notes": ["SIE disabled (MODEL.SIE_CAMERA/SIE_VIEW False); camera/view labels = None"],
        }


register("clipreid_vit_veri")(lambda **kw: CLIPReIDEncoder("clipreid_vit_veri", **kw))
