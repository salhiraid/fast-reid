"""Bin definitions (configs/bins_v1.yaml) and the joint-cell indexing shared by accumulation and aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml

from .common import sha256_file

AXES = ("delta_position", "delta_azimuth", "occlusion", "keypoint_iou")
N_HIST = 200   # per-cell histogram bins over [-1, 1]
N_FINE = 2000  # global fine histogram bins over [-1, 1]


def _fmt(x):
    return "inf" if np.isinf(x) else f"{x:g}"


def edge_labels(edges):
    n = len(edges) - 1
    return [f"[{_fmt(edges[i])},{_fmt(edges[i + 1])}" + (")" if i < n - 1 else "]") for i in range(n)]


@dataclass
class Bins:
    version: str
    pos_edges: np.ndarray
    az_edges: np.ndarray
    occ_names: list
    kp_edges: np.ndarray
    kpmin_edges: np.ndarray
    far_targets: list
    min_pos_pairs: int
    min_objects: int
    vavg_pos: int
    vavg_neg: int
    boot_resamples: int
    boot_seed: int
    boot_level: float
    sha256: str = ""
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        fl = lambda k: np.asarray([float(x) for x in d[k]], dtype=np.float64)
        b = cls(version=str(d["version"]), pos_edges=fl("delta_position_m"), az_edges=fl("delta_azimuth_deg"),
                occ_names=list(d["occlusion"]), kp_edges=fl("shared_keypoints_iou"), kpmin_edges=fl("min_visible_keypoints"),
                far_targets=[float(x) for x in d["far_targets"]], min_pos_pairs=int(d["min_support"]["positive_pairs"]),
                min_objects=int(d["min_support"]["objects"]), vavg_pos=int(d["video_average_min_pairs"]["positive"]),
                vavg_neg=int(d["video_average_min_pairs"]["negative"]), boot_resamples=int(d["bootstrap"]["resamples"]),
                boot_seed=int(d["bootstrap"]["seed"]), boot_level=float(d["bootstrap"]["level"]),
                sha256=sha256_file(path), raw=d)
        assert len(b.occ_names) == 3 and len(b.far_targets) == 2
        for e in (b.pos_edges, b.az_edges, b.kp_edges, b.kpmin_edges):
            assert np.all(np.diff(e) > 0), "bin edges must increase"
        return b

    # number of KNOWN bins per axis; each axis also has an "unknown" bin at the last index (except occlusion)
    @property
    def n_known(self):
        return (len(self.pos_edges) - 1, len(self.az_edges) - 1, 3, len(self.kp_edges) - 1)

    @property
    def cell_shape(self):  # (P+1, A+1, 3, K+1)
        p, a, o, k = self.n_known
        return (p + 1, a + 1, o, k + 1)

    @property
    def n_cells(self):
        return int(np.prod(self.cell_shape))

    @property
    def n_kpmin(self):
        return len(self.kpmin_edges)  # known bins + unknown

    @property
    def thr_names(self):
        return [f"t{-int(np.round(np.log10(t)))}" if t > 0 else "t?" for t in self.far_targets]  # t2, t3

    def labels(self, axis):
        if axis == "delta_position":
            return edge_labels(self.pos_edges) + ["unknown"]
        if axis == "delta_azimuth":
            return edge_labels(self.az_edges) + ["unknown"]
        if axis == "occlusion":
            return list(self.occ_names)
        if axis == "keypoint_iou":
            return edge_labels(self.kp_edges) + ["unknown"]
        if axis == "min_visible_keypoints":
            return edge_labels(self.kpmin_edges) + ["unknown"]
        raise KeyError(axis)

    def is_unknown(self, axis, idx):
        return axis != "occlusion" and idx == len(self.labels(axis)) - 1
