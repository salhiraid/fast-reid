"""FastReID wrapper vs FastReID's own test-time path, with a randomly initialised checkpoint (no real weights needed)."""
import numpy as np
import pytest
import torch

from reid_eval.encoders import build_encoder
from reid_eval.encoders.fastreid_encoder import FastReIDEncoder


@pytest.fixture(scope="module")
def fake_ckpt(tmp_path_factory):
    from fastreid.config import get_cfg
    from fastreid.modeling.meta_arch import build_model
    cfg = get_cfg()
    cfg.merge_from_file("configs/VERIWild/bagtricks_R50-ibn.yml")
    cfg.MODEL.BACKBONE.PRETRAIN = False; cfg.MODEL.HEADS.NUM_CLASSES = 50; cfg.MODEL.DEVICE = "cpu"
    torch.manual_seed(0)
    m = build_model(cfg).eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_(0, 0.1); mod.running_var.uniform_(0.5, 1.5)
    p = tmp_path_factory.mktemp("ck") / "fake.pth"
    torch.save({"model": m.state_dict()}, p)
    return cfg, m, p


def test_dim_parity_and_in_place_safety(fake_ckpt):
    cfg, ref, p = fake_ckpt
    enc = build_encoder("fastreid_veriwild_r50ibn", weights=str(p), device="cpu")
    d = enc.describe()
    assert d["embedding_dim"] == 2048 and d["input_size"] == [256, 256] and d["checkpoint_sha256"]
    img = (np.random.RandomState(0).rand(224, 224, 3) * 255).astype(np.uint8)
    x = enc.preprocess(img, 20, 30, "unpad_stretch")[None]
    assert x.shape == (1, 3, 256, 256) and 0 <= x.min() and x.max() <= 255          # raw 0-255 RGB; the model normalises
    before = x.clone()
    a = enc.encode(x)
    assert torch.equal(x, before)                                                    # caller's batch untouched (flip TTA safe)
    with torch.no_grad():
        b = ref({"images": before.clone()})                                          # FastReID's own eval forward (what ReidEvaluator uses)
    assert torch.allclose(a, b, atol=1e-5)


def test_wrong_config_fails_loudly(fake_ckpt):
    with pytest.raises(RuntimeError, match="does not match"):
        FastReIDEncoder("x", "configs/VeRi/sbs_R50-ibn.yml", str(fake_ckpt[2]), "cpu")


def test_weights_required():
    with pytest.raises(ValueError):
        build_encoder("fastreid_veriwild_r50ibn")
