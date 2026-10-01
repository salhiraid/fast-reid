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


def _legacy(fake_ckpt, tmp_path, pixel_mean=None):
    """Checkpoint in the format of the FastReID model zoo: heads.classifier.weight + stored pixel_mean/pixel_std."""
    cfg, ref, p = fake_ckpt
    state = {k: v.clone() for k, v in ref.state_dict().items()}
    state["heads.classifier.weight"] = state.pop("heads.weight")
    state["pixel_mean"] = torch.tensor(pixel_mean or cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1)
    state["pixel_std"] = torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1)
    out = tmp_path / "legacy.pth"
    torch.save({"model": state}, out)
    return out


def test_legacy_model_zoo_checkpoint_loads(fake_ckpt, tmp_path):
    cfg, ref, p = fake_ckpt
    enc = build_encoder("fastreid_veriwild_r50ibn", weights=str(_legacy(fake_ckpt, tmp_path)), device="cpu")
    assert any("heads.classifier.weight" in n for n in enc.describe()["notes"])
    assert any("pixel_mean" in n for n in enc.describe()["notes"])
    x = enc.preprocess((np.random.RandomState(1).rand(224, 224, 3) * 255).astype(np.uint8), 0, 0, "letterbox")[None]
    with torch.no_grad():
        assert torch.allclose(enc.encode(x), ref({"images": x.clone()}), atol=1e-5)   # same features as the current-format load


def test_legacy_checkpoint_with_other_normalisation_is_refused(fake_ckpt, tmp_path):
    bad = _legacy(fake_ckpt, tmp_path, pixel_mean=[0.5 * 255, 0.5 * 255, 0.5 * 255])
    with pytest.raises(RuntimeError, match="pixel_mean"):
        build_encoder("fastreid_veriwild_r50ibn", weights=str(bad), device="cpu")
