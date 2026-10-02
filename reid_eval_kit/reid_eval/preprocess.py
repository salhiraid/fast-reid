"""Crop preprocessing modes (spec section 4). Encoders add their own normalisation on top of this."""
import numpy as np
from PIL import Image

MODES = ("letterbox", "unpad_stretch")


def crop_for_mode(img_bgr: np.ndarray, pad_x: int, pad_y: int, mode: str) -> np.ndarray:
    """letterbox: stored image as is. unpad_stretch: cut the black padding away (vehicle area only)."""
    if mode == "letterbox":
        return img_bgr
    if mode == "unpad_stretch":
        h, w = img_bgr.shape[:2]
        pad_x, pad_y = int(pad_x), int(pad_y)
        if pad_x < 0 or pad_y < 0 or w - 2 * pad_x < 2 or h - 2 * pad_y < 2:
            raise ValueError(f"invalid letterbox padding pad_x={pad_x}, pad_y={pad_y} for a {w}x{h} crop")
        return img_bgr[pad_y:h - pad_y, pad_x:w - pad_x]
    raise ValueError(f"unknown preprocessing mode {mode!r}; expected one of {MODES}")


def to_rgb_resized(img_bgr: np.ndarray, pad_x: int, pad_y: int, mode: str, size_hw) -> np.ndarray:
    """BGR 224x224 crop -> RGB uint8 (H, W, 3) at the model input size, aspect ratio NOT preserved.

    Uses PIL bicubic like the FastReID / CLIP-ReID test transforms (T.Resize(..., interpolation=3 / BICUBIC)).
    """
    c = crop_for_mode(img_bgr, pad_x, pad_y, mode)[:, :, ::-1]
    h, w = size_hw
    return np.asarray(Image.fromarray(np.ascontiguousarray(c)).resize((w, h), Image.BICUBIC))
