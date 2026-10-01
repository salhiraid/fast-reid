"""Encoder registry: `--model <name>` selects one. Factories import their heavy dependencies lazily."""
from __future__ import annotations

from typing import Callable, Dict

from .base import BaseEncoder, Encoder

_REGISTRY: Dict[str, Callable[..., Encoder]] = {}


def register(name: str):
    def deco(factory):
        if name in _REGISTRY:
            raise KeyError(f"encoder {name!r} registered twice")
        _REGISTRY[name] = factory
        return factory
    return deco


def available_encoders():
    return sorted(_REGISTRY)


def build_encoder(name: str, **kwargs) -> Encoder:
    if name not in _REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; available: {available_encoders()}")
    enc = _REGISTRY[name](**kwargs)
    enc.name = name
    return enc


from . import debug_colorhist, fastreid_encoder, clipreid_encoder  # noqa: E402,F401  (registration side effects)

__all__ = ["Encoder", "BaseEncoder", "register", "build_encoder", "available_encoders"]
