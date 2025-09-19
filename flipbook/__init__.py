"""Public API for the :mod:`flipbook` package."""

from __future__ import annotations

from ._astropy_init import *  # noqa: F403
from .core import animate_from_emcee, animate_walkers, precompute_curves, snapshot_step

__all__ = [
    "animate_from_emcee",
    "animate_walkers",
    "precompute_curves",
    "snapshot_step",
]
