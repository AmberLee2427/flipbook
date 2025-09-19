"""Tests for the public :mod:`flipbook` API."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.animation as mpl_animation
import matplotlib.figure as mpl_figure
import numpy as np

import flipbook


def _linear_model(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    theta_array = np.asarray(theta)
    if theta_array.ndim == 1:
        slope, intercept = theta_array
        return slope * t + intercept
    slopes = theta_array[:, 0][:, None]
    intercepts = theta_array[:, 1][:, None]
    return slopes * t[None, :] + intercepts


def _sample_chain() -> tuple[np.ndarray, np.ndarray]:
    nsteps, nwalkers = 6, 4
    base = np.stack(
        [
            np.linspace(0.5, 1.5, nsteps),
            np.linspace(0.0, 0.2, nsteps),
        ],
        axis=-1,
    )
    chain = np.empty((nsteps, nwalkers, 2))
    for walker in range(nwalkers):
        chain[:, walker, :] = base + 0.1 * walker
    log_prob = np.linspace(-5.0, 0.0, nsteps * nwalkers).reshape(nsteps, nwalkers)
    return chain, log_prob


def test_animate_walkers_returns_animation():
    t = np.linspace(0.0, 1.0, 20)
    chain, log_prob = _sample_chain()

    animation = flipbook.animate_walkers(
        _linear_model,
        t,
        chain,
        log_prob=log_prob,
        walkers=[0, 1, 2],
        step_slice=slice(0, 5),
        thin=2,
        vectorized=True,
        percentile_bands=[0.5],
        per_step_aggregate="median",
        color_by="walker",
    )

    assert isinstance(animation, mpl_animation.FuncAnimation)


def test_snapshot_step_produces_figure():
    t = np.linspace(0.0, 1.0, 15)
    chain, log_prob = _sample_chain()

    figure = flipbook.snapshot_step(
        _linear_model,
        t,
        chain,
        log_prob=log_prob,
        step=2,
        walkers=[0, 1],
        vectorized=True,
        percentile_bands=[0.5, 0.9],
        color_by="logp",
    )

    assert isinstance(figure, mpl_figure.Figure)


def test_precompute_curves_generator():
    t = np.linspace(0.0, 1.0, 10)
    chain, log_prob = _sample_chain()

    generator = flipbook.precompute_curves(
        _linear_model,
        t,
        chain,
        log_prob=log_prob,
        steps=range(3),
        walkers=[0, 1],
        vectorized=False,
        topk_by_logp=1,
        max_curves_per_frame=1,
    )

    item = next(generator)
    assert set(item) == {"step_index", "walker_indices", "curves", "log_prob"}
    assert item["curves"].shape[-1] == t.size


class _StubSampler:
    def __init__(self, chain: np.ndarray, log_prob: np.ndarray | None = None) -> None:
        self._chain = chain
        self._log_prob = log_prob

    def get_chain(self) -> np.ndarray:
        return self._chain

    def get_log_prob(self) -> np.ndarray | None:
        if self._log_prob is None:
            raise AttributeError
        return self._log_prob


def test_animate_from_emcee_uses_sampler():
    t = np.linspace(0.0, 1.0, 12)
    chain, log_prob = _sample_chain()
    sampler = _StubSampler(chain, log_prob)

    animation = flipbook.animate_from_emcee(
        _linear_model,
        t,
        sampler,
        walkers=[0, 1],
        step_slice=(0, 4),
        vectorized=True,
    )

    assert isinstance(animation, mpl_animation.FuncAnimation)


def test_precompute_curves_exhausts_generator():
    t = np.linspace(0.0, 1.0, 6)
    chain, log_prob = _sample_chain()
    generator = flipbook.precompute_curves(
        _linear_model,
        t,
        chain,
        log_prob=log_prob,
        steps=[0, 1],
        walkers=[0, 1, 2],
        vectorized=True,
    )

    entries = list(generator)
    assert len(entries) == 2
    assert entries[0]["curves"].shape[0] <= 3
