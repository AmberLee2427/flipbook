"""Microlensing light-curve fitting demo with Flipbook + VBMicrolensing.

This script:
- Generates a synthetic binary-lens microlensing magnification curve using
  VBMicrolensing (extended source, BinaryMag2)
- Adds heteroscedastic Gaussian noise
- Fits the parameters with emcee (chi-squared likelihood)
- Animates the walker evolution using Flipbook

Run
---
python flipbook/examples/microlensing_fit.py \
  --out flipbook_microlensing.mp4 --static-out flipbook_microlensing.png

Options: see ``-h`` for walkers/steps and truth/initial settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import emcee
import matplotlib.pyplot as plt

try:
    import VBMicrolensing as vbm
except Exception as exc:  # pragma: no cover - external dependency
    raise SystemExit(
        "VBMicrolensing is required for this example. Please install it first."
    ) from exc

from flipbook import animate_walkers


# -----------------------
# VBMicrolensing utilities
# -----------------------


def vbm_magnification_from_log10(
    theta_log10: np.ndarray, t: np.ndarray, engine: vbm.VBMicrolensing
) -> np.ndarray:
    """Compute extended-source magnification A(t) via BinaryMag2.

    Parameters
    ----------
    theta_log10 : ndarray
        Parameter vector ``[log10 s, log10 q, u0, alpha_deg, log10 rho, log10 tE, t0]``.
    t : ndarray
        Epochs array.
    engine : VBMicrolensing.VBMicrolensing
        VBMicrolensing engine instance (with tolerances configured).

    Returns
    -------
    ndarray
        Magnification array ``A(t)``.
    """
    log10_s, log10_q, u0, alpha_deg, log10_rho, log10_tE, t0 = theta_log10
    s = float(10 ** log10_s)
    q = float(10 ** log10_q)
    rho = float(10 ** log10_rho)
    tE = float(10 ** log10_tE)

    alpha = np.deg2rad(alpha_deg)
    tau = (t - t0) / tE
    y1 = -u0 * np.sin(alpha) + tau * np.cos(alpha)
    y2 = u0 * np.cos(alpha) + tau * np.sin(alpha)

    A = np.empty_like(t, dtype=float)
    for i in range(t.size):
        A[i] = engine.BinaryMag2(s, q, float(y1[i]), float(y2[i]), rho)
    return A


# -----------------------
# Likelihood and prior
# -----------------------


def log_prior(theta_log10: np.ndarray, t_bounds: tuple[float, float]) -> float:
    """Broad but practical uniform prior in base-10 log parameterization.

    Parameters
    ----------
    theta_log10 : ndarray
        ``[log10 s, log10 q, u0, alpha_deg, log10 rho, log10 tE, t0]``.
    t_bounds : tuple of float
        (tmin, tmax) bounds for t0.

    Returns
    -------
    float
        0.0 if parameters are within bounds, else -inf.
    """
    log10_s, log10_q, u0, alpha_deg, log10_rho, log10_tE, t0 = theta_log10
    s = 10 ** log10_s
    q = 10 ** log10_q
    rho = 10 ** log10_rho
    tE = 10 ** log10_tE
    tmin, tmax = t_bounds

    if not (0.5 <= s <= 2.0):
        return -np.inf
    if not (1e-4 <= q <= 1.0):
        return -np.inf
    if not (0.0 < u0 <= 1.0):
        return -np.inf
    if not (0.0 <= alpha_deg <= 180.0):
        return -np.inf
    if not (1e-4 <= rho <= 0.05):
        return -np.inf
    if not (0.5 <= tE <= 2.0):  # benchmark regime: tE≈1
        return -np.inf
    if not (tmin <= t0 <= tmax):
        return -np.inf
    return 0.0


def log_likelihood_chi2(
    theta_log10: np.ndarray,
    t: np.ndarray,
    A_obs: np.ndarray,
    A_err: np.ndarray,
    engine: vbm.VBMicrolensing,
) -> float:
    """Chi-squared log-likelihood for magnification data.

    Parameters
    ----------
    theta_log10 : ndarray
        Parameters in base-10 log parameterization.
    t : ndarray
        Times.
    A_obs : ndarray
        Observed magnification values.
    A_err : ndarray
        1-sigma uncertainties on A_obs.
    engine : VBMicrolensing.VBMicrolensing
        Engine instance for magnification.

    Returns
    -------
    float
        -0.5 * chi^2.
    """
    try:
        A_model = vbm_magnification_from_log10(theta_log10, t, engine)
    except Exception:
        return -np.inf
    chi2 = np.sum(((A_obs - A_model) / A_err) ** 2)
    return -0.5 * chi2


def make_log_prob(
    t: np.ndarray,
    A_obs: np.ndarray,
    A_err: np.ndarray,
    engine: vbm.VBMicrolensing,
) -> callable:
    """Create log-probability function for emcee."""

    t_bounds = (float(np.min(t)), float(np.max(t)))

    def _log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta, t_bounds)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_chi2(theta, t, A_obs, A_err, engine)

    return _log_prob


# -----------------------
# Demo main
# -----------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Microlensing fit + Flipbook animation demo")
    p.add_argument("--out", type=str, default="flipbook_microlensing.mp4", help="Output animation (mp4/gif)")
    p.add_argument("--static-out", type=str, default="flipbook_microlensing.png", help="Static summary plot")
    p.add_argument("--walkers", type=int, default=48)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--burn", type=int, default=300)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--noise", type=float, default=0.01, help="Baseline noise scale")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Truth (Attempt 3 baseline, tE=1, t0=0)
    s_true, q_true, rho_true = 0.9, 0.4, 0.005
    alpha_true, tE_true, t0_true, u0_true = 85.0, 1.0, 0.0, 0.15
    theta_true_log10 = np.array([
        np.log10(s_true), np.log10(q_true), u0_true, alpha_true, np.log10(rho_true), np.log10(tE_true), t0_true
    ])

    # Time grid and synthetic data
    t = np.linspace(-2.0, 2.0, 800)
    engine = vbm.VBMicrolensing(); engine.RelTol = 1e-4
    A_true = vbm_magnification_from_log10(theta_true_log10, t, engine)
    A_err = args.noise * np.sqrt(np.maximum(A_true, 1e-6))
    A_obs = A_true + rng.normal(scale=A_err, size=A_true.shape)

    # emcee setup (poor-ish initial guess)
    initial = np.array([
        np.log10(1.0), np.log10(0.3), 0.2, 75.0, np.log10(0.01), np.log10(1.2), 0.3
    ])
    ndim = initial.size
    nwalkers = int(args.walkers)
    scatter = np.array([0.08, 0.08, 0.05, 3.0, 0.15, 0.08, 0.2])
    p0 = initial + scatter * rng.normal(size=(nwalkers, ndim))

    log_prob = make_log_prob(t, A_obs, A_err, engine)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, int(args.steps), progress=True)

    chain = sampler.get_chain(discard=int(args.burn), flat=False)
    logp = sampler.get_log_prob(discard=int(args.burn), flat=False)

    # Static summary plot (best fit over data)
    flat_chain = chain.reshape(-1, ndim)
    flat_logp = logp.reshape(-1)
    best = flat_chain[np.argmax(flat_logp)]
    A_best = vbm_magnification_from_log10(best, t, engine)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.errorbar(t, A_obs, yerr=A_err, fmt=".k", alpha=0.4, ms=3, label="Observed")
    ax.plot(t, A_true, color="C2", lw=2.0, label="True")
    ax.plot(t, A_best, color="C3", lw=2.0, label="Best (chi2)")
    ax.set_xlabel("t (tE units)")
    ax.set_ylabel("Magnification")
    ax.legend()
    fig.savefig(args.static_out, dpi=160)
    plt.close(fig)

    # Animate walkers via Flipbook
    def model_fn(theta_log10: np.ndarray, tt: np.ndarray) -> np.ndarray:
        return vbm_magnification_from_log10(theta_log10, tt, engine)

    animate_walkers(
        model_fn,
        t,
        chain,
        log_prob=logp,
        walkers=min(64, chain.shape[1]),
        thin=2,
        fps=12,
        vectorized=False,
        n_jobs=6,
        chunk_size=24,
        per_step_aggregate="median",
        percentile_bands=[0.5, 0.9],
        title="Microlensing fit — Flipbook",
        out=args.out,
        y_label="Magnification",
        progress=True,
    )
    print(f"Saved {args.out} and {args.static_out}")


if __name__ == "__main__":
    main()

