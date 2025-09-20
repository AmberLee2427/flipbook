"""Flipbook demo script.

Animate model curves from an MCMC chain using the flipbook package. Works with
either a synthetic toy model (default) or a user‑provided model function and
chain file.

Examples
--------
1) Synthetic demo (sinusoid), saves anim.mp4:
    python flipbook/examples/flipbook_demo.py --out anim.mp4

2) Animate a saved chain npz (expects keys 'chain' and optionally 'logp'):
    python flipbook/examples/flipbook_demo.py \
        --chain bench_attempt3/trial_0001/chain_chi2.npz \
        --t-linspace -2 2 800 \
        --model vbm --title "Trial 0001 — chi2" \
        --out bench_attempt3/trial_0001/flipbook_walkers_chi2.mp4

3) Use a dotted path to a custom model function:
    python flipbook/examples/flipbook_demo.py \
        --chain my_chain.npz --t-file my_time.npy \
        --model-path mypkg.mymodule:my_model_fn --vectorized \
        --out my_anim.mp4
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from flipbook import animate_walkers


def _load_npz_chain(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    z = np.load(str(path))
    chain = None
    logp = None
    # Common key variants
    for key in ("chain", "chains", "samples"):
        if key in z.files:
            chain = z[key]
            break
    for key in ("logp", "log_prob", "lnprob"):
        if key in z.files:
            logp = z[key]
            break
    if chain is None:
        raise ValueError(f"No chain found in {path} (looked for 'chain', 'chains', 'samples')")
    return chain, logp


def _toy_chain(nsteps: int = 200, nwalkers: int = 64, ndim: int = 3):
    """Generate a simple toy chain with a slow drift on the first parameter.

    Shapes:
    - chain: (nsteps, nwalkers, ndim)
    - logp: None (not used in the toy example)
    """
    rng = np.random.default_rng(42)
    base = np.array([1.0, 0.2, 0.0], dtype=float)  # (ndim,)
    # Drift only along the first parameter across steps
    drift = np.linspace(0.0, 0.2, nsteps, dtype=float)[:, None, None]  # (nsteps,1,1)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)[None, None, :]   # (1,1,ndim)
    # Small walker-wise noise
    noise = 0.05 * rng.normal(size=(nsteps, nwalkers, ndim))
    chain = base[None, None, :] + drift * direction + noise
    logp = None
    return chain, logp


def _toy_model(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    amp, freq, phase = theta
    return amp * np.sin(2 * np.pi * freq * t + phase)


def _vbm_wrapper() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    import Attempt_3.compare_merit_functions_attempt3 as A3  # local project import

    engine = A3.VBMWrapper(reltol=1e-4)

    def model_vbm(theta_log10: np.ndarray, t: np.ndarray) -> np.ndarray:
        return A3.model_magnification(theta_log10, t, engine)

    return model_vbm


def _load_model_from_path(spec: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Load a model function from dotted path 'pkg.mod:func'."""
    if ":" not in spec:
        raise ValueError("--model-path must be in the form 'package.module:function'")
    mod_name, func_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise TypeError(f"Resolved object {func_name} is not callable")
    return fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flipbook demo animator")
    p.add_argument("--chain", type=str, default=None, help="Path to npz with 'chain' and optional 'logp'")
    p.add_argument("--t-file", type=str, default=None, help="Path to time array (.npy or .npz with 't')")
    p.add_argument("--t-linspace", nargs=3, type=float, metavar=("TMIN", "TMAX", "N"), default=None, help="Generate time grid")

    model_grp = p.add_mutually_exclusive_group()
    model_grp.add_argument("--model", type=str, choices=["toy", "vbm"], default="toy", help="Built-in model choice")
    model_grp.add_argument("--model-path", type=str, default=None, help="Custom model function 'pkg.mod:func'")

    p.add_argument("--out", type=str, default="flipbook_demo.mp4", help="Output animation file")
    p.add_argument("--walkers", type=str, default="50", help="'all' or integer count")
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--vectorized", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--aggregate", type=str, choices=["median", "mean", "none"], default="median")
    p.add_argument("--bands", nargs="*", type=float, default=[0.5, 0.9], help="Percentile bands")
    p.add_argument("--title", type=str, default="Flipbook")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load/generate chain
    if args.chain:
        chain, logp = _load_npz_chain(Path(args.chain))
    else:
        chain, logp = _toy_chain()

    # Time array
    if args.t_file:
        p = Path(args.t_file)
        if p.suffix == ".npy":
            t = np.load(p)
        else:
            z = np.load(p)
            t = z["t"] if "t" in z.files else list(z.values())[0]
    elif args.t_linspace is not None:
        tmin, tmax, n = args.t_linspace
        t = np.linspace(float(tmin), float(tmax), int(n))
    else:
        # Default synthetic grid
        t = np.linspace(0, 10, 800)

    # Model function
    if args.model_path:
        model_fn = _load_model_from_path(args.model_path)
    elif args.model == "vbm":
        model_fn = _vbm_wrapper()
    else:
        model_fn = _toy_model

    walkers_sel: str | int
    if args.walkers.strip().lower() == "all":
        walkers_sel = "all"
    else:
        walkers_sel = int(args.walkers)

    agg = None if args.aggregate == "none" else args.aggregate
    bands = None if not args.bands else args.bands

    animate_walkers(
        model_fn,
        t,
        chain,
        log_prob=logp,
        walkers=walkers_sel,
        thin=int(args.thin),
        fps=int(args.fps),
        vectorized=bool(args.vectorized),
        n_jobs=int(args.n_jobs),
        chunk_size=int(args.chunk_size),
        per_step_aggregate=agg,
        percentile_bands=bands,
        title=args.title,
        out=args.out,
        y_label="Magnification" if args.model == "vbm" else "Model",
        progress=True,
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
