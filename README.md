# Flipbook

## Core idea

You pass a model function f(theta, t) → y. Flipbook reads an MCMC chain and animates model curves for chosen walkers across steps, with optional data overlays. Works for microlensing magnification/flux or any time‑series model.

## Public API

flipbook.animate_walkers(model_fn, t, chain, *, log_prob=None, out=None, …) -> matplotlib.animation.FuncAnimation

* model_fn(theta, t) -> y
 - theta: 1D array (ndim,)
 - t: 1D array (T,)
 - y: 1D array (T,)
* t: 1D array of epochs (even or uneven)
* chain: 3D array (nsteps, nwalkers, ndim) or a Chain object (see below)
* log_prob: 2D array (nsteps, nwalkers) or None (for top‑K selection)
* keyword options (sane defaults):
 - data_t, data_y, data_err: arrays for overlay (optional)
 - param_transform: callable(theta) -> theta2 (e.g., log10→physical)
 - walkers: 'all' | int | array-like (IDs to render)
 - step_slice: slice | (start, stop) | array-like | None
 - thin: int (frame thinning)
 - topk_by_logp: int | None (per‑step top‑K walkers by log_prob)
 - vectorized: bool (model_fn supports batch: model_fn(Thetas[K,:], t)->Y[K,T])
 - n_jobs: int (for non‑vectorized batched evaluation)
 - chunk_size: int (batch size when evaluating walkers)
 - percentile_bands: list[float] | None (e.g., [0.5, 0.9])
 - per_step_aggregate: {'median','mean',None}
 - color_by: {'walker','logp',None}
 - alpha: float (per‑curve transparency, default 0.15)
 - max_curves_per_frame: int (cap for performance)
 - fps: int (default 15), dpi: int (default 120)
 - writer: {'ffmpeg','imagemagick','pillow'} | mpl writer
 - out: str | Path | None ('flipbook.mp4' or 'flipbook.gif'); None returns the animation
 - title: str | callable(step_idx)->str
 - y_label: str (default 'Model')
 - ylim: tuple | None
 - progress: bool (tqdm)

flipbook.animate_from_emcee(model_fn, t, sampler, *, out=None, **kwargs) -> animation

* Convenience wrapper to pull chain and log_prob from emcee.EnsembleSampler.

flipbook.snapshot_step(model_fn, t, chain, *, step, walkers='all', …) -> matplotlib.figure.Figure

* Static frame for reports.

flipbook.precompute_curves(model_fn, t, chain, *, steps=None, walkers=None, vectorized=False, n_jobs=1, chunk_size=32, param_transform=None) -> generator | ndarray

* Optional helper to precompute curves if you want to export frames offline.

## Types and helpers

### Chain container (optional convenience)

* class Chain: chain: (nsteps,nwalkers,ndim), log_prob: (nsteps,nwalkers) | None, labels: list[str] | None

### Selection utilities

* select_walkers(chain, walkers='all', topk_by_logp=None, rng=None) -> ndarray[int]
* compute_bands(curves, percentiles=[50, 90]) -> dict[str, (lo, hi)]

### Behavior (per frame)

* Pick frame step → select walkers → transform params if requested
* Evaluate model_fn for those walkers (vectorized or batched/parallel)
* Plot:
 - data overlay with error bars (once)
 - per‑walker curves (thin, alpha)
 - aggregate curve (median/mean)
 - optional percentile bands via fill_between
* Update title/labels; write frame

## Performance notes

* Vectorize if your model allows; otherwise batch and optionally parallelize with n_jobs.
* Limit max_curves_per_frame and thin frames to keep it smooth.
* Provide a small LRU cache for model_fn results keyed by (theta hash, id(t)) to avoid redundant evals when step thinning or reusing walkers.
* For very heavy models, allow “preview mode”: fewer walkers, more thinning, then a high‑quality render.

### Backends and writers
- Matplotlib animation requires a writer; recommended:
  - mp4: `ffmpeg` (install with `brew install ffmpeg` or `apt-get install ffmpeg`)
  - gif: `imagemagick` or `pillow`
- In headless environments, set `MPLBACKEND=Agg`.
- Flipbook auto-detects available writers and raises a helpful error if none is found.

### Vectorization and batching
- If your `model_fn` accepts batched parameters (`Thetas[K, :]`), set `vectorized=True` for large speedups.
- Otherwise, flipbook evaluates in chunks (`chunk_size`, default 32) and can parallelize with `n_jobs`.

### Determinism
- Selections that involve randomness accept `rng`/`random_state`. With the same inputs, Flipbook produces identical animations.

## Installation

```
pip install -e .[dev,test]
```

Optional runtime dependencies for writers:
- ffmpeg (mp4), imagemagick/pillow (gif)

## Troubleshooting

- “No MovieWriter available”: install a writer (e.g., ffmpeg) and ensure it is on PATH.
- “Model output shape mismatch”: ensure `model_fn(theta, t)` returns a 1D array with the same length as `t`.
- “Animations are slow”: try `vectorized=True`, reduce `max_curves_per_frame`, increase `thin`, or use batching with `n_jobs>1`.

## Roadmap

- CLI entry point (`flipbook-cli`) for quick animations from `.npy/.npz` chains.
- Export frames to disk and compose video externally.
- Optional panel/ipywidgets UI for interactive exploration.

## Example

Minimal microlensing magnification animation (VBMicrolensing wrapper):

```
def model_vbm(theta_log10, t):
    # theta_log10 = [log10 s, log10 q, u0, alpha_deg, log10 rho, log10 tE, t0]
    return magnification_from_engine(theta_log10, t) # your existing wrapper

from flipbook import animate_walkers

anim = animate_walkers(
    model_vbm, t_obs, chain, log_prob=logp,
    data_t=t_obs, data_y=mag_obs, data_err=mag_err,
    percentile_bands=[0.5, 0.9], per_step_aggregate='median',
    walkers=100, thin=2, vectorized=False, n_jobs=6,
    fps=12, out='flipbook.mp4', title=lambda i: f'Step {i}', y_label='Magnification',
    progress=True,
)
```

## Why this is universal

Flipbook is agnostic to model semantics; any time‑series model fits as long as you provide model_fn(theta, t) → y.

## Documentation

Build the docs locally (macOS zsh):

```bash
python -m pip install -e .[docs]
make -C docs html
# open docs/_build/html/index.html
```

Using tox:

```bash
python -m pip install tox
tox -e build_docs
```

Hosted docs: add this repo to Read the Docs and keep `.readthedocs.yaml` at the project root. RTD will install with the `docs` extra and build `docs/conf.py`.
