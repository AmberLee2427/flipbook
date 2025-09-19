"""Core animation routines for :mod:`flipbook`."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.animation as mpl_animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike, NDArray

from .types import Array1D, Array2D, Chain, StepSelection, WalkerSelection, as_chain
from .utils.selection import (
    compute_percentile_bands,
    resolve_step_indices,
    resolve_walker_indices,
    select_topk_by_log_prob,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is an optional dependency
    tqdm = None


@dataclass
class _FrameResult:
    """Container holding all information required to render a frame."""

    step_index: int
    walker_indices: NDArray[np.int_]
    curves: Array2D
    aggregate: Array1D | None
    percentile_bands: dict[str, tuple[Array1D, Array1D]]
    colors: list[tuple[float, float, float, float]] | None
    log_prob: Array1D | None


def _apply_param_transform(
    parameters: Array2D,
    transform: Callable[[Array1D], Array1D] | Callable[[Array2D], Array2D] | None,
) -> Array2D:
    """Apply an optional parameter transformation."""

    if transform is None:
        return parameters
    try:
        transformed = np.asarray(transform(parameters))
    except Exception:
        transformed = None
    if transformed is not None and transformed.shape == parameters.shape:
        return transformed.astype(float, copy=False)

    transformed_rows = np.empty_like(parameters, dtype=float)
    for idx, theta in enumerate(parameters):
        transformed_rows[idx] = np.asarray(transform(theta), dtype=float)
    return transformed_rows


def _hash_theta(theta: Array1D) -> int:
    """Compute a stable hash for a parameter vector."""

    return hash(theta.tobytes())


def _evaluate_model_curves(
    model_fn: Callable[[Array1D, Array1D], Array1D],
    t: Array1D,
    parameters: Array2D,
    *,
    vectorized: bool,
    chunk_size: int,
    n_jobs: int,
    cache: dict[tuple[int, int], Array1D],
) -> Array2D:
    """Evaluate the model for a batch of parameter vectors."""

    if parameters.size == 0:
        return np.empty((0, t.size), dtype=float)

    if vectorized:
        try:
            candidate = np.asarray(model_fn(parameters, t), dtype=float)
        except Exception:
            candidate = None
        if candidate is not None:
            if candidate.shape != (parameters.shape[0], t.size):
                raise ValueError("Vectorized model function returned unexpected shape")
            if not np.all(np.isfinite(candidate)):
                raise ValueError("Model function returned non-finite values")
            return candidate

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer")

    results = np.empty((parameters.shape[0], t.size), dtype=float)
    t_identifier = id(t)

    def evaluate_single(theta: Array1D) -> Array1D:
        key = (_hash_theta(theta), t_identifier)
        cached = cache.get(key)
        if cached is not None:
            return cached.copy()
        curve = np.asarray(model_fn(theta, t), dtype=float)
        if curve.shape != t.shape:
            raise ValueError(
                "Model function must return an array matching the shape of 't'"
            )
        if curve.ndim != 1:
            raise ValueError("Model output must be one-dimensional")
        if not np.all(np.isfinite(curve)):
            raise ValueError("Model function returned non-finite values")
        cache[key] = curve.copy()
        return curve

    if n_jobs == 1 or parameters.shape[0] == 1:
        for idx, theta in enumerate(parameters):
            results[idx] = evaluate_single(theta)
        return results

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        start = 0
        while start < parameters.shape[0]:
            end = min(start + chunk_size, parameters.shape[0])
            futures = [
                executor.submit(evaluate_single, parameters[i])
                for i in range(start, end)
            ]
            for offset, future in enumerate(futures):
                results[start + offset] = future.result()
            start = end
    return results


def _compute_aggregate(curves: Array2D, mode: str | None) -> Array1D | None:
    """Compute per-step aggregate curves."""

    if mode is None or curves.size == 0:
        return None
    if mode == "median":
        return np.median(curves, axis=0)
    if mode == "mean":
        return np.mean(curves, axis=0)
    raise ValueError("per_step_aggregate must be one of {'median', 'mean', None}")


def _frame_data_generator(
    chain: Chain,
    model_fn: Callable[[Array1D, Array1D], Array1D],
    t: Array1D,
    step_indices: NDArray[np.int_],
    walker_indices: NDArray[np.int_],
    *,
    param_transform: Callable[[Array1D], Array1D] | Callable[[Array2D], Array2D] | None,
    vectorized: bool,
    chunk_size: int,
    n_jobs: int,
    topk_by_logp: int | None,
    max_curves_per_frame: int | None,
    per_step_aggregate: str | None,
    percentile_bands: Sequence[float] | None,
    color_by: str | None,
    progress: bool,
) -> Iterator[_FrameResult]:
    """Yield frame specifications for downstream rendering."""

    cache: dict[tuple[int, int], Array1D] = {}

    if max_curves_per_frame is not None and max_curves_per_frame <= 0:
        raise ValueError("max_curves_per_frame must be positive when provided")

    percentile_bands = list(percentile_bands) if percentile_bands is not None else None

    if color_by not in {None, "walker", "logp"}:
        raise ValueError("color_by must be one of {'walker', 'logp', None}")
    color_map = None
    norm = None
    if color_by == "walker":
        color_map = cm.get_cmap("viridis")
        norm = Normalize(vmin=0, vmax=max(chain.nwalkers - 1, 1))
    elif color_by == "logp":
        if chain.log_prob is None:
            raise ValueError("log_prob is required when color_by='logp'")
        subset = chain.log_prob[np.ix_(step_indices, walker_indices)]
        vmin = float(np.min(subset))
        vmax = float(np.max(subset))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("log_prob contains non-finite values")
        if vmin == vmax:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        color_map = cm.get_cmap("plasma")

    step_iterator: Iterable[int]
    if progress and tqdm is not None:
        step_iterator = tqdm(step_indices, desc="Generating frames")
    else:
        step_iterator = step_indices

    for step_index in step_iterator:
        selected = walker_indices
        if topk_by_logp is not None:
            selected = select_topk_by_log_prob(
                chain.log_prob, step_index, selected, topk_by_logp
            )
        if max_curves_per_frame is not None and selected.size > max_curves_per_frame:
            selected = selected[:max_curves_per_frame]
        parameters = chain.chain[step_index, selected, :]
        parameters = _apply_param_transform(parameters, param_transform)
        curves = _evaluate_model_curves(
            model_fn,
            t,
            parameters,
            vectorized=vectorized,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            cache=cache,
        )
        aggregate = _compute_aggregate(curves, per_step_aggregate)
        percentile_info = compute_percentile_bands(curves, percentile_bands)

        colors = None
        if color_map is not None and norm is not None:
            if color_by == "walker":
                values = selected
            else:
                assert chain.log_prob is not None
                values = chain.log_prob[step_index, selected]
            colors = [color_map(norm(float(value))) for value in values]

        log_prob_values = None
        if chain.log_prob is not None:
            log_prob_values = chain.log_prob[step_index, selected]

        yield _FrameResult(
            step_index=step_index,
            walker_indices=selected,
            curves=curves,
            aggregate=aggregate,
            percentile_bands=percentile_info,
            colors=colors,
            log_prob=log_prob_values,
        )


def animate_walkers(
    model_fn: Callable[[Array1D, Array1D], Array1D],
    t: ArrayLike,
    chain: ArrayLike | Chain,
    *,
    log_prob: ArrayLike | None = None,
    out: str | Path | None = None,
    data_t: ArrayLike | None = None,
    data_y: ArrayLike | None = None,
    data_err: ArrayLike | None = None,
    param_transform: Callable[[Array1D], Array1D]
    | Callable[[Array2D], Array2D]
    | None = None,
    walkers: WalkerSelection = "all",
    step_slice: StepSelection | None = None,
    thin: int = 1,
    topk_by_logp: int | None = None,
    vectorized: bool = False,
    n_jobs: int = 1,
    chunk_size: int = 32,
    percentile_bands: Sequence[float] | None = None,
    per_step_aggregate: str | None = None,
    color_by: str | None = None,
    alpha: float = 0.15,
    max_curves_per_frame: int | None = None,
    fps: int = 15,
    dpi: int = 120,
    writer: str | mpl_animation.AbstractMovieWriter | None = None,
    title: str | Callable[[int], str] | None = None,
    y_label: str = "Model",
    ylim: tuple[float, float] | None = None,
    progress: bool = False,
) -> mpl_animation.FuncAnimation:
    """Animate walkers from an MCMC chain for a given model function.

    Parameters
    ----------
    model_fn : callable
        Callable implementing ``f(theta, t) -> y``.
    t : array_like
        One-dimensional array of time samples.
    chain : array_like or Chain
        Either a raw chain array with shape ``(nsteps, nwalkers, ndim)`` or a
        :class:`~flipbook.types.Chain` instance.
    log_prob : array_like, optional
        Log-probability values associated with ``chain``. Only required when the
        chain object does not already include them.
    out : str or Path, optional
        Output path. When provided, the animation is saved to disk using the
        requested writer. The animation object is still returned for further
        manipulation.
    data_t, data_y, data_err : array_like, optional
        Observational data to overlay. ``data_err`` is interpreted as symmetric
        uncertainties when supplied.
    param_transform : callable, optional
        Transformation applied to each walker prior to evaluating ``model_fn``.
    walkers : {'all', int, sequence of int}, optional
        Walker selection specification.
    step_slice : slice, tuple, sequence of int, optional
        Steps to include in the animation.
    thin : int, optional
        Frame thinning factor applied after ``step_slice``.
    topk_by_logp : int, optional
        If provided, restricts each frame to the top-K walkers by log
        probability.
    vectorized : bool, optional
        Indicates that ``model_fn`` supports vectorized evaluation with an
        ``(nwalkers, ndim)`` parameter array.
    n_jobs : int, optional
        Number of threads to use for non-vectorized evaluation.
    chunk_size : int, optional
        Number of walkers evaluated together when ``n_jobs > 1``.
    percentile_bands : sequence of float, optional
        Percentile bands to shade in each frame.
    per_step_aggregate : {'median', 'mean', None}, optional
        Aggregate curve to highlight in each frame.
    color_by : {'walker', 'logp', None}, optional
        Strategy used to color individual walker curves.
    alpha : float, optional
        Base transparency applied to walker curves.
    max_curves_per_frame : int, optional
        Upper bound on the number of curves rendered per frame.
    fps : int, optional
        Target frames per second for the animation.
    dpi : int, optional
        Resolution when writing to disk.
    writer : str or matplotlib writer, optional
        Animation writer. When ``None`` a reasonable default is inferred from
        the output file extension.
    title : str or callable, optional
        Static title string or callable ``title(step_index) -> str`` executed per
        frame.
    y_label : str, optional
        Y-axis label for the plot.
    ylim : tuple, optional
        Y-axis limits.
    progress : bool, optional
        Display a progress bar while generating frames. Requires :mod:`tqdm`.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The generated animation object.
    """

    if fps <= 0:
        raise ValueError("fps must be a positive integer")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be within [0, 1]")

    chain_obj = as_chain(chain, log_prob=log_prob)
    t_array = np.asarray(t, dtype=float)
    if t_array.ndim != 1:
        raise ValueError("t must be a one-dimensional array")

    nsteps, nwalkers = chain_obj.nsteps, chain_obj.nwalkers
    step_indices = resolve_step_indices(nsteps, step_slice=step_slice, thin=thin)
    walker_indices = resolve_walker_indices(nwalkers, walkers)
    if walker_indices.size == 0:
        raise ValueError("No walkers selected for animation")

    if max_curves_per_frame is None:
        max_artists = walker_indices.size
    else:
        max_artists = max_curves_per_frame

    frame_iter = _frame_data_generator(
        chain_obj,
        model_fn,
        t_array,
        step_indices,
        walker_indices,
        param_transform=param_transform,
        vectorized=vectorized,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        topk_by_logp=topk_by_logp,
        max_curves_per_frame=max_curves_per_frame,
        per_step_aggregate=per_step_aggregate,
        percentile_bands=percentile_bands,
        color_by=color_by,
        progress=progress,
    )

    fig, ax = plt.subplots()
    ax.set_xlabel("t")
    ax.set_ylabel(y_label)
    ax.set_xlim(float(t_array.min()), float(t_array.max()))
    if ylim is not None:
        ax.set_ylim(*ylim)

    if data_t is not None and data_y is not None:
        data_t_array = np.asarray(data_t, dtype=float)
        data_y_array = np.asarray(data_y, dtype=float)
        if data_t_array.shape != data_y_array.shape:
            raise ValueError("data_t and data_y must share the same shape")
        if data_err is not None:
            data_err_array = np.asarray(data_err, dtype=float)
            if data_err_array.shape != data_y_array.shape:
                raise ValueError("data_err must match the shape of data_y")
            ax.errorbar(
                data_t_array,
                data_y_array,
                yerr=data_err_array,
                fmt="o",
                color="black",
                alpha=0.8,
                label="data",
            )
        else:
            ax.plot(data_t_array, data_y_array, "o", color="black", label="data")

    walker_lines: list[Line2D] = [
        ax.plot([], [], lw=1.2, alpha=alpha)[0] for _ in range(max_artists)
    ]
    aggregate_line = None
    if per_step_aggregate is not None:
        (aggregate_line,) = ax.plot(
            [], [], color="black", lw=2.0, label=per_step_aggregate
        )

    percentile_artists: dict[str, PolyCollection] = {}

    def init() -> list:
        artists: list = []
        for line in walker_lines:
            line.set_data([], [])
            artists.append(line)
        if aggregate_line is not None:
            aggregate_line.set_data([], [])
            artists.append(aggregate_line)
        return artists

    def update(frame: _FrameResult) -> list:
        artists: list = []
        curves = frame.curves
        for idx, line in enumerate(walker_lines):
            if idx < curves.shape[0]:
                line.set_data(t_array, curves[idx])
                if frame.colors is not None:
                    line.set_color(frame.colors[idx])
                line.set_alpha(alpha)
            else:
                line.set_data([], [])
            artists.append(line)

        if aggregate_line is not None:
            if frame.aggregate is None:
                aggregate_line.set_data([], [])
            else:
                aggregate_line.set_data(t_array, frame.aggregate)
            artists.append(aggregate_line)

        for artist in percentile_artists.values():
            artist.remove()
        percentile_artists.clear()
        for label, (lo, hi) in frame.percentile_bands.items():
            band = ax.fill_between(t_array, lo, hi, alpha=0.2, label=label)
            percentile_artists[label] = band
            artists.append(band)

        if callable(title):
            ax.set_title(title(frame.step_index))
        elif isinstance(title, str):
            ax.set_title(title)

        return artists

    animation = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=frame_iter,
        init_func=init,
        blit=False,
        interval=1000.0 / float(fps),
    )

    if out is not None:
        output_path = Path(out)
        resolved_writer = _resolve_writer(writer, output_path.suffix)
        animation.save(str(output_path), writer=resolved_writer, dpi=dpi, fps=fps)

    return animation


def _resolve_writer(
    writer: str | mpl_animation.AbstractMovieWriter | None,
    suffix: str,
) -> str | mpl_animation.AbstractMovieWriter:
    """Resolve the animation writer from user input and file suffix."""

    if writer is not None:
        if isinstance(writer, str) and not mpl_animation.writers.is_available(writer):
            raise ValueError(f"Requested writer '{writer}' is not available")
        return writer
    suffix = suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        candidate = "ffmpeg"
    elif suffix == ".gif":
        candidate = "imagemagick"
    else:
        candidate = "pillow"
    if not mpl_animation.writers.is_available(candidate):
        raise ValueError(
            "No suitable animation writer available. Install ffmpeg, imagemagick, or pillow."
        )
    return candidate


def snapshot_step(
    model_fn: Callable[[Array1D, Array1D], Array1D],
    t: ArrayLike,
    chain: ArrayLike | Chain,
    *,
    step: int,
    walkers: WalkerSelection = "all",
    log_prob: ArrayLike | None = None,
    param_transform: Callable[[Array1D], Array1D]
    | Callable[[Array2D], Array2D]
    | None = None,
    vectorized: bool = False,
    n_jobs: int = 1,
    chunk_size: int = 32,
    percentile_bands: Sequence[float] | None = None,
    per_step_aggregate: str | None = None,
    color_by: str | None = None,
    alpha: float = 0.15,
    max_curves_per_frame: int | None = None,
    title: str | None = None,
    y_label: str = "Model",
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Generate a static snapshot of a specific step from the chain.

    Parameters
    ----------
    model_fn : callable
        Callable implementing ``f(theta, t) -> y``.
    t : array_like
        One-dimensional array of time samples.
    chain : array_like or Chain
        Chain data or :class:`~flipbook.types.Chain` instance.
    step : int
        Step index to visualise.
    walkers : {'all', int, sequence of int}, optional
        Walker selection specification.
    log_prob : array_like, optional
        Log-probability values for the chain.
    param_transform : callable, optional
        Optional transformation applied prior to model evaluation.
    vectorized : bool, optional
        Indicates that ``model_fn`` supports vectorized evaluation.
    n_jobs : int, optional
        Number of worker threads for non-vectorized evaluation.
    chunk_size : int, optional
        Batch size for threaded evaluation.
    percentile_bands : sequence of float, optional
        Percentile bands to shade in the snapshot.
    per_step_aggregate : {'median', 'mean', None}, optional
        Aggregate curve to highlight.
    color_by : {'walker', 'logp', None}, optional
        Strategy used to colour individual walker curves.
    alpha : float, optional
        Transparency applied to walker curves.
    max_curves_per_frame : int, optional
        Upper bound on the number of curves rendered.
    title : str, optional
        Title for the generated figure.
    y_label : str, optional
        Y-axis label.
    ylim : tuple, optional
        Y-axis limits.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing the snapshot.
    """

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be within [0, 1]")

    chain_obj = as_chain(chain, log_prob=log_prob)
    t_array = np.asarray(t, dtype=float)
    if t_array.ndim != 1:
        raise ValueError("t must be a one-dimensional array")

    step_indices = np.asarray([int(step)], dtype=int)
    walker_indices = resolve_walker_indices(chain_obj.nwalkers, walkers)

    frame_iter = _frame_data_generator(
        chain_obj,
        model_fn,
        t_array,
        step_indices,
        walker_indices,
        param_transform=param_transform,
        vectorized=vectorized,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        topk_by_logp=None,
        max_curves_per_frame=max_curves_per_frame,
        per_step_aggregate=per_step_aggregate,
        percentile_bands=percentile_bands,
        color_by=color_by,
        progress=False,
    )

    frame = next(frame_iter)

    fig, ax = plt.subplots()
    ax.set_xlabel("t")
    ax.set_ylabel(y_label)
    ax.set_xlim(float(t_array.min()), float(t_array.max()))
    if ylim is not None:
        ax.set_ylim(*ylim)

    for idx, curve in enumerate(frame.curves):
        color = frame.colors[idx] if frame.colors is not None else None
        ax.plot(t_array, curve, color=color, alpha=alpha)

    if frame.aggregate is not None:
        ax.plot(t_array, frame.aggregate, color="black", lw=2.0)

    for label, (lo, hi) in frame.percentile_bands.items():
        ax.fill_between(t_array, lo, hi, alpha=0.2, label=label)

    if title is not None:
        ax.set_title(title)

    return fig


def precompute_curves(
    model_fn: Callable[[Array1D, Array1D], Array1D],
    t: ArrayLike,
    chain: ArrayLike | Chain,
    *,
    steps: StepSelection | None = None,
    walkers: WalkerSelection = "all",
    log_prob: ArrayLike | None = None,
    param_transform: Callable[[Array1D], Array1D]
    | Callable[[Array2D], Array2D]
    | None = None,
    vectorized: bool = False,
    n_jobs: int = 1,
    chunk_size: int = 32,
    topk_by_logp: int | None = None,
    max_curves_per_frame: int | None = None,
) -> Iterator[dict[str, object]]:
    """Pre-compute model curves for later use.

    Parameters
    ----------
    model_fn : callable
        Callable implementing ``f(theta, t) -> y``.
    t : array_like
        One-dimensional array of time samples.
    chain : array_like or Chain
        Chain data or :class:`~flipbook.types.Chain` instance.
    steps : slice, tuple, sequence of int, optional
        Steps to pre-compute. When ``None`` all steps are considered.
    walkers : {'all', int, sequence of int}, optional
        Walker selection specification.
    log_prob : array_like, optional
        Log-probability values for the chain.
    param_transform : callable, optional
        Optional transformation applied prior to model evaluation.
    vectorized : bool, optional
        Indicates that ``model_fn`` supports vectorized evaluation.
    n_jobs : int, optional
        Number of worker threads for non-vectorized evaluation.
    chunk_size : int, optional
        Batch size for threaded evaluation.
    topk_by_logp : int, optional
        If provided, restricts walkers to the top-K by log probability per step.
    max_curves_per_frame : int, optional
        Maximum number of curves retained per step.

    Returns
    -------
    generator of dict
        Generator yielding dictionaries with keys ``'step_index'``,
        ``'walker_indices'``, ``'curves'``, and ``'log_prob'``.
    """

    chain_obj = as_chain(chain, log_prob=log_prob)
    t_array = np.asarray(t, dtype=float)
    if t_array.ndim != 1:
        raise ValueError("t must be a one-dimensional array")

    step_indices = resolve_step_indices(chain_obj.nsteps, step_slice=steps, thin=1)
    walker_indices = resolve_walker_indices(chain_obj.nwalkers, walkers)

    frame_iter = _frame_data_generator(
        chain_obj,
        model_fn,
        t_array,
        step_indices,
        walker_indices,
        param_transform=param_transform,
        vectorized=vectorized,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        topk_by_logp=topk_by_logp,
        max_curves_per_frame=max_curves_per_frame,
        per_step_aggregate=None,
        percentile_bands=None,
        color_by=None,
        progress=False,
    )

    def generator() -> Iterator[dict[str, object]]:
        for frame in frame_iter:
            yield {
                "step_index": frame.step_index,
                "walker_indices": frame.walker_indices,
                "curves": frame.curves,
                "log_prob": frame.log_prob,
            }

    return generator()


def animate_from_emcee(
    model_fn: Callable[[Array1D, Array1D], Array1D],
    t: ArrayLike,
    sampler: object,
    *,
    out: str | Path | None = None,
    **kwargs: Any,
) -> mpl_animation.FuncAnimation:
    """Animate walkers directly from an :mod:`emcee` sampler.

    Parameters
    ----------
    model_fn : callable
        Callable implementing ``f(theta, t) -> y``.
    t : array_like
        One-dimensional array of time samples.
    sampler : object
        ``emcee`` sampler providing ``get_chain`` and optionally ``get_log_prob``.
    out : str or Path, optional
        Output filename passed through to :func:`animate_walkers`.
    **kwargs
        Additional keyword arguments forwarded to :func:`animate_walkers`.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The generated animation object.

    Raises
    ------
    TypeError
        If ``sampler`` does not provide the expected ``get_chain`` method.
    """

    if not hasattr(sampler, "get_chain"):
        raise TypeError("sampler must provide a 'get_chain' method")
    chain_array = sampler.get_chain()
    log_prob_array = None
    if hasattr(sampler, "get_log_prob"):
        log_prob_array = sampler.get_log_prob()
    return animate_walkers(
        model_fn,
        t,
        chain_array,
        log_prob=log_prob_array,
        out=out,
        **kwargs,
    )
