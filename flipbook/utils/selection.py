"""Utilities for selecting walkers, steps, and computing summary statistics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..types import Array2D, Array3D, StepSelection, WalkerSelection


def resolve_step_indices(
    nsteps: int,
    step_slice: StepSelection | None = None,
    *,
    thin: int = 1,
) -> NDArray[np.int_]:
    """Return the concrete step indices that should be animated.

    Parameters
    ----------
    nsteps : int
        Total number of steps available in the chain.
    step_slice : slice, tuple, sequence of int, optional
        Specification of the steps that should be considered. When ``None``
        all steps are included. Tuple inputs are interpreted as ``(start, stop)``
        bounds similar to :class:`slice`.
    thin : int, optional
        Step thinning factor. A value of ``thin=2`` keeps every other step.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of unique, sorted step indices.

    Raises
    ------
    ValueError
        If the thinning factor is invalid or requested indices fall outside the
        available range.
    """

    if thin <= 0:
        raise ValueError("'thin' must be a positive integer")

    if step_slice is None:
        indices = np.arange(nsteps, dtype=int)
    elif isinstance(step_slice, slice):
        indices = np.arange(nsteps, dtype=int)[step_slice]
    elif isinstance(step_slice, tuple):
        if len(step_slice) != 2:
            raise ValueError("Step tuple must be of the form (start, stop)")
        start, stop = step_slice
        start_idx = 0 if start is None else int(start)
        stop_idx = nsteps if stop is None else int(stop)
        indices = np.arange(start_idx, stop_idx, dtype=int)
    else:
        indices = np.asarray(step_slice, dtype=int)
        if indices.ndim != 1:
            raise ValueError("step_slice sequence must be one-dimensional")

    indices = np.unique(indices.astype(int))[::thin]
    if indices.size == 0:
        raise ValueError("No steps selected after applying step_slice/thin")
    if (indices < 0).any() or (indices >= nsteps).any():
        raise ValueError("Requested steps fall outside the available range")
    return indices


def resolve_walker_indices(
    nwalkers: int,
    walkers: WalkerSelection = "all",
) -> NDArray[np.int_]:
    """Return concrete walker indices based on a user specification.

    Parameters
    ----------
    nwalkers : int
        Total number of walkers available in the chain.
    walkers : {'all', int, sequence of int}, optional
        Specification of the walkers to include. When ``'all'`` (default) every
        walker is returned. When an integer is provided, the first ``walkers``
        indices are returned. Sequence inputs are validated and de-duplicated.

    Returns
    -------
    numpy.ndarray
        Array containing walker indices sorted in ascending order.

    Raises
    ------
    ValueError
        If the selection is incompatible with the available walkers.
    """

    if isinstance(walkers, str):
        if walkers != "all":
            raise ValueError("String walker specification must be 'all'")
        return np.arange(nwalkers, dtype=int)

    if walkers is None:
        return np.arange(nwalkers, dtype=int)

    if isinstance(walkers, int):
        if walkers <= 0:
            raise ValueError("Walker count must be a positive integer")
        if walkers > nwalkers:
            raise ValueError("Requested more walkers than available")
        return np.arange(walkers, dtype=int)

    indices = np.asarray(list(walkers), dtype=int)
    if indices.ndim != 1:
        raise ValueError("Walker sequence must be one-dimensional")
    if indices.size == 0:
        raise ValueError("Walker sequence cannot be empty")
    if (indices < 0).any() or (indices >= nwalkers).any():
        raise ValueError("Walker indices fall outside the available range")
    return np.unique(indices)


def select_topk_by_log_prob(
    log_prob: Array2D | None,
    step_index: int,
    walker_indices: NDArray[np.int_],
    topk: int | None,
) -> NDArray[np.int_]:
    """Select the top-K walkers for a given step based on log-probability.

    Parameters
    ----------
    log_prob : numpy.ndarray, optional
        Log-probability values with shape ``(nsteps, nwalkers)``.
    step_index : int
        Step identifier into ``log_prob``.
    walker_indices : numpy.ndarray
        Candidate walkers to rank.
    topk : int, optional
        Number of walkers to retain. When ``None`` no filtering is performed.

    Returns
    -------
    numpy.ndarray
        Sorted indices of the selected walkers.

    Raises
    ------
    ValueError
        If ``topk`` is invalid or ``log_prob`` is not available.
    """

    if topk is None:
        return walker_indices
    if topk <= 0:
        raise ValueError("topk_by_logp must be a positive integer")
    if log_prob is None:
        raise ValueError("log_prob must be provided when using topk_by_logp")
    if step_index < 0 or step_index >= log_prob.shape[0]:
        raise ValueError("step_index outside the range of the log_prob array")

    scores = log_prob[step_index, walker_indices]
    order = np.argsort(scores)[::-1]
    return np.sort(walker_indices[order[: min(topk, order.size)]])


def compute_percentile_bands(
    curves: Array3D | Array2D,
    percentiles: Sequence[float] | None,
) -> dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]]:
    """Compute percentile envelopes for a collection of model curves.

    Parameters
    ----------
    curves : numpy.ndarray
        Either a 2-D array with shape ``(ncurves, ntime)`` or a 3-D array with
        shape ``(nsteps, ncurves, ntime)``.
    percentiles : sequence of float, optional
        Percentile levels expressed as fractions between 0 and 1. Each entry
        ``p`` produces a band spanning ``[(1-p)/2, (1+p)/2]`` of the posterior.

    Returns
    -------
    dict
        Mapping of percentile labels to ``(lower, upper)`` envelope arrays.

    Raises
    ------
    ValueError
        If invalid percentile values or array shapes are provided.
    """

    if not percentiles:
        return {}

    curves_array = np.asarray(curves, dtype=float)
    if curves_array.ndim == 3:
        # Assume shape (nsteps, ncurves, ntime); flatten step dimension.
        curves_array = curves_array.reshape(-1, curves_array.shape[-1])
    if curves_array.ndim != 2:
        raise ValueError(
            "curves array must be 2-dimensional for percentile computation"
        )

    percentiles_array = np.asarray(percentiles, dtype=float)
    if percentiles_array.ndim != 1:
        raise ValueError("percentiles must be provided as a one-dimensional sequence")
    if ((percentiles_array <= 0) | (percentiles_array >= 1)).any():
        raise ValueError("percentiles must fall in the open interval (0, 1)")

    bands: dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]] = {}
    for percentile in percentiles_array:
        lower = 50.0 * (1.0 - percentile)
        upper = 50.0 * (1.0 + percentile)
        lo, hi = np.percentile(curves_array, [lower, upper], axis=0)
        key = f"{int(percentile * 100):d}%"
        bands[key] = (lo, hi)
    return bands
