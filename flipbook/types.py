"""Type definitions and lightweight containers used throughout :mod:`flipbook`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

ArrayLike: TypeAlias = npt.ArrayLike
Array1D: TypeAlias = npt.NDArray[np.float64]
Array2D: TypeAlias = npt.NDArray[np.float64]
Array3D: TypeAlias = npt.NDArray[np.float64]
IndexArray: TypeAlias = npt.NDArray[np.integer[Any]]
StepBounds: TypeAlias = Tuple[Optional[int], Optional[int]]

ModelFunction = Callable[[Array1D, Array1D], Array1D]
VectorizedModelFunction = Callable[[Array2D, Array1D], Array2D]

WalkerSelection = Union[str, int, Sequence[int], IndexArray]
StepSelection = Union[slice, StepBounds, Sequence[int], IndexArray]


@dataclass
class Chain:
    """Container holding an MCMC chain and optional metadata."""

    chain: Array3D
    log_prob: Array2D | None = None
    labels: Sequence[str] | None = None

    def __post_init__(self) -> None:
        self.chain = np.asarray(self.chain, dtype=float)
        if self.chain.ndim != 3:
            raise ValueError(
                "Chain array must be 3-dimensional (nsteps, nwalkers, ndim)"
            )
        if self.log_prob is not None:
            log_prob_array = np.asarray(self.log_prob, dtype=float)
            if log_prob_array.shape != self.chain.shape[:2]:
                raise ValueError(
                    "log_prob must match the first two dimensions of the chain array"
                )
            self.log_prob = log_prob_array

    @property
    def nsteps(self) -> int:
        """Number of steps stored in the chain."""

        return int(self.chain.shape[0])

    @property
    def nwalkers(self) -> int:
        """Number of walkers stored in the chain."""

        return int(self.chain.shape[1])

    @property
    def ndim(self) -> int:
        """Dimensionality of the model parameters."""

        return int(self.chain.shape[2])

    def select_steps(self, indices: ArrayLike) -> Array3D:
        """Return a view of the chain restricted to the requested steps."""

        step_indices = np.asarray(indices, dtype=int)
        return self.chain[step_indices]

    def select_walkers(self, indices: ArrayLike) -> Array3D:
        """Return a view of the chain restricted to the requested walkers."""

        walker_indices = np.asarray(indices, dtype=int)
        return self.chain[:, walker_indices]


def as_chain(chain: ArrayLike | Chain, log_prob: ArrayLike | None = None) -> Chain:
    """Normalize input into a :class:`Chain` instance.

    Parameters
    ----------
    chain : array_like or Chain
        Either a raw chain array with shape ``(nsteps, nwalkers, ndim)`` or an
        existing :class:`Chain` instance.
    log_prob : array_like, optional
        Log-probability array with shape ``(nsteps, nwalkers)``. Only required
        when ``chain`` does not already carry a ``log_prob`` attribute.

    Returns
    -------
    Chain
        Normalized chain container.

    Raises
    ------
    ValueError
        If the provided arrays do not have compatible dimensions or if
        ``log_prob`` is supplied more than once.
    """

    if isinstance(chain, Chain):
        if log_prob is not None:
            if chain.log_prob is not None:
                raise ValueError("log_prob provided twice")
            return Chain(
                chain.chain,
                log_prob=np.asarray(log_prob, dtype=float),
                labels=chain.labels,
            )
        return chain

    chain_array = np.asarray(chain, dtype=float)
    if chain_array.ndim != 3:
        raise ValueError("Chain input must be 3-dimensional")
    log_prob_array: Array2D | None = None
    if log_prob is not None:
        log_prob_array = np.asarray(log_prob, dtype=float)
        if log_prob_array.shape != chain_array.shape[:2]:
            raise ValueError(
                "Provided log_prob array must match chain steps and walkers"
            )
    return Chain(chain_array, log_prob=log_prob_array)
