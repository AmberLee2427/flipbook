# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

# Main API functions (to be implemented)
def animate_walkers(model_fn, t, chain, *, log_prob=None, out=None, **kwargs):
    """
    Animate MCMC walkers for a given model function.
    
    This is a placeholder function that will be implemented later.
    """
    raise NotImplementedError("animate_walkers function is not yet implemented")


def animate_from_emcee(model_fn, t, sampler, *, out=None, **kwargs):
    """
    Convenience wrapper to pull chain and log_prob from emcee.EnsembleSampler.
    
    This is a placeholder function that will be implemented later.
    """
    raise NotImplementedError("animate_from_emcee function is not yet implemented")


def snapshot_step(model_fn, t, chain, *, step, walkers='all', **kwargs):
    """
    Create a static frame for reports.
    
    This is a placeholder function that will be implemented later.
    """
    raise NotImplementedError("snapshot_step function is not yet implemented")


def precompute_curves(model_fn, t, chain, *, steps=None, walkers=None, **kwargs):
    """
    Optional helper to precompute curves for offline frame export.
    
    This is a placeholder function that will be implemented later.
    """
    raise NotImplementedError("precompute_curves function is not yet implemented")


__all__ = [
    'animate_walkers',
    'animate_from_emcee', 
    'snapshot_step',
    'precompute_curves'
]
