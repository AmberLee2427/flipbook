# AGENTS.md - Development Guide for AI Assistants

## Project Overview

**flipbook** is a Python package for animating MCMC chains as model curves evolve across walkers and steps. It's designed to be universal - any time-series model that can be expressed as `f(theta, t) → y` can be animated.

### Core Concept
- Pass a model function `f(theta, t) → y`
- Provide an MCMC chain (3D array: nsteps × nwalkers × ndim)
- Get animated plots showing how model curves evolve across the chain
- Perfect for microlensing, but works for any time-series model

### Key Contracts (must be upheld)
- Model function signature: `y = model_fn(theta, t)`
  - `theta`: 1D numpy array of shape `(ndim,)`
  - `t`: 1D numpy array of shape `(T,)`
  - `y`: 1D numpy array of shape `(T,)` (no NaNs/inf)
- Vectorized model (optional): `Y = model_fn(Thetas, t)`
  - `Thetas`: 2D array `(K, ndim)` → `Y`: 2D `(K, T)`
- Chain array: 3D `(nsteps, nwalkers, ndim)` (float64)
- Log prob array (optional): 2D `(nsteps, nwalkers)` (float64)
- Time axis `t` is fixed for every frame of a given animation

## Development Standards & Tools

### Documentation Style
- **Use NumPy docstring format exclusively**
- All public functions must have comprehensive docstrings
- Include `Parameters`, `Returns`, `Examples`, and `Notes` sections
- Follow astropy documentation standards

Example docstring template:
```python
def animate_walkers(model_fn, t, chain, *, log_prob=None, out=None, **kwargs):
    """
    Animate MCMC walkers for a given model function.

    Parameters
    ----------
    model_fn : callable
        Model function with signature f(theta, t) -> y where:
        - theta: 1D array of shape (ndim,) containing model parameters
        - t: 1D array of shape (T,) containing time points
        - y: 1D array of shape (T,) containing model predictions
    t : array_like
        1D array of time points or epochs for evaluation
    chain : array_like
        3D array of shape (nsteps, nwalkers, ndim) containing MCMC chain
    log_prob : array_like, optional
        2D array of shape (nsteps, nwalkers) containing log probabilities
    out : str or Path, optional
        Output filename. If None, returns animation object

    Returns
    -------
    animation : matplotlib.animation.FuncAnimation
        Animation object that can be saved or displayed

    Examples
    --------
    >>> import numpy as np
    >>> from flipbook import animate_walkers
    >>> def linear_model(theta, t):
    ...     slope, intercept = theta
    ...     return slope * t + intercept
    >>> t = np.linspace(0, 10, 100)
    >>> # Assuming you have an MCMC chain...
    >>> anim = animate_walkers(linear_model, t, chain, out='animation.mp4')

    Notes
    -----
    This function supports both vectorized and non-vectorized model functions.
    For performance with heavy models, consider using the `vectorized=True`
    option or `n_jobs` for parallel evaluation.
    """
```

### Code Quality Tools (Already Configured)

1. **ruff** - Primary linter and formatter
   - Replaces flake8, isort, and some pyupgrade functionality
   - Configuration in `pyproject.toml`
   - Run with: `ruff check .` and `ruff format .`

2. **black** - Code formatting (backup/compatibility)
   - Configured for 88-character line length
   - Run with: `black .`

3. **mypy** - Type checking
   - Strict mode enabled for main package
   - Relaxed for tests and docs
   - Run with: `mypy flipbook/`

4. **pytest** - Testing framework
   - Use pytest fixtures, parametrization, and marks
   - Coverage reporting configured
   - Run with: `pytest` or `pytest --cov=flipbook`

5. **pre-commit** - Automated quality checks
   - Install with: `pre-commit install`
   - Run manually: `pre-commit run --all-files`

### Package Architecture

```
flipbook/
├── __init__.py          # Main API exports
├── core.py             # Core animation functions
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── selection.py    # Walker/step selection utilities
│   ├── plotting.py     # Plotting helpers
│   └── performance.py  # Performance optimization utilities
├── types.py            # Type definitions and data structures
└── tests/              # Package-level tests

Implementation starter tasks
- Implement `flipbook.core.animate_walkers` using `matplotlib.animation.FuncAnimation`
- Add vectorized and chunked, optionally parallel, evaluation paths
- Provide `flipbook.core.snapshot_step` for static frames
- Provide `flipbook.core.animate_from_emcee` (extract chain+log_prob)
- Add `flipbook.utils.selection` (walker/step selection, top-K by logp)
- Add `flipbook.utils.plotting` (percentile bands, legends, styles)
- Add `flipbook.utils.performance` (LRU cache, batching, progress)
- Keep a small, optional `joblib`/`concurrent.futures` dependency for n_jobs
```

### Key Design Principles

1. **Universal Model Interface**: `model_fn(theta, t) -> y`
   - theta: 1D parameter array
   - t: 1D time array  
   - y: 1D output array
   - Support both vectorized (`model_fn(thetas[K,:], t) -> y[K,T]`) and scalar modes

2. **Performance First**:
   - Vectorization preferred over loops
   - Optional parallelization with `n_jobs`
   - Chunked evaluation for memory efficiency
   - LRU caching for repeated parameter evaluations

3. **Flexible Selection**:
   - `walkers='all'` | `int` | `array-like` (walker IDs)
   - `step_slice=slice(start, stop)` | `array-like` (step indices)
   - `topk_by_logp=N` (top-K walkers by log probability per step)

4. **Rich Visualization Options**:
   - Per-walker curves with transparency
   - Aggregate curves (median/mean)
   - Percentile bands via `fill_between`
   - Data overlay with error bars
   - Color by walker ID or log probability

## Dependencies & Versions

### Core Dependencies
```python
"numpy>=1.20",
"matplotlib>=3.5.0", 
"astropy>=5.0",        # For package structure, not astronomy
"tqdm>=4.60.0",        # Progress bars
"emcee>=3.0.0",        # MCMC support
```

### Development Dependencies
```python
"pytest>=7.0",         # Testing
"pytest-cov>=4.0",     # Coverage
"pytest-xdist>=3.0",   # Parallel testing
"pytest-mock>=3.10",   # Mocking
"ruff>=0.1.0",         # Linting/formatting
"black>=23.0",         # Backup formatter
"mypy>=1.5",           # Type checking
"pre-commit>=3.0",     # Git hooks
```

## Implementation Priorities

### Phase 1: Core Functionality
1. **Basic Animation Engine** (`flipbook/core.py`)
   - `animate_walkers()` - Main animation function
   - `snapshot_step()` - Static frame generation
   - Basic walker selection and parameter transformation

2. **Utility Functions** (`flipbook/utils/`)
   - `selection.py`: `select_walkers()`, step filtering
   - `plotting.py`: Plotting helpers, color schemes
   - `performance.py`: Vectorization detection, chunking

3. **Type System** (`flipbook/types.py`)
   - `Chain` class for convenient data handling
   - Type aliases and protocols

### Phase 2: Advanced Features
1. **Performance Optimization**
   - Vectorized model evaluation detection
   - Parallel processing with `joblib` or `multiprocessing`
   - Memory-efficient chunked evaluation
   - Result caching

2. **Enhanced Visualization**
   - Percentile bands computation
   - Advanced color schemes
   - Custom title/label functions
   - Multiple data overlay support

3. **Convenience Functions**
   - `animate_from_emcee()` - Direct emcee sampler support
   - `precompute_curves()` - Offline curve generation
   - Export utilities for frames

### Phase 3: Polish & Integration
1. **Documentation**
   - Comprehensive API docs
   - Tutorial notebooks
   - Gallery of examples
   - Performance optimization guide

2. **Testing**
   - Unit tests for all functions
   - Integration tests with real MCMC data
   - Performance benchmarks
   - Cross-platform testing

## Testing Guidelines

### Test Structure
```python
# tests/test_core.py
def test_animate_walkers_basic():
    """Test basic animation functionality."""
    
def test_animate_walkers_with_data_overlay():
    """Test animation with data overlay."""
    
def test_vectorized_vs_scalar_models():
    """Ensure vectorized and scalar models give same results."""

# tests/test_utils.py  
def test_select_walkers():
    """Test walker selection utilities."""
    
def test_compute_percentile_bands():
    """Test percentile band computation."""
```

### Test Data
- Use simple analytical models (linear, quadratic, sinusoidal)
- Generate synthetic MCMC chains with known properties
- Include edge cases (single walker, single step, empty data)
- Add a heavy-model simulator mock that sleeps ~5–10 ms per call to exercise
  batching/parallel paths without requiring external libraries

### Fixtures
```python
@pytest.fixture
def simple_linear_model():
    """Simple linear model for testing."""
    def model(theta, t):
        slope, intercept = theta
        return slope * t + intercept
    return model

@pytest.fixture  
def sample_mcmc_chain():
    """Generate sample MCMC chain for testing."""
    # Return 3D array (nsteps, nwalkers, ndim)
    pass
```

## Performance Considerations

### Model Function Optimization
1. **Vectorization Detection**:
   ```python
   # Test if model_fn supports vectorized input
   test_thetas = np.random.randn(3, ndim)  # 3 parameter sets
   try:
       result = model_fn(test_thetas, t)
       vectorized = result.shape == (3, len(t))
   except:
       vectorized = False
   ```

2. **Chunked Evaluation**:
   - Process walkers in chunks to control memory usage
   - Default chunk_size=32, configurable
   - Balance between memory and function call overhead

3. **Caching Strategy**:
   - Cache results by parameter hash + time array ID
   - Use `functools.lru_cache` or custom cache
   - Clear cache between animation frames if needed

### Animation Performance
1. **Frame Thinning**: Skip steps to reduce frame count
2. **Walker Limiting**: Cap `max_curves_per_frame` for performance
3. **Progressive Quality**: Preview mode vs. final render

## Error Handling

### Input Validation
```python
def validate_model_function(model_fn, t, theta_sample):
    """Validate that model function has correct signature."""
    try:
        result = model_fn(theta_sample, t)
        if not isinstance(result, np.ndarray):
            raise TypeError("Model function must return numpy array")
        if result.shape != t.shape:
            raise ValueError("Model output shape must match time array")
    except Exception as e:
        raise ValueError(f"Model function validation failed: {e}")
```

### Graceful Degradation
- Fall back to scalar evaluation if vectorization fails
- Skip problematic parameter sets with warnings
- Provide helpful error messages for common mistakes

## Integration Notes

### Matplotlib Backend Handling
- Detect and handle different matplotlib backends
- Support for headless environments
- Writer selection (ffmpeg, imagemagick, pillow)
  - Validate writer availability at runtime; provide clear error if missing
  - Document how to install ffmpeg (`brew install ffmpeg`, `apt-get install ffmpeg`)

### Memory Management
- Monitor memory usage during animation
- Implement frame-by-frame generation for large datasets
- Clear intermediate results when possible
 - Warn users if the product `max_curves_per_frame * T` is very large

### Cross-Platform Considerations
- Test on Windows, macOS, and Linux
- Handle path separators correctly
- Consider different default system fonts

### Reproducibility & Determinism
- Accept `random_state`/`rng` to seed any sampling of walkers/steps
- Ensure frame order is deterministic given the same inputs

### Logging & Errors
- Use `warnings.warn` for non-fatal issues (bad model output shape, NaNs)
- Raise `ValueError` for contract violations (shape mismatch, writer missing)
- Prefer returning an animation object even if some walkers fail (skip + warn)

### Minimal public API (must not break)
```
from flipbook import animate_walkers, animate_from_emcee, snapshot_step, precompute_curves
```

### Performance targets (guidance)
- 1k frames with 50 curves per frame and T≈1k should render in minutes, not hours
- Vectorized path should be ~5–10× faster than scalar loop for cheap models

### Documentation TODOs for agents
- Add README usage examples with both scalar and vectorized `model_fn`
- Add guidance for microlensing users (VBMicrolensing/MulensModel wrappers)
- Provide troubleshooting (missing ffmpeg, slow models, memory spikes)

## Environment & Secrets

The `.env` file contains tokens for:
- `PYPI_TOKEN`: For package publishing
- `GITHUB_TOKEN`: For automated releases
- `CODECOV_TOKEN`: For coverage reporting

**Never commit the `.env` file!** It's in `.gitignore` for safety.

## Common Development Commands

```bash
# Install in development mode
pip install -e ".[dev,test]"

# Run all quality checks
ruff check . && ruff format . && mypy flipbook/ && pytest

# Generate coverage report
pytest --cov=flipbook --cov-report=html

# Build documentation
cd docs && make html

# Run pre-commit on all files
pre-commit run --all-files

# Build package for distribution
python -m build
```

## Notes for AI Assistants

1. **Always run tests** after implementing new features
2. **Use type hints** throughout the codebase  
3. **Profile performance** for animation-critical code paths
4. **Consider memory usage** when processing large MCMC chains
5. **Validate inputs** thoroughly - users will pass unexpected data
6. **Write examples** that actually work and demonstrate real usage
7. **Keep the API simple** - hide complexity behind sensible defaults
8. **Think about edge cases** - single walker, single step, empty data, etc.
9. **Make it fast** - this is for interactive data exploration
10. **Document performance trade-offs** clearly in docstrings
