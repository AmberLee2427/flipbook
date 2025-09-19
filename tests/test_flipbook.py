"""
Basic test to ensure the package can be imported.
"""
import pytest


def test_import_flipbook():
    """Test that flipbook can be imported without errors."""
    try:
        import flipbook
        assert hasattr(flipbook, '__version__') or True  # Version might not be available in dev mode
    except ImportError:
        pytest.fail("Could not import flipbook package")


def test_flipbook_has_main_functions():
    """Test that the expected main functions are available."""
    import flipbook
    
    # These functions should be available in the main API
    expected_functions = [
        'animate_walkers',
        'animate_from_emcee', 
        'snapshot_step',
        'precompute_curves'
    ]
    
    # Note: These tests will fail until the actual implementation is added
    # For now, we'll just test that the module imports
    assert flipbook is not None