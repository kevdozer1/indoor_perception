"""Shared pytest fixtures and configuration."""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (requires model download or heavy computation)",
    )


@pytest.fixture
def sample_rgb_image():
    """Fixture providing a sample RGB image."""
    import numpy as np
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth_map():
    """Fixture providing a sample depth map."""
    import numpy as np
    # Planar surface at 2 meters with some noise
    depth = np.ones((480, 640), dtype=np.float32) * 2.0
    noise = np.random.randn(480, 640) * 0.1
    return np.clip(depth + noise, 0.5, 5.0)


@pytest.fixture
def sample_intrinsics():
    """Fixture providing sample camera intrinsics."""
    from indoor_perception.projection import compute_intrinsics_matrix
    return compute_intrinsics_matrix(fx=525.0, fy=525.0, cx=319.5, cy=239.5)
