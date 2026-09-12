"""Smoke test: confirms the package installs and imports cleanly.

Deliberately has no ML dependencies (torch etc.) so it runs in CI without
needing GPU or heavy install — real tests land alongside each phase's code.
"""

import worldoptbench


def test_version_is_set():
    assert worldoptbench.__version__


def test_subpackages_importable():
    import worldoptbench.metrics  # noqa: F401
    import worldoptbench.models  # noqa: F401
    import worldoptbench.optimizations  # noqa: F401
