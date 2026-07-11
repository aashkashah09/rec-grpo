"""Phase 0 smoke test: the package imports and exposes a well-formed version.

This is the only test in Phase 0. It exercises the scaffolding without asserting any
experimental result (there are none yet), keeping ``make test`` honestly green.
"""

import specialist_router


def test_package_version_is_dotted_digits() -> None:
    """__version__ is a non-empty dotted-numeric string (e.g. ``0.0.0``)."""
    version = specialist_router.__version__
    assert isinstance(version, str)
    assert version
    parts = version.split(".")
    assert all(part.isdigit() for part in parts)
