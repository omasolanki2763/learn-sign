"""Smoke test for learnsign package import and basic metadata."""


def test_package_imports() -> None:
    """Verify that the learnsign package can be imported and exposes a version."""
    import learnsign

    assert hasattr(learnsign, "__version__")
    assert learnsign.__version__ == "0.1.0"
