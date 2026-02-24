from __future__ import annotations

from importlib import metadata


class DistributionNotFound(Exception):
    """Compatibility shim for packages expecting pkg_resources.DistributionNotFound."""


def get_distribution(name: str):
    """Compatibility shim returning an object with a `version` attribute."""

    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise DistributionNotFound(str(exc)) from exc

    class _Distribution:
        def __init__(self, value: str) -> None:
            self.version = value

    return _Distribution(version)
