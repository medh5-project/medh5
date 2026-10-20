"""One place that says how to install an optional dependency.

The converters and the MONAI bridge each keep the ``require_<package>``
callers already import (``require_pydicom``, ``require_highdicom``,
``require_monai``), and each is a call to :func:`require`; ``Codec.kwargs``
calls it for ``hdf5plugin``.  ``require_nibabel`` is the other exception:
no converter needs nibabel since 2.2.0, and no extra installs it.  The torch
bridge is the exception: ``medh5.torch`` tries ``import torch`` once, when it
is imported, and its ``require_torch`` raises its own ``ImportError`` from
what that found.
"""

from __future__ import annotations

import importlib
from typing import Any


def require(module: str, *, extra: str, purpose: str) -> Any:
    """Import *module*, or raise an ``ImportError`` naming the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"{module} is required for {purpose}. Install it with: "
            f"pip install 'medh5[{extra}]'"
        ) from exc


__all__ = ["require"]
