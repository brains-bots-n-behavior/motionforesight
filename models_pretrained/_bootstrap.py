"""Import isolation for the vendored TrackCraft3r code.

``models_pretrained`` ships a *copy* of TrackCraft3r's ``diffsynth`` and
``evaluation`` packages plus the pretrained weights, so the future-prediction
architecture can be developed without ever touching the original
``external/TrackCraft3r`` submodule.

The original submodule is installed editable (``pip install -e .``), which
registers a ``_EditableFinder`` on ``sys.meta_path`` that redirects
``import diffsynth`` / ``import evaluation`` (and *their submodules*) back to
``external/TrackCraft3r``.  Importing this module first neutralizes that
redirect so the vendored copy under ``models_pretrained`` is the one that loads.

Usage (must run before any ``import diffsynth`` / ``import evaluation``)::

    import models_pretrained  # auto-runs activate()
    # or, from a standalone script:
    from models_pretrained import _bootstrap; _bootstrap.activate()
"""

from __future__ import annotations

import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent  # .../models_pretrained
_VENDORED = ("diffsynth", "evaluation")
_activated = False


def _is_vendored(module) -> bool:
    file = getattr(module, "__file__", None) or ""
    return str(VENDOR_DIR) in str(file)


def activate() -> None:
    """Make the vendored ``diffsynth`` / ``evaluation`` win all imports."""
    global _activated
    if _activated:
        return

    # 1. Drop the editable-install finder(s) that redirect diffsynth/evaluation
    #    to external/TrackCraft3r.  Identified by a MAPPING attribute that names
    #    one of our vendored packages, or by the setuptools class name.
    kept = []
    for finder in sys.meta_path:
        mapping = getattr(finder, "MAPPING", None)
        is_editable_redirect = (
            type(finder).__name__ == "_EditableFinder"
            or (isinstance(mapping, dict) and any(k in mapping for k in _VENDORED))
        )
        if not is_editable_redirect:
            kept.append(finder)
    sys.meta_path[:] = kept

    # 2. Evict any already-imported (non-vendored) copies so the next import
    #    resolves freshly against the vendored tree.
    for name in list(sys.modules):
        head = name.split(".", 1)[0]
        if head in _VENDORED and not _is_vendored(sys.modules[name]):
            del sys.modules[name]

    # 3. Put the vendored tree first on sys.path so the standard PathFinder
    #    resolves diffsynth/evaluation here.
    vendor = str(VENDOR_DIR)
    if vendor in sys.path:
        sys.path.remove(vendor)
    sys.path.insert(0, vendor)

    _activated = True


activate()
