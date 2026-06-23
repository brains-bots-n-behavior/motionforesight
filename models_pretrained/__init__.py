"""Self-contained snapshot of TrackCraft3r + the future-prediction architecture.

Importing this package activates :mod:`models_pretrained._bootstrap`, which makes
the *vendored* copies of ``diffsynth`` and ``evaluation`` (shipped in this
folder) shadow the editable-installed originals in ``external/TrackCraft3r``.
This keeps the original submodule untouched.
"""

from . import _bootstrap  # noqa: F401  (side effect: activate import isolation)

_bootstrap.activate()
