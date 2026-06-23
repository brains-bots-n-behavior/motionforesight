"""Future 3D scene-flow prediction built on the pretrained TrackCraft3r model.

This package augments the frozen TrackCraft3r (Wan2.1 DiT + LoRA + dual VAEs)
single-pass dense 3D tracker so that, given only the first ``obs_frames`` of a
clip, it predicts dense 3D point tracks for the *future* frames as well.

See :class:`model.FutureSceneFlowModel` and ``README.md`` for the design.
"""

import models_pretrained  # noqa: F401  (activates vendored-import isolation)

from .model import FutureSceneFlowConfig, FutureSceneFlowModel  # noqa: E402,F401
