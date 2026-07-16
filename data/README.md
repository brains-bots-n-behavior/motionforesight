# Example Inputs

This folder contains model-ready example inputs for MotionForesight inference.

- `processed7_uniform/`: checked-in `*_user.npz` files. These are the tensors passed to the model.

Raw videos and generated HTML visualizations are intentionally not tracked.
The examples use the full-video-context path: 7 frames are selected uniformly
from the full video and passed as observed context.
