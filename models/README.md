# Models

This folder will hold model definitions, training configs, and evaluation notes for future 3D scene flow prediction.

Near-term model plan:

1. Use short Action100M segments as action-centric video inputs.
2. Estimate depth and camera trajectories for each clip.
3. Run dense 3D point tracking to produce scene-flow-style supervision.
4. Train or probe video models to predict future 3D motion from observed frames.

Candidate starting points:

- TrackCraft3r-style dense 3D trajectory outputs as supervision.
- Video diffusion transformer features as motion priors.
- Future-window targets such as per-point displacement, visibility, and occlusion-aware flow.

This directory is intentionally lightweight until the data curation and tracking outputs stabilize.
