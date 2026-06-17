"""Dataset for future 3D point-track prediction from TrackCraft3r outputs.

The dense TrackCraft3r NPZs are used as pseudo-ground-truth.  Each item samples
points from the SAM3 object mask on the anchor frame, gives the model only the
first `obs_frames` RGB frames and 3D point history, and targets the remaining
future 3D tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from collections import Counter
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrackSampleIndex:
    video_id: str
    dense_path: Path
    mask_paths: tuple[Path, ...]
    text: str = ""


TEXT_PAD = "<pad>"
TEXT_UNK = "<unk>"
TEXT_CLS = "<cls>"
TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+")


def text_tokens(text: str) -> list[str]:
    return TEXT_TOKEN_RE.findall(text.lower())


def build_text_vocab(
    items: Iterable[TrackSampleIndex],
    max_size: int = 4096,
    min_count: int = 1,
) -> dict[str, int]:
    """Build a compact word vocabulary from clip labels/prompts."""

    vocab = {TEXT_PAD: 0, TEXT_UNK: 1, TEXT_CLS: 2}
    counts: Counter[str] = Counter()
    for item in items:
        counts.update(text_tokens(item.text))
    for token, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count < min_count:
            continue
        if token in vocab:
            continue
        if len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(
    text: str,
    vocab: dict[str, int],
    max_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids = [vocab[TEXT_CLS]]
    ids.extend(vocab.get(token, vocab[TEXT_UNK]) for token in text_tokens(text))
    ids = ids[:max_tokens]
    mask = np.zeros(max_tokens, dtype=bool)
    arr = np.zeros(max_tokens, dtype=np.int64)
    arr[: len(ids)] = ids
    mask[: len(ids)] = True
    return arr, mask


def _load_manifest(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    items = json.loads(path.read_text())
    if isinstance(items, dict):
        items = items.get("items", items.get("clips", []))
    out: dict[str, dict] = {}
    for item in items:
        clip_path = item.get("clip_path")
        if clip_path:
            out[Path(clip_path).stem] = item
        video_uid = item.get("video_uid")
        if video_uid:
            out[f"{video_uid}_anchor"] = item
    return out


def build_track_index(
    root: Path,
    tracks_name: str = "anchor_tracks32_500",
    manifest: Path | None = None,
    require_masks: bool = True,
    limit_clips: int = 0,
) -> list[TrackSampleIndex]:
    """Build an index of dense TrackCraft3r outputs matched to SAM3 masks."""

    root = root.expanduser().resolve()
    manifest_path = manifest
    if manifest_path is not None and not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_items = _load_manifest(manifest_path)

    items: list[TrackSampleIndex] = []
    for dense_path in sorted((root / tracks_name).glob("*_dense.npz")):
        video_id = dense_path.name[: -len("_dense.npz")]
        meta = manifest_items.get(video_id, {})
        mask_paths: list[Path] = []
        for inst in meta.get("instances", []):
            rel = inst.get("mask_path")
            if not rel:
                continue
            path = root / rel
            if path.exists():
                mask_paths.append(path)

        if not mask_paths and not require_masks:
            mask_dir = root / "sam3_anchor_masks" / "clips" / video_id
            mask_paths = sorted(mask_dir.glob("mask_*.png"))
        if require_masks and not mask_paths:
            continue

        text = (
            meta.get("segment_text")
            or meta.get("label")
            or meta.get("template")
            or video_id
        )
        items.append(
            TrackSampleIndex(
                video_id=video_id,
                dense_path=dense_path,
                mask_paths=tuple(mask_paths),
                text=text,
            )
        )
        if limit_clips > 0 and len(items) >= limit_clips:
            break
    return items


def split_items(
    items: list[TrackSampleIndex],
    val_fraction: float,
    seed: int,
) -> tuple[list[TrackSampleIndex], list[TrackSampleIndex]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(items))
    rng.shuffle(order)
    val_count = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
    val_ids = set(order[:val_count].tolist())
    train = [item for idx, item in enumerate(items) if idx not in val_ids]
    val = [item for idx, item in enumerate(items) if idx in val_ids]
    return train, val


def _hash_text(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if dim <= 0:
        return vec
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) else -1.0
        vec[bucket] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _resize_frames(rgb: np.ndarray, image_size: tuple[int, int]) -> torch.Tensor:
    out_h, out_w = image_size
    frames = []
    for frame in rgb:
        resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        frames.append(resized.astype(np.float32) / 127.5 - 1.0)
    arr = np.stack(frames, axis=0)  # T,H,W,C
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr)


def _load_union_mask(mask_paths: Iterable[Path], target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    union = np.zeros((target_h, target_w), dtype=bool)
    for path in mask_paths:
        mask = np.array(Image.open(path).convert("L")) > 0
        if mask.shape != (target_h, target_w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        union |= mask
    return union


def _safe_point_sample(
    track_map: np.ndarray,
    mask: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(track_map).all(axis=(0, 3))
    candidates = np.argwhere(mask & valid)
    if len(candidates) == 0:
        candidates = np.argwhere(valid)
    if len(candidates) == 0:
        h, w = mask.shape
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        candidates = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
    replace = len(candidates) < num_points
    choice = rng.choice(len(candidates), size=num_points, replace=replace)
    pts_yx = candidates[choice]
    ys = pts_yx[:, 0]
    xs = pts_yx[:, 1]
    return ys.astype(np.int64), xs.astype(np.int64)


class DenseTrackDataset(Dataset):
    """Samples object-mask point trajectories from dense TrackCraft3r NPZs."""

    def __init__(
        self,
        items: list[TrackSampleIndex],
        obs_frames: int = 10,
        total_frames: int = 32,
        num_points: int = 256,
        image_size: tuple[int, int] = (128, 224),
        text_dim: int = 256,
        text_vocab: dict[str, int] | None = None,
        max_text_tokens: int = 16,
        samples_per_clip: int = 1,
        seed: int = 0,
    ) -> None:
        if obs_frames >= total_frames:
            raise ValueError("obs_frames must be smaller than total_frames")
        if not items:
            raise ValueError("DenseTrackDataset received an empty item list")
        self.items = items
        self.obs_frames = obs_frames
        self.total_frames = total_frames
        self.num_points = num_points
        self.image_size = image_size
        self.text_dim = text_dim
        self.text_vocab = text_vocab or build_text_vocab(items)
        self.max_text_tokens = max_text_tokens
        self.samples_per_clip = max(1, samples_per_clip)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items) * self.samples_per_clip

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx % len(self.items)]
        rng = np.random.default_rng(self.seed + idx * 9973)

        with np.load(item.dense_path, allow_pickle=True) as data:
            track_map = data["track_map"][: self.total_frames].astype(np.float32)
            rgb = data["rgb"][: self.obs_frames]

        if track_map.shape[0] < self.total_frames:
            raise RuntimeError(f"{item.dense_path} has only {track_map.shape[0]} frames")
        h, w = track_map.shape[1:3]
        mask = _load_union_mask(item.mask_paths, (h, w))
        ys, xs = _safe_point_sample(track_map, mask, self.num_points, rng)

        tracks = track_map[:, ys, xs, :]  # T,N,3
        tracks = np.transpose(tracks, (1, 0, 2)).astype(np.float32)  # N,T,3

        obs = tracks[:, : self.obs_frames]
        center = obs.reshape(-1, 3).mean(axis=0)
        dist = np.linalg.norm(obs.reshape(-1, 3) - center[None], axis=1)
        scale = float(np.percentile(dist, 90))
        if not np.isfinite(scale) or scale < 1e-3:
            scale = 1.0
        tracks_norm = (tracks - center[None, None, :]) / scale

        uv = np.stack(
            [
                (xs.astype(np.float32) / max(1, w - 1)) * 2.0 - 1.0,
                (ys.astype(np.float32) / max(1, h - 1)) * 2.0 - 1.0,
            ],
            axis=-1,
        )
        token_ids, token_mask = encode_text(item.text, self.text_vocab, self.max_text_tokens)

        return {
            "frames": _resize_frames(rgb, self.image_size),
            "observed_tracks": torch.from_numpy(tracks_norm[:, : self.obs_frames]),
            "future_tracks": torch.from_numpy(tracks_norm[:, self.obs_frames : self.total_frames]),
            "point_uv": torch.from_numpy(uv.astype(np.float32)),
            "text_bow": torch.from_numpy(_hash_text(item.text, self.text_dim)),
            "text_tokens": torch.from_numpy(token_ids),
            "text_mask": torch.from_numpy(token_mask),
            "track_center": torch.from_numpy(center.astype(np.float32)),
            "track_scale": torch.tensor(scale, dtype=torch.float32),
            "video_id": item.video_id,
            "text": item.text,
        }
