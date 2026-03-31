from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
_EPS = 1e-12

_cached_model: nn.Module | None = None
_cached_tfm: Any = None
_cached_device: torch.device | None = None


def _is_cuda_oom(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg and "cuda" in msg


def _resolve_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_model(device: torch.device) -> tuple[nn.Module, Any]:
    """Load DINOv2 ViT-S/14 from timm. Cached module-level after first call."""
    import timm

    model = timm.create_model(_MODEL_NAME, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)

    data_cfg = timm.data.resolve_model_data_config(model)
    tfm = timm.data.create_transform(**data_cfg, is_training=False)
    return model, tfm


def get_model(device_str: str = "auto") -> tuple[nn.Module, Any, torch.device]:
    global _cached_model, _cached_tfm, _cached_device

    preferred_device = _resolve_device(device_str)

    if _cached_model is not None and _cached_tfm is not None and _cached_device is not None:
        if device_str == "auto" or _cached_device == preferred_device:
            return _cached_model, _cached_tfm, _cached_device

    try:
        model, tfm = _load_model(preferred_device)
        _cached_model = model
        _cached_tfm = tfm
        _cached_device = preferred_device
    except RuntimeError as err:
        if preferred_device.type == "cuda" and _is_cuda_oom(err):
            torch.cuda.empty_cache()
            cpu_device = torch.device("cpu")
            model, tfm = _load_model(cpu_device)
            _cached_model = model
            _cached_tfm = tfm
            _cached_device = cpu_device
        else:
            raise

    return _cached_model, _cached_tfm, _cached_device


class EmbeddingGallery:
    """Per-player gallery of DINOv2 embeddings for re-identification.

    Each player has up to MAX_GALLERY_SIZE embeddings stored as a (N, D) tensor.
    New embeddings are added by EMA update on the gallery mean.
    Matching uses cosine similarity against the gallery mean.
    """

    MAX_GALLERY_SIZE = 10
    EMA_ALPHA = 0.1
    EMBED_DIM = 384

    def __init__(self, player_ids: list[str], device: str = "auto"):
        # Model loading is intentionally lazy; defer until first embed request.
        self._player_ids = list(player_ids)
        self._device_pref = device
        self._gallery: dict[str, torch.Tensor] = {
            pid: torch.empty((0, self.EMBED_DIM), dtype=torch.float32) for pid in self._player_ids
        }
        self._means: dict[str, torch.Tensor | None] = {pid: None for pid in self._player_ids}

    def enroll(self, player_id: str, crop_bgr: np.ndarray) -> None:
        """Add a crop to a player's gallery. First crop sets the mean.
        Subsequent crops update via EMA."""
        if player_id not in self._gallery:
            raise KeyError(f"Unknown player_id: {player_id}")

        emb = self._embed(crop_bgr)

        prev = self._gallery[player_id]
        if prev.numel() == 0:
            self._gallery[player_id] = emb.unsqueeze(0)
            self._means[player_id] = emb
            return

        updated = torch.cat([prev, emb.unsqueeze(0)], dim=0)
        if updated.shape[0] > self.MAX_GALLERY_SIZE:
            updated = updated[-self.MAX_GALLERY_SIZE :]
        self._gallery[player_id] = updated

        prev_mean = self._means[player_id]
        if prev_mean is None:
            new_mean = emb
        else:
            alpha = self.EMA_ALPHA
            new_mean = (1.0 - alpha) * prev_mean + alpha * emb
            new_mean = new_mean / (new_mean.norm(p=2) + _EPS)
        self._means[player_id] = new_mean

    def identify(self, crop_bgr: np.ndarray, candidates: list[str] | None = None) -> tuple[str | None, float]:
        """Return (player_id, cosine_similarity) for the best matching player.
        Returns (None, 0.0) if no player has any gallery entries.
        candidates: if provided, restrict matching to these player_ids.
        """
        if candidates is not None and len(candidates) == 0:
            return None, 0.0

        candidate_ids = candidates if candidates is not None else self._player_ids
        eligible = [pid for pid in candidate_ids if self.has_enrollment(pid)]
        if not eligible:
            return None, 0.0

        emb = self._embed(crop_bgr)
        best_pid: str | None = None
        best_sim = float("-inf")

        for pid in eligible:
            mean = self._means[pid]
            if mean is None:
                continue
            sim = float(torch.dot(emb, mean).item())
            if sim > best_sim:
                best_sim = sim
                best_pid = pid

        if best_pid is None:
            return None, 0.0
        return best_pid, best_sim

    def has_enrollment(self, player_id: str) -> bool:
        """True if this player has at least one enrolled crop."""
        return player_id in self._gallery and self._gallery[player_id].shape[0] > 0

    def similarity(self, crop_bgr: np.ndarray, player_id: str) -> float:
        """Cosine similarity of crop embedding against player gallery mean.
        Returns 0.0 if player has no gallery."""
        if not self.has_enrollment(player_id):
            return 0.0

        mean = self._means[player_id]
        if mean is None:
            return 0.0

        emb = self._embed(crop_bgr)
        return float(torch.dot(emb, mean).item())

    def _embed(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Extract DINOv2 embedding for a single BGR crop. Returns (D,) tensor on CPU."""
        model, tfm, model_device = get_model(self._device_pref)

        if crop_bgr is None:
            raise ValueError("crop_bgr cannot be None")

        arr = np.asarray(crop_bgr)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected BGR image with shape (H, W, 3); got {arr.shape}")

        if arr.shape[0] == 0 or arr.shape[1] == 0:
            arr = np.zeros((1, 1, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inp = tfm(pil_img)
        if not isinstance(inp, torch.Tensor):
            inp = torch.as_tensor(inp)
        if inp.ndim == 3:
            inp = inp.unsqueeze(0)

        def _forward(m: nn.Module, x: torch.Tensor, dev: torch.device) -> torch.Tensor:
            with torch.no_grad():
                out = m(x.to(dev, non_blocking=dev.type == "cuda"))
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim == 3:
                out = out[:, 0, :]
            if out.ndim == 2:
                out = out[0]
            elif out.ndim != 1:
                out = out.reshape(-1)
            return out

        try:
            vec = _forward(model, inp, model_device)
        except RuntimeError as err:
            if model_device.type != "cuda" or not _is_cuda_oom(err):
                raise
            torch.cuda.empty_cache()
            cpu_model, cpu_tfm, cpu_device = get_model("cpu")
            inp = cpu_tfm(pil_img)
            if not isinstance(inp, torch.Tensor):
                inp = torch.as_tensor(inp)
            if inp.ndim == 3:
                inp = inp.unsqueeze(0)
            vec = _forward(cpu_model, inp, cpu_device)

        vec = vec.detach().to("cpu", dtype=torch.float32)
        vec = vec / (vec.norm(p=2) + _EPS)
        return vec


__all__ = ["EmbeddingGallery", "get_model"]
