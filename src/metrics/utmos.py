"""UTMOSv2 naturalness MOS prediction (Baba et al. 2024, SLT VoiceMOS Challenge winner)."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
REMOVE_SILENT_SECTION = False  # off: synthetic wavs have no leading/trailing silence
DISABLE_MIXUP_INNER = True  # training-time augmentation; halves CPU spectrogram cost at inference


def _resolve_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_model():
    import utmosv2

    m = utmosv2.create_model(pretrained=True, device=_resolve_device())
    if DISABLE_MIXUP_INNER:
        try:
            m._cfg.dataset.spec_frames.mixup_inner = False
        except AttributeError:
            pass
    return m


def _predict_kwargs() -> dict:
    return {
        "device": _resolve_device(),
        "batch_size": DEFAULT_BATCH_SIZE,
        "num_workers": DEFAULT_NUM_WORKERS,
        "remove_silent_section": REMOVE_SILENT_SECTION,
        "verbose": False,
    }


def score_wav(path: str) -> float:
    return float(_load_model().predict(input_path=path, **_predict_kwargs()))


def score_dir(dir_path: str) -> dict[str, float]:
    """Batch-score all wavs in dir_path (no recursion). Returns {basename: mos}."""
    results = _load_model().predict(input_dir=dir_path, **_predict_kwargs())
    return {Path(r["file_path"]).name: float(r["predicted_mos"]) for r in results}


def score_wavs(paths: list[str]) -> dict[str, float]:
    """Score a list of arbitrary wav paths via a single batched UTMOSv2 call.

    Builds a flat tempdir of symlinks so all `paths` are scored in one dataloader pass,
    avoiding per-dir dataloader instantiation overhead.
    """
    if not paths:
        return {}
    paths = [str(p) for p in paths]
    with tempfile.TemporaryDirectory(prefix="utmos_") as td:
        link_to_orig: dict[str, str] = {}
        for i, p in enumerate(paths):
            link_name = f"{i:06d}_{Path(p).name}"
            link_path = os.path.join(td, link_name)
            os.symlink(os.path.abspath(p), link_path)
            link_to_orig[link_name] = p
        results = _load_model().predict(input_dir=td, **_predict_kwargs())
    return {
        link_to_orig[Path(r["file_path"]).name]: float(r["predicted_mos"])
        for r in results
        if Path(r["file_path"]).name in link_to_orig
    }
