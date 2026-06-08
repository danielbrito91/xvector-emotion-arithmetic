"""Dataset root resolution via environment variables.

Override the defaults without editing code:
  - ``DATA_ROOT``    parent dir holding the resampled datasets
                     (default: ``~/data/processed``).
  - ``ESD_ROOT``     full path to the resampled ESD root
                     (default: ``$DATA_ROOT/esd_24k``).
  - ``EMOUERJ_ROOT`` full path to the resampled emoUERJ root
                     (default: ``$DATA_ROOT/emouerj_24k``).
"""

from __future__ import annotations

import os


def data_root() -> str:
    return os.environ.get("DATA_ROOT", os.path.expanduser("~/data/processed"))


def esd_root() -> str:
    return os.environ.get("ESD_ROOT", os.path.join(data_root(), "esd_24k"))


def emouerj_root() -> str:
    return os.environ.get("EMOUERJ_ROOT", os.path.join(data_root(), "emouerj_24k"))
