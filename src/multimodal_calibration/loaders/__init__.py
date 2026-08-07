"""Per-dataset loader modules for the multi-modal calibration study.

Folder name is ``loaders/`` (NOT ``datasets/``) to avoid shadowing the
HuggingFace ``datasets`` library that some loaders import internally.

Each loader exposes a single function ``load(split: str) -> dict`` returning::

    {
      "y": np.ndarray,
      "tab": pd.DataFrame | None,
      "text": dict[str, list[str]] | None,
      "image": dict[str, list[Path]] | None,
      "id": np.ndarray,
    }

A dataset is "registered" by exposing it from ``REGISTRY`` below. New datasets
can be added by appending an entry: ``"name": loader_module.load``.
"""

from __future__ import annotations

from typing import Callable, Dict

from . import imdb_wiki, mercari, pawpularity


REGISTRY: Dict[str, Callable[[str], dict]] = {
    "imdb_wiki": imdb_wiki.load,
    "pawpularity": pawpularity.load,
    "mercari": mercari.load,
}
