"""Model registry: lazy load/unload of audio foundation models.

Each MVP registers a `loader_fn` keyed by a short name. The first `get(name)`
call triggers the load and caches the resulting object. `unload(name)` evicts
it and (best-effort) frees CUDA memory.

This module is import-light: torch is only imported inside `_cuda_mem`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

LoaderFn = Callable[[], Any]


@dataclass
class _Slot:
    loader: LoaderFn
    device: str
    obj: Any = None


@dataclass
class ModelRegistry:
    """Thread-safe lazy model registry. One instance is shared across modules."""

    _slots: dict[str, _Slot] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def register(self, name: str, loader: LoaderFn, device: str = "cuda") -> None:
        with self._lock:
            if name in self._slots:
                raise ValueError(f"model already registered: {name}")
            self._slots[name] = _Slot(loader=loader, device=device)
            logger.info("registry: registered slot %s on %s", name, device)

    def get(self, name: str) -> Any:
        with self._lock:
            slot = self._slots.get(name)
            if slot is None:
                raise KeyError(f"unknown model: {name}")
            if slot.obj is None:
                logger.info("registry: loading %s on %s", name, slot.device)
                slot.obj = slot.loader()
                self._log_mem(name)
            return slot.obj

    def unload(self, name: str) -> None:
        with self._lock:
            slot = self._slots.get(name)
            if slot is None or slot.obj is None:
                return
            slot.obj = None
            self._free_cuda()
            logger.info("registry: unloaded %s", name)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._slots.keys())

    @staticmethod
    def _log_mem(name: str) -> None:
        try:
            import torch  # noqa: WPS433
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info("registry: post-load CUDA mem %.2f GiB (%s)", allocated, name)
        except ImportError:
            pass

    @staticmethod
    def _free_cuda() -> None:
        try:
            import torch  # noqa: WPS433
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


_default = ModelRegistry()


def default_registry() -> ModelRegistry:
    """Return the process-wide default registry."""
    return _default
