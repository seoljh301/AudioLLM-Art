"""Chunked audio streaming utilities with a thread-safe ring buffer.

Used by realtime modules (MVP-A, MVP-C) to bridge variable-rate model
inference with fixed-rate audio callbacks.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioStreamConfig:
    sample_rate: int = 48000
    chunk_size: int = 2048
    channels: int = 1
    ring_capacity_chunks: int = 64


class RingBuffer:
    """Thread-safe, fixed-capacity float32 ring buffer for audio chunks."""

    def __init__(self, capacity_samples: int, channels: int = 1) -> None:
        self._buf = np.zeros((capacity_samples, channels), dtype=np.float32)
        self._capacity = capacity_samples
        self._channels = channels
        self._write_idx = 0
        self._read_idx = 0
        self._size = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    def write(self, data: np.ndarray) -> int:
        """Write a chunk. Drops oldest samples if full. Returns samples written."""
        if data.ndim == 1:
            data = data[:, None]
        n = data.shape[0]
        with self._lock:
            for i in range(n):
                self._buf[self._write_idx] = data[i]
                self._write_idx = (self._write_idx + 1) % self._capacity
                if self._size < self._capacity:
                    self._size += 1
                else:
                    self._read_idx = (self._read_idx + 1) % self._capacity
            self._not_empty.notify_all()
        return n

    def read(self, n: int, timeout: float | None = None) -> np.ndarray:
        """Read up to n samples. Pads with zeros if not enough data."""
        out = np.zeros((n, self._channels), dtype=np.float32)
        with self._not_empty:
            if self._size < n and timeout is not None:
                self._not_empty.wait(timeout=timeout)
            available = min(n, self._size)
            for i in range(available):
                out[i] = self._buf[self._read_idx]
                self._read_idx = (self._read_idx + 1) % self._capacity
            self._size -= available
        return out


def chunk_iter(audio: np.ndarray, chunk_size: int) -> "list[np.ndarray]":
    """Yield contiguous chunks of `chunk_size` from a 1D or 2D audio array."""
    if audio.ndim == 1:
        audio = audio[:, None]
    n = audio.shape[0]
    return [audio[i : i + chunk_size] for i in range(0, n, chunk_size)]
