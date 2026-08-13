"""Timing / memory measurement and CSV I/O helpers.

Used by every benchmark script to record per-step time, peak GPU memory,
training/validation loss, and accuracy in a consistent CSV layout.

Typical usage::

    with open(csv_path, "w", newline="") as fh:
        write_csv_header(fh, ["variant", "epoch", "train_loss", "val_loss"])
    ...
    write_csv_row(fh, ["dense+rand", 1, 0.123, 0.456])
"""

from __future__ import annotations

import csv
import gc
import logging
import time
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Device selection
# ------------------------------------------------------------------ #

def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available :class:`torch.device`.

    Parameters
    ----------
    prefer_cuda : bool, default True
        If ``True`` and CUDA is available, return a CUDA device.
        Otherwise fall back to CPU.

    Returns
    -------
    torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_device_info(device: torch.device) -> None:
    """Log a one-line description of the active device."""
    if device.type == "cuda":
        logger.info(
            "Device: %s, memory: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        logger.info("Device: CPU")


# ------------------------------------------------------------------ #
# Timing / memory
# ------------------------------------------------------------------ #

def measure_timing(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    warmup_steps: int = 3,
    repeat_steps: int = 10,
    positive_weight: float = 10.0,
) -> tuple[float, float]:
    """Measure average step time and peak incremental GPU memory.

    Runs ``warmup_steps`` unmeasured warm-up iterations, then
    ``repeat_steps`` timed iterations and returns the mean step time
    and the peak GPU memory allocated during the timed iterations
    minus the memory allocated before the warm-up loop.

    Parameters
    ----------
    model : nn.Module
        The model to benchmark.  Must accept the batch as keyword
        arguments ``cs``, ``fs``, ``ct``, ``ft`` and optional ``ps``,
        ``pt``.
    batch : dict
        Input batch with the same keys the model accepts.
    device : torch.device
        Device on which the model is loaded.
    warmup_steps : int, default 3
        Number of unmeasured warm-up iterations.
    repeat_steps : int, default 10
        Number of timed iterations.
    positive_weight : float, default 10.0
        ``pos_weight`` argument for the binary-cross-entropy loss.

    Returns
    -------
    (time_per_step_seconds, peak_memory_megabytes)
        ``(-1, -1)`` if a CUDA out-of-memory error occurs (the error
        is swallowed so that the outer loop can mark the row ``oom``
        and move on).
    """
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    pos_weight = torch.tensor(positive_weight, device=device)
    moved_batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    try:
        for _ in range(warmup_steps):
            logits = model(
                moved_batch["cs"], moved_batch["fs"], moved_batch["ct"], moved_batch["ft"],
                moved_batch.get("ps"), moved_batch.get("pt"),
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, moved_batch["a"], pos_weight=pos_weight
            )
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
    except RuntimeError as exc:
        logger.info("Warmup failed: %s", exc)
        return -1.0, -1.0

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated() if device.type == "cuda" else 0

    start = time.perf_counter()
    try:
        for _ in range(repeat_steps):
            logits = model(
                moved_batch["cs"], moved_batch["fs"], moved_batch["ct"], moved_batch["ft"],
                moved_batch.get("ps"), moved_batch.get("pt"),
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, moved_batch["a"], pos_weight=pos_weight
            )
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
    except RuntimeError as exc:
        logger.info("Timed loop failed: %s", exc)
        return -1.0, -1.0

    elapsed = (time.perf_counter() - start) / repeat_steps
    peak_mb = (
        (torch.cuda.max_memory_allocated() - memory_before) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    return elapsed, peak_mb


# ------------------------------------------------------------------ #
# CSV I/O
# ------------------------------------------------------------------ #

def write_csv_header(file_handle, columns: Sequence[str]) -> None:
    """Write a CSV header row.  Thin wrapper over :mod:`csv.writer`."""
    csv.writer(file_handle).writerow(list(columns))


def write_csv_row(file_handle, values: Iterable) -> None:
    """Write a single CSV row."""
    csv.writer(file_handle).writerow(list(values))


def open_csv_writer(path, columns: Sequence[str]):
    """Open ``path`` for writing and write the CSV header.

    Parameters
    ----------
    path : str | Path
        Destination path.
    columns : Sequence[str]
        Column names for the header row.

    Returns
    -------
    tuple[file, csv.writer]
        The opened file handle and a writer bound to it.  The caller is
        responsible for closing the file when done.
    """
    file_handle = open(path, "w", newline="")
    write_csv_header(file_handle, columns)
    return file_handle, csv.writer(file_handle)
