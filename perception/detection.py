"""Pure-Python geometry, velocity, and configuration helpers for YOLO tracking."""

from __future__ import annotations

import math
from pathlib import Path

DEFAULT_WEIGHTS_RELATIVE_PATH = Path("input_layer") / "yolov8n.pt"


def resolve_device(cuda_available: bool) -> str:
    """Return 'cuda' when a GPU is available, otherwise fall back cleanly to 'cpu'."""
    return "cuda" if cuda_available else "cpu"


def resolve_weights_path(weights_path: str | Path | None = None) -> Path:
    """Resolve YOLO model weights path relative to the repository root by default."""
    if weights_path is None:
        return Path(__file__).resolve().parent.parent / DEFAULT_WEIGHTS_RELATIVE_PATH
    return Path(weights_path)


def compute_centroid(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int]:
    """Compute the integer pixel center `(cx, cy)` for an `(x1, y1, x2, y2)` bounding box."""
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def compute_velocity(
    cx: int,
    cy: int,
    current_time: float,
    previous_entry: tuple[int, int, float] | None,
) -> tuple[float, float, float]:
    """Compute `(vx, vy, speed)` in pixels per second from consecutive centroids."""
    if previous_entry is None:
        return 0.0, 0.0, 0.0

    px, py, pt = previous_entry
    dt = max(current_time - pt, 1e-5)
    vx = (cx - px) / dt
    vy = (cy - py) / dt
    return vx, vy, math.hypot(vx, vy)


def smooth_height(
    current_height: float,
    previous_smooth: float | None,
    alpha: float = 0.7,
) -> float:
    """Apply exponential moving-average smoothing to bounding-box height."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")
    if previous_smooth is None:
        return float(current_height)
    return alpha * previous_smooth + (1.0 - alpha) * current_height


def compute_approach_velocity(
    smoothed_height: float,
    current_time: float,
    previous_entry: tuple[float, float] | None,
) -> float:
    """Estimate height expansion rate (px/s) as a relative approach-velocity proxy."""
    if previous_entry is None:
        return 0.0
    prev_height, prev_time = previous_entry
    dt = max(current_time - prev_time, 1e-5)
    return (smoothed_height - prev_height) / dt
