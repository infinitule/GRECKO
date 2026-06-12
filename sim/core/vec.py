"""2-D vector helpers that keep the API dimension-agnostic for a future 3-D upgrade.

All vectors are plain numpy arrays of shape (2,) or (3,).  Functions accept
either dimensionality transparently so callers never hardcode 2-D assumptions.
"""
import numpy as np


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def normalise(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def angle_of(v: np.ndarray) -> float:
    """Angle (radians) of a 2-D vector from +x axis, CCW positive."""
    return float(np.arctan2(v[1], v[0]))


def rotate2d(v: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2-D vector by `angle` radians."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ v[:2]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
