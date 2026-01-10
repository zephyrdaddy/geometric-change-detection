# shapes.py
import numpy as np
from typing import Tuple, Optional

def sample_line_ordered(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)
    pts = (1 - t[:, None]) * start + t[:, None] * end
    return pts.astype(np.float32)

def sample_rectangle_ordered(center: np.ndarray, size: Tuple[float, float], n: int) -> np.ndarray:
    w, h = size
    corners = np.array([
        [ w/2,  h/2],
        [ w/2, -h/2],
        [-w/2, -h/2],
        [-w/2,  h/2],
    ]) + center

    per_side = n // 4
    pts = []
    for i in range(4):
        pts.append(sample_line_ordered(corners[i], corners[(i+1)%4], per_side))
    return np.vstack(pts)

def sample_circle_ordered(center: np.ndarray, radius: float, n: int) -> np.ndarray:
    theta = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, n)
    pts = center + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return pts.astype(np.float32)
