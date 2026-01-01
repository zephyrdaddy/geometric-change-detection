import numpy as np
from typing import List, Tuple

def sample_line(start: np.ndarray, end: np.ndarray, n_points: int = 50, noise_std: float = 0.02) -> np.ndarray:
    """Sample points along line segment with Gaussian noise."""
    t = np.linspace(0, 1, n_points)
    points = (1 - t[:, None]) * start + t[:, None] * end
    noise = np.random.normal(0, noise_std, points.shape)
    return points + noise

def sample_rectangle(center: np.ndarray, size: Tuple[float, float], n_points: int = 100, noise_std: float = 0.02) -> np.ndarray:
    """Sample points on rectangle boundary."""
    w, h = size
    corners = np.array([
        [center[0]-w/2, center[1]-h/2],
        [center[0]+w/2, center[1]-h/2],
        [center[0]+w/2, center[1]+h/2],
        [center[0]-w/2, center[1]+h/2]
    ])
    sides = [sample_line(corners[i], corners[(i+1)%4], n_points//4, noise_std) for i in range(4)]
    return np.vstack(sides)

def sample_circle(center: np.ndarray, radius: float, n_points: int = 100, noise_std: float = 0.02) -> np.ndarray:
    """Sample points on circle."""
    theta = np.linspace(0, 2*np.pi, n_points)
    points = center[None, :] + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    noise = np.random.normal(0, noise_std, points.shape)
    return points + noise
