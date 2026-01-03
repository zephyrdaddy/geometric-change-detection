import numpy as np
from typing import List, Tuple

# def sample_line(start: np.ndarray, end: np.ndarray, n_points: int = 50, noise_std: float = 0.02) -> np.ndarray:
#     """Sample points along line segment with Gaussian noise."""
#     t = np.linspace(0, 1, n_points)
#     points = (1 - t[:, None]) * start + t[:, None] * end
#     noise = np.random.normal(0, noise_std, points.shape)
#     return points + noise

# def sample_rectangle(center: np.ndarray, size: Tuple[float, float], n_points: int = 100, noise_std: float = 0.02) -> np.ndarray:
#     """Sample points on rectangle boundary."""
#     w, h = size
#     corners = np.array([
#         [center[0]-w/2, center[1]-h/2],
#         [center[0]+w/2, center[1]-h/2],
#         [center[0]+w/2, center[1]+h/2],
#         [center[0]-w/2, center[1]+h/2]
#     ])
#     sides = [sample_line(corners[i], corners[(i+1)%4], n_points//4, noise_std) for i in range(4)]
#     return np.vstack(sides)

# def sample_circle(center: np.ndarray, radius: float, n_points: int = 100, noise_std: float = 0.02) -> np.ndarray:
#     """Sample points on circle."""
#     theta = np.linspace(0, 2*np.pi, n_points)
#     points = center[None, :] + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
#     noise = np.random.normal(0, noise_std, points.shape)
#     return points + noise



def sample_rectangle(center: np.ndarray, size: Tuple[float, float], n_points: int = 100, noise_std: float = 0.02, remaining_corners: List = None) -> np.ndarray:
    """Sample rectangle boundary, supports partial (missing sides)."""
    if remaining_corners is not None:
        # Partial rect: sample only specified corner connections
        pts = []
        for i in range(len(remaining_corners)):
            start = remaining_corners[i]
            end = remaining_corners[(i+1) % len(remaining_corners)]
            side_pts = n_points // len(remaining_corners)
            pts.append(sample_line(start, end, side_pts, noise_std))
        return np.vstack(pts)
    else:
        # Original full rect
        w, h = size
        corners = np.array([
            [center[0]-w/2, center[1]-h/2],
            [center[0]+w/2, center[1]-h/2],
            [center[0]+w/2, center[1]+h/2],
            [center[0]-w/2, center[1]+h/2]
        ])
        sides = [sample_line(corners[i], corners[(i+1)%4], n_points//4, noise_std) for i in range(4)]
        return np.vstack(sides)

def sample_circle(center: np.ndarray, radius: float, n_points: int = 100, noise_std: float = 0.02, 
                  arc_start: float = None, arc_end: float = None) -> np.ndarray:
    """Sample circle arc only if specified."""
    if arc_start is not None and arc_end is not None:
        theta = np.linspace(arc_start, arc_end, n_points)
    else:
        theta = np.linspace(0, 2*np.pi, n_points)
    points = center[None, :] + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    noise = np.random.normal(0, noise_std, points.shape)
    return points + noise

def sample_line(start: np.ndarray, end: np.ndarray, n_points: int = 50, noise_std: float = 0.02, 
                seg2_start: np.ndarray = None, seg2_end: np.ndarray = None) -> np.ndarray:
    """Sample line, supports split segments (gap)."""
    if seg2_start is not None and seg2_end is not None:
        # Two segments with gap
        pts1 = sample_line(start, end, n_points//2, noise_std)  # seg1_end is 'end' here
        pts2 = sample_line(seg2_start, seg2_end, n_points//2, noise_std)
        return np.vstack([pts1, pts2])
    else:
        # Original single line
        t = np.linspace(0, 1, n_points)
        points = (1 - t[:, None]) * start + t[:, None] * end
        noise = np.random.normal(0, noise_std, points.shape)
        return points + noise