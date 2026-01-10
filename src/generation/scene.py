# scene.py
import numpy as np
import copy
from typing import Dict, List, Tuple
from .shapes import (
    sample_line_ordered,
    sample_rectangle_ordered,
    sample_circle_ordered,
)

class Scene:
    def __init__(self):
        self.shapes: List[Dict] = []

    def add_shape(self, shape_type: str, params: Dict, n_points: int):
        self.shapes.append({
            "type": shape_type,
            "params": params,
            "n": n_points,
            "changed": False,
            "id": len(self.shapes),
        })

    def mutate_shape(self, shape: Dict) -> Dict:
        # Deep copy the entire shape so nothing is shared
        new = copy.deepcopy(shape)
        new["changed"] = True

        def sample_offset(min_mag=0.5, scale=0.5):
            """Sample a 2D offset with minimum magnitude."""
            while True:
                offset = np.random.normal(0, scale, 2)
                if np.linalg.norm(offset) >= min_mag:
                    return offset

        if new["type"] == "line":
            offset = sample_offset(min_mag=0.5, scale=1.0)
            new["params"]["start"] += offset
            new["params"]["end"] += offset

        elif new["type"] in ("rect", "circle"):
            offset = sample_offset(min_mag=0.5, scale=1.0)
            new["params"]["center"] += offset

        return new

    def apply_change(self) -> "Scene":
        new_scene = Scene()

        for shp in self.shapes:
            action = np.random.choice(["keep", "mutate", "remove"], p=[0.7, 0.2, 0.10])

            if action == "keep":
                new_scene.shapes.append(dict(shp))

            elif action == "mutate":
                # Mark the original as changed as well for bi-directional training.
                shp["changed"] = True  
                new_scene.shapes.append(self.mutate_shape(shp))

            elif action == "remove":
                # mark original shape as changed
                shp["changed"] = True

        return new_scene

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        pts, shape_ids = [], []

        for i, shp in enumerate(self.shapes):
            if shp["type"] == "line":
                p = sample_line_ordered(
                    np.array(shp["params"]["start"]),
                    np.array(shp["params"]["end"]),
                    shp["n"],
                )
            elif shp["type"] == "rect":
                p = sample_rectangle_ordered(
                    np.array(shp["params"]["center"]),
                    shp["params"]["size"],
                    shp["n"],
                )
            else:
                p = sample_circle_ordered(
                    np.array(shp["params"]["center"]),
                    shp["params"]["radius"],
                    shp["n"],
                )

            pts.append(p)
            shape_ids.append(np.full(len(p), i))

        if len(pts) == 0:
            return np.zeros((0,2), np.float32), np.zeros((0,), np.int32)  # safe fallback

        return np.vstack(pts), np.concatenate(shape_ids)
