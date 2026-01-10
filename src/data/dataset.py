# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from src.generation.scene import Scene

class ChangeDetectionDataset(Dataset):
    def __init__(self, size=10000, n_points=512, noise_std=0.03):
        self.size = size
        self.n = n_points
        self.noise = noise_std
        self.samples = [self._generate_one() for _ in range(size)]

    def _make_scene(self) -> Scene:
        s = Scene()
        # Random number of shapes, now more shapes per scene
        n_shapes = np.random.randint(3, 8)  # 3 to 7 shapes
        for _ in range(n_shapes):
            t = np.random.choice(["line", "rect", "circle"])
            if t == "line":
                s.add_shape(
                    "line",
                    {
                        "start": np.random.uniform(-10, 10, 2),  # wider range
                        "end": np.random.uniform(-10, 10, 2)
                    },
                    n_points=np.random.randint(50, 120)  # vary points per line
                )
            elif t == "rect":
                s.add_shape(
                    "rect",
                    {
                        "center": np.random.uniform(-10, 10, 2),
                        "size": (np.random.uniform(1, 5), np.random.uniform(1, 5))  # bigger rects
                    },
                    n_points=np.random.randint(80, 150)
                )
            else:  # circle
                s.add_shape(
                    "circle",
                    {
                        "center": np.random.uniform(-10, 10, 2),
                        "radius": np.random.uniform(0.5, 3)  # bigger circles
                    },
                    n_points=np.random.randint(60, 130)
                )
        return s


    def _generate_one(self):
        scene_p = self._make_scene()
        scene_q = scene_p.apply_change()

        P, sid_p = scene_p.sample()
        Q, sid_q = scene_q.sample()

        change_p = np.array([scene_p.shapes[i]["changed"] for i in sid_p], np.float32)
        change_q = np.array([scene_q.shapes[i]["changed"] for i in sid_q], np.float32)

        # Add noise (no resampling)
        if self.noise > 0:
            P += np.random.normal(0, self.noise, P.shape)
            Q += np.random.normal(0, self.noise, Q.shape)


        # permutation
        idx_p = np.random.permutation(len(P))
        idx_q = np.random.permutation(len(Q))

        return {
            "P": torch.from_numpy(P[idx_p]),
            "Q": torch.from_numpy(Q[idx_q]),
            "change_p": torch.from_numpy(change_p[idx_p]),
            "change_q": torch.from_numpy(change_q[idx_q]),
        }


    def __len__(self): return self.size
    def __getitem__(self, i): return self.samples[i]
