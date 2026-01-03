# data_generators/dataset.py

import os
import glob
from typing import List, Dict, Tuple, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from .scene import Scene


class ChangeDetectionDataset(Dataset):
    """
    Synthetic 2D point-cloud change detection dataset.

    Modes:
      - 'generate': generate all samples in memory on init.
      - 'load': load samples from HDF5 files in data_dir.
      - 'mixed': half generated in-memory, half loaded from files (if available).

    Each item returns:
      {
        'P':       [N_max, 2] float32   # padded point cloud P
        'Q':       [N_max, 2] float32   # padded point cloud Q
        'mask_p':  [N_max]    float32   # 1 for valid P points, 0 for padding
        'mask_q':  [N_max]    float32   # 1 for valid Q points, 0 for padding
        'change_p':[N_max]    float32   # per-point change label for P (0/1)
        'change_q':[N_max]    float32   # per-point change label for Q (0/1)
        'y_global':[1]        float32   # global change label (0/1)
      }
    """

    def __init__(
        self,
        size: int = 10000,
        n_points_per_cloud: int = 512,
        noise_std: float = 0.05,
        mode: Literal['generate', 'load', 'mixed'] = 'generate',
        data_dir: str = 'data/generated',
        change_prob: float = 0.3,
        change_radius: float = 0.25,
    ) -> None:
        super().__init__()
        self.size = size
        self.n_points = n_points_per_cloud
        self.noise_std = noise_std
        self.mode = mode
        self.data_dir = data_dir
        self.change_prob = change_prob
        self.change_radius = change_radius

        if mode == 'generate':
            self.samples = self._generate_all()
        elif mode == 'load':
            self.samples = self._load_from_files()
            if len(self.samples) == 0:
                raise RuntimeError(f"No HDF5 samples found in {data_dir}")
        elif mode == 'mixed':
            generated = self._generate_all(size // 2)
            loaded = self._load_from_files()
            # truncate/extend loaded to match total desired size
            if len(loaded) < size - len(generated):
                repeats = int(np.ceil((size - len(generated)) / len(loaded)))
                loaded = (loaded * repeats)[: (size - len(generated))]
            else:
                loaded = loaded[: (size - len(generated))]
            self.samples = generated + loaded
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def _make_base_scene(self) -> Scene:
        """Create a base scene with a few fixed shapes."""
        scene = Scene()
        # Horizontal line
        scene.add_shape('line', {'start': np.array([-5.0, 0.0]), 'end': np.array([5.0, 0.0])}, 150)
        # Rectangle
        scene.add_shape('rect', {'center': np.array([0.0, 3.0]), 'size': (2.0, 1.5)}, 120)
        # Circle
        scene.add_shape('circle', {'center': np.array([3.0, -2.0]), 'radius': 1.2}, 150)
        # Another line
        scene.add_shape('line', {'start': np.array([-2.0, -4.0]), 'end': np.array([2.0, -4.0])}, 100)
        return scene

    def _generate_one(self) -> Dict[str, torch.Tensor]:
        """Generate one synthetic (P, Q) pair and labels."""
        # Base scene P
        scene_p = self._make_base_scene()
        P_raw = scene_p.sample_points()

        # Scene Q with some changed shapes
        scene_q = scene_p.apply_change(change_prob=self.change_prob)
        Q_raw = scene_q.sample_points()

        # Resample and add noise
        P = self._resample_and_noise(P_raw, self.n_points, self.noise_std)
        Q = self._resample_and_noise(Q_raw, self.n_points, self.noise_std)

        # Per-point change masks (heuristic)
        change_mask_p = self._compute_change_mask(P, scene_p, scene_q)
        change_mask_q = self._compute_change_mask(Q, scene_p, scene_q)

        # Global label: did any change occur?
        y_global = float(np.any(change_mask_p) or np.any(change_mask_q))

        # Pad to fixed length with sentinel
        P_pad, mask_p = self._pad_points(P, self.n_points)
        Q_pad, mask_q = self._pad_points(Q, self.n_points)

        # Align change masks to padded length
        change_p_pad = self._pad_labels(change_mask_p, self.n_points)
        change_q_pad = self._pad_labels(change_mask_q, self.n_points)

        sample = {
            'P': torch.from_numpy(P_pad).float(),
            'Q': torch.from_numpy(Q_pad).float(),
            'mask_p': torch.from_numpy(mask_p).float(),
            'mask_q': torch.from_numpy(mask_q).float(),
            'change_p': torch.from_numpy(change_p_pad).float(),
            'change_q': torch.from_numpy(change_q_pad).float(),
            'y_global': torch.tensor([y_global], dtype=torch.float32),
        }
        return sample

    def _generate_all(self, override_size: Optional[int] = None) -> List[Dict[str, torch.Tensor]]:
        """Generate all samples in memory."""
        n = override_size if override_size is not None else self.size
        samples: List[Dict[str, torch.Tensor]] = []
        for _ in range(n):
            samples.append(self._generate_one())
        return samples

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _resample_and_noise(self, points: np.ndarray, target_n: int, noise_std: float) -> np.ndarray:
        """Resample to exactly target_n points and add Gaussian noise."""
        if points.shape[0] == 0:
            # edge case: no points
            pts = np.zeros((target_n, 2), dtype=np.float32)
        elif points.shape[0] > target_n:
            idx = np.random.choice(points.shape[0], target_n, replace=False)
            pts = points[idx]
        else:
            # upsample by random duplication
            extra = target_n - points.shape[0]
            idx = np.random.choice(points.shape[0], extra, replace=True)
            pts = np.concatenate([points, points[idx]], axis=0)
        noise = np.random.normal(0.0, noise_std, size=pts.shape)
        return (pts + noise).astype(np.float32)

    def _compute_change_mask(self, pts: np.ndarray, scene_p: Scene, scene_q: Scene) -> np.ndarray:
        """
        Heuristic per-point change mask:
        - Sample dense points from shapes that changed vs unchanged,
          and label points based on distance to changed vs unchanged sets.
        """

        # Identify changed vs unchanged shapes using label field
        changed_shapes: List[np.ndarray] = []
        unchanged_shapes: List[np.ndarray] = []

        for shp_p, shp_q in zip(scene_p.shapes, scene_q.shapes):
            # In scene.apply_change we can mark changed shapes by label -1
            if shp_q.get('label', shp_p['label']) == -1:
                # changed
                changed_shapes.append(self._sample_shape_points(shp_q))
            else:
                unchanged_shapes.append(self._sample_shape_points(shp_q))

        if len(changed_shapes) > 0:
            changed_pts = np.concatenate(changed_shapes, axis=0)
        else:
            changed_pts = np.empty((0, 2), dtype=np.float32)

        if len(unchanged_shapes) > 0:
            unchanged_pts = np.concatenate(unchanged_shapes, axis=0)
        else:
            unchanged_pts = np.empty((0, 2), dtype=np.float32)

        # Default: no change
        mask = np.zeros(pts.shape[0], dtype=np.float32)
        if changed_pts.shape[0] == 0 and unchanged_pts.shape[0] == 0:
            return mask

        # For each point, compute distance to nearest changed and unchanged regions
        if changed_pts.shape[0] > 0:
            d_changed = self._nearest_distance(pts, changed_pts)
        else:
            d_changed = np.full(pts.shape[0], np.inf, dtype=np.float32)

        if unchanged_pts.shape[0] > 0:
            d_unchanged = self._nearest_distance(pts, unchanged_pts)
        else:
            d_unchanged = np.full(pts.shape[0], np.inf, dtype=np.float32)

        # Label as changed if closer to changed shapes and within radius
        mask[(d_changed < d_unchanged) & (d_changed < self.change_radius)] = 1.0
        return mask

    def _nearest_distance(self, pts: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Compute nearest-neighbor distance from pts to ref (brute force)."""
        # pts: [N, 2], ref: [M, 2]
        # Expand and compute all pairwise distances
        diff = pts[:, None, :] - ref[None, :, :]  # [N, M, 2]
        dist2 = np.sum(diff * diff, axis=-1)      # [N, M]
        d_min = np.sqrt(np.min(dist2, axis=1))    # [N]
        return d_min.astype(np.float32)

    def _sample_shape_points(self, shape_def: Dict) -> np.ndarray:
        """Sample dense points from a shape definition (for change mask)."""
        from .shapes import sample_line, sample_rectangle, sample_circle

        n = max(80, shape_def.get('n_points', 80))
        if shape_def['shape_type'] == 'line':
            return sample_line(
                np.asarray(shape_def['params']['start'], dtype=np.float32),
                np.asarray(shape_def['params']['end'], dtype=np.float32),
                n_points=n,
                noise_std=0.0,
            )
        elif shape_def['shape_type'] == 'rect':
            return sample_rectangle(
                np.asarray(shape_def['params']['center'], dtype=np.float32),
                shape_def['params']['size'],
                n_points=n,
                noise_std=0.0,
            )
        elif shape_def['shape_type'] == 'circle':
            return sample_circle(
                np.asarray(shape_def['params']['center'], dtype=np.float32),
                float(shape_def['params']['radius']),
                n_points=n,
                noise_std=0.0,
            )
        else:
            return np.zeros((0, 2), dtype=np.float32)

    def _pad_points(self, pts: np.ndarray, max_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pad points to max_n with sentinel value and create mask."""
        N = pts.shape[0]
        out = np.full((max_n, 2), -100.0, dtype=np.float32)
        mask = np.zeros((max_n,), dtype=np.float32)
        n_use = min(N, max_n)
        out[:n_use] = pts[:n_use]
        mask[:n_use] = 1.0
        return out, mask

    def _pad_labels(self, labels: np.ndarray, max_n: int) -> np.ndarray:
        """Pad label vector to max_n with zeros."""
        N = labels.shape[0]
        out = np.zeros((max_n,), dtype=np.float32)
        n_use = min(N, max_n)
        out[:n_use] = labels[:n_use]
        return out

    # ------------------------------------------------------------------
    # HDF5 I/O (for offline generated data)
    # ------------------------------------------------------------------
    def _load_from_files(self) -> List[Dict[str, torch.Tensor]]:
        """Load samples from HDF5 files in self.data_dir."""
        samples: List[Dict[str, torch.Tensor]] = []
        os.makedirs(self.data_dir, exist_ok=True)
        h5_files = sorted(glob.glob(os.path.join(self.data_dir, "*.h5")))
        if len(h5_files) == 0:
            return samples

        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                for key in f.keys():
                    if not key.startswith("sample_"):
                        continue
                    grp = f[key]
                    P = np.array(grp['P'], dtype=np.float32)
                    Q = np.array(grp['Q'], dtype=np.float32)
                    mask_p = np.array(grp['mask_p'], dtype=np.float32)
                    mask_q = np.array(grp['mask_q'], dtype=np.float32)
                    change_p = np.array(grp['change_p'], dtype=np.float32)
                    change_q = np.array(grp['change_q'], dtype=np.float32)
                    y_global = np.array(grp.get('y_global', [0.0]), dtype=np.float32)

                    samples.append({
                        'P': torch.from_numpy(P),
                        'Q': torch.from_numpy(Q),
                        'mask_p': torch.from_numpy(mask_p),
                        'mask_q': torch.from_numpy(mask_q),
                        'change_p': torch.from_numpy(change_p),
                        'change_q': torch.from_numpy(change_q),
                        'y_global': torch.from_numpy(y_global),
                    })
        # respect requested size
        if len(samples) > self.size:
            samples = samples[: self.size]
        return samples

    # ------------------------------------------------------------------
    # Standard Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
