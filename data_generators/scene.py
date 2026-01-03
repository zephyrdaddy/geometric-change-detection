import numpy as np
from typing import Dict, List, Tuple, Optional
from .shapes import sample_line, sample_rectangle, sample_circle



class Scene:
    def __init__(self, bounds: Tuple[float, float] = (-10, 10)):
        self.bounds = bounds
        self.shapes: List[Dict] = []
    
    def add_shape(self, shape_type: str, params: Dict, n_points: int = 100):
        """Add shape to scene with parameters."""
        self.shapes.append({
            'shape_type': shape_type,
            'params': params,
            'n_points': n_points,
            'label': len(self.shapes)  # Unique ID for change tracking
        })
    
    def sample_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample all points from scene shapes."""
        all_points = []
        for shape in self.shapes:
            if shape['shape_type'] == 'line':
                pts = sample_line(
                    shape['params']['start'], 
                    shape['params']['end'], 
                    shape['n_points']
                )
            elif shape['shape_type'] == 'rect':
                pts = sample_rectangle(
                    shape['params']['center'],
                    shape['params']['size'],
                    shape['n_points']
                )
            elif shape['shape_type'] == 'circle':
                pts = sample_circle(
                    shape['params']['center'],
                    shape['params']['radius'],
                    shape['n_points']
                )
            all_points.append(pts)
        return np.vstack(all_points) if all_points else np.empty((0, 2))
        
    def apply_change(self, change_prob: float = 0.3) -> 'Scene':
        """Create modified scene by changing some shapes."""
        new_scene = Scene(self.bounds)
        for shape in self.shapes:
            if np.random.random() < change_prob:
                mutated = self._mutate_shape(shape)
                # mutated has the same structure as shape: type, params, n_points, label
                shape_type = mutated['shape_type']
                params = mutated['params']
                n_points = mutated.get('n_points', 100)
                new_scene.add_shape(shape_type, params, n_points)
            else:
                # unchanged
                shape_type = shape['shape_type']
                params = shape['params']
                n_points = shape.get('n_points', 100)
                new_scene.add_shape(shape_type, params, n_points)
        return new_scene

    # def apply_change(self, change_prob: float = 0.3) -> 'Scene':
    #     new_scene = Scene(self.bounds)
    #     for shape in self.shapes:
    #         if np.random.random() < change_prob:
    #             new_shape = self._mutate_shape(shape)
    #         else:
    #             new_shape = shape.copy()
    #         new_scene.add_shape(**new_shape)  # ← this is the problem
    #     return new_scene
    
    def _mutate_shape(self, shape: Dict) -> Dict:
        mutated = shape.copy()
        mutated['label'] = -1  # Mark as changed
        if shape['shape_type'] == 'line':
            # Move endpoint
            mutated['params']['end'] += np.random.normal(0, 0.5, 2)
        elif shape['shape_type'] == 'rect':
            mutated['params']['center'] += np.random.normal(0, 0.3, 2)
        elif shape['shape_type'] == 'circle':
            mutated['params']['radius'] *= (1 + np.random.normal(0, 0.1))
        return mutated
