import numpy as np
from typing import Dict, List, Tuple, Optional
from .shapes import sample_line, sample_rectangle, sample_circle



class Scene:
    def __init__(self, bounds: Tuple[float, float] = (-10, 10)):
        self.bounds = bounds
        self.shapes: List[Dict] = []
    
    def _mutate_shape(self, shape: Dict) -> Dict:
        """Apply realistic geometric change: translate, rotate, scale, or partial removal."""
        mutated = shape.copy()
        mutated['label'] = -1  # Mark as changed
        
        change_type = np.random.choice(['translate', 'rotate', 'partial_remove'])
        
        if change_type == 'translate':
            # Move shape by random offset
            offset = np.random.normal(0, 0.6, 2)
            if shape['shape_type'] == 'line':
                mutated['params']['start'] += offset
                mutated['params']['end'] += offset
            else:
                mutated['params']['center'] += offset
        
        elif change_type == 'rotate':
            # Rotate around center/origin
            r = np.random.uniform(-0.8, 0.8)  # radians (~45°)
            cos_r, sin_r = np.cos(r), np.sin(r)
            rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            
            if shape['shape_type'] == 'line':
                ctr = (np.array(shape['params']['start']) + np.array(shape['params']['end'])) / 2
                rel_start = np.array(shape['params']['start']) - ctr
                rel_end = np.array(shape['params']['end']) - ctr
                mutated['params']['start'] = ctr + rot @ rel_start
                mutated['params']['end'] = ctr + rot @ rel_end
            else:
                # rect/circle: rotate center around scene origin for translation effect
                ctr = np.array(shape['params']['center'])
                mutated['params']['center'] = rot @ ctr
        
        elif change_type == 'partial_remove':
            # Reduce points to simulate partial occlusion/removal
            remove_ratio = np.random.uniform(0.2, 0.6)
            mutated['n_points'] = int(shape['n_points'] * (1 - remove_ratio))
        
        return mutated

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
        """Create modified scene: mutate existing + add/remove shapes."""
        new_scene = Scene(self.bounds)
        
        # Process existing shapes
        for shape in self.shapes:
            action = np.random.choice(['keep', 'mutate', 'remove'], 
                                    p=[1-change_prob*1.5, change_prob, change_prob*0.5])
            
            if action == 'mutate':
                mutated = self._mutate_shape(shape)
                new_scene.add_shape(mutated['shape_type'], mutated['params'], mutated['n_points'])
            elif action == 'keep':
                new_scene.add_shape(shape['shape_type'], shape['params'], shape['n_points'])
        return new_scene