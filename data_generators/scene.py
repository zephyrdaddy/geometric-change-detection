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
            remove_ratio = np.random.uniform(0.25, 0.65)
    
            if shape['shape_type'] == 'circle':
                # Remove ARC by restricting angle range in sample_circle
                center = np.array(shape['params']['center'])
                radius = shape['params']['radius']
                arc_start = np.random.uniform(0, 2*np.pi)
                arc_length = (1 - remove_ratio) * 2 * np.pi  # KEEP arc length
                mutated['params'].update({
                    'arc_start': arc_start,
                    'arc_end': arc_start + arc_length
                })
                mutated['n_points'] = int(shape['n_points'] * (1 - remove_ratio) * 1.2)  # density adjust
                
            elif shape['shape_type'] == 'rect':
                # Remove ENTIRE side(s) - modify corners directly
                w, h = shape['params']['size']
                center = np.array(shape['params']['center'])
                corners = np.array([
                    [-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]
                ]) + center
                
                # Randomly remove 1-2 continuous sides
                n_remove_sides = np.random.randint(1, 3)
                start_side = np.random.randint(0, 4)
                remove_sides = [(start_side + i) % 4 for i in range(n_remove_sides)]
                
                # Rebuild remaining sides only
                remaining_sides = [i for i in range(4) if i not in remove_sides]
                side_pts = int(shape['n_points'] / len(remaining_sides))
                mutated['params'] = {'remaining_corners': corners[remaining_sides].tolist()}
                mutated['n_points'] = side_pts * len(remaining_sides)
                
            elif shape['shape_type'] == 'line':
                # Remove MIDDLE SEGMENT - split into two shorter lines
                total_len = np.linalg.norm(np.array(shape['params']['end']) - np.array(shape['params']['start']))
                gap_start_rel = np.random.uniform(0.2, 0.5)
                gap_end_rel = gap_start_rel + remove_ratio * 0.6  # shorter gap
                
                start = np.array(shape['params']['start'])
                end = np.array(shape['params']['end'])
                direction = (end - start) / total_len
                
                seg1_end = start + gap_start_rel * total_len * direction
                seg2_start = start + gap_end_rel * total_len * direction
                
                # Store as TWO line segments (gap in middle)
                mutated['params'] = {
                    'seg1_start': start.tolist(),
                    'seg1_end': seg1_end.tolist(),
                    'seg2_start': seg2_start.tolist(), 
                    'seg2_end': end.tolist()
                }
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
        
    def sample_points(self) -> np.ndarray:
        all_points = []
        for shape in self.shapes:
            params = shape['params']
            
            if shape['shape_type'] == 'line':
                if 'seg1_start' in params:
                    pts = sample_line(
                        np.asarray(params['seg1_start']),
                        np.asarray(params['seg1_end']),
                        shape['n_points']//2,
                        seg2_start=np.asarray(params['seg2_start']),
                        seg2_end=np.asarray(params['seg2_end'])
                    )
                else:
                    pts = sample_line(
                        np.asarray(params['start']), 
                        np.asarray(params['end']), 
                        shape['n_points']
                    )
            elif shape['shape_type'] == 'rect':
                if 'remaining_corners' in params:
                    pts = sample_rectangle(
                        np.zeros(2), (0,0), shape['n_points'],
                        remaining_corners=[np.asarray(c) for c in params['remaining_corners']]
                    )
                else:
                    pts = sample_rectangle(
                        np.asarray(params['center']),
                        params['size'],
                        shape['n_points']
                    )
            elif shape['shape_type'] == 'circle':
                arc_start = params.get('arc_start')
                arc_end = params.get('arc_end')
                pts = sample_circle(
                    np.asarray(params['center']),
                    float(params['radius']),
                    shape['n_points'],
                    arc_start=arc_start, arc_end=arc_end
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