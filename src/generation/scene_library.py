# data_generators/scene_library.py
import numpy as np
from typing import Dict, List
# from .shapes import sample_line, sample_rectangle, sample_circle

def create_random_scene(n_shapes_range: tuple = (3, 6), bounds: tuple = (-8.0, 8.0)) -> List[Dict]:
    """
    Generate completely random scene with:
    - Random number of shapes (3-6)
    - Random shape types (line/rect/circle)
    - Random poses/sizes within bounds
    - No templates - pure randomness for maximum diversity
    """
    scene_bounds_min, scene_bounds_max = bounds
    shapes = []
    
    # Random number of shapes
    n_shapes = np.random.randint(*n_shapes_range)
    
    shape_types = ['line', 'rect', 'circle']
    for _ in range(n_shapes):
        shape_type = np.random.choice(shape_types)
        n_points = np.random.randint(60, 151)
        
        if shape_type == 'line':
            # Random line: start/end within bounds
            start = np.random.uniform(scene_bounds_min, scene_bounds_max, 2)
            length = np.random.uniform(1.5, 4.0)
            angle = np.random.uniform(0, np.pi * 2)
            end = start + length * np.array([np.cos(angle), np.sin(angle)])
            params = {'start': start, 'end': end}
            
        elif shape_type == 'rect':
            # Random rectangle
            center = np.random.uniform(scene_bounds_min, scene_bounds_max, 2)
            size = (np.random.uniform(0.8, 2.5), np.random.uniform(0.6, 2.0))
            params = {'center': center, 'size': size}
            
        elif shape_type == 'circle':
            # Random circle
            center = np.random.uniform(scene_bounds_min + 1.0, scene_bounds_max - 1.0, 2)
            radius = np.random.uniform(0.4, 1.8)
            params = {'center': center, 'radius': radius}
        
        shapes.append({
            'shape_type': shape_type,
            'params': params,
            'n_points': n_points
        })
    
    return shapes
