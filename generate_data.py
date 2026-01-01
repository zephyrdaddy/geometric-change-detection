#!/usr/bin/env python3
"""
Generate dataset to files for fast loading.
"""
import argparse
from data_generators.hdf5_utils import generate_dataset_to_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--output_dir', default='data/generated')
    parser.add_argument('--n_points', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1000)
    args = parser.parse_args()
    
    generate_dataset_to_files(args.n_samples, args.output_dir, args.n_points, args.batch_size)
