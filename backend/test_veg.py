#!/usr/bin/env python
from Krishna.simulation import create_grid, load_dense_areas, create_vegetation_probability_map, load_masks
import numpy as np

# Load masks and dense areas
ca_mask, fire_mask, bounds = load_masks()
dense_coords, radius_deg = load_dense_areas(year=2020, threshold=10.0)

print('Dense coords shape:', dense_coords.shape if dense_coords is not None else None)
print('Radius deg:', radius_deg)
print('Bounds:', bounds)

# Create prob map
prob_map = create_vegetation_probability_map(
    ca_mask, bounds, dense_coords, radius_deg,
    base_prob=0.4 * 0.6,
    dense_prob=0.85 * 0.6,
    min_prob=0.35 * 0.6
)

print('\nProb map stats (only CA):')
ca_probs = prob_map[ca_mask == 1]
print('  Min:', ca_probs.min())
print('  Max:', ca_probs.max())
print('  Mean:', ca_probs.mean())
print('  Std:', ca_probs.std())
print('  Unique values count:', len(np.unique(ca_probs)))

# Sample distribution of prob values
print('\nProbability distribution:')
for threshold in [0.21, 0.3, 0.4, 0.51]:
    pct = (ca_probs >= threshold).sum() / len(ca_probs) * 100
    print(f'  >= {threshold:.2f}: {pct:.1f}%')
