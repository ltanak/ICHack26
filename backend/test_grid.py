#!/usr/bin/env python
from Krishna.simulation import create_grid
import numpy as np

# Create grid with default p_tree=0.6
print("Creating grid with p_tree=0.6...")
grid, mask = create_grid(p_tree=0.6)

# Count trees
ca_cells = mask == 1
total_cells = ca_cells.sum()
tree_cells = (grid[ca_cells] == 1).sum()
tree_pct = tree_cells / total_cells * 100

print(f"\nTotal CA cells: {total_cells}")
print(f"Cells with trees: {tree_cells}")
print(f"Tree coverage: {tree_pct:.1f}%")
print(f"\nExpected range: ~21-51% based on prob_map")
print(f"Actual: {tree_pct:.1f}%")

# Try with higher p_tree
print("\n" + "="*50)
print("Creating grid with p_tree=1.0...")
grid2, mask2 = create_grid(p_tree=1.0)

tree_cells2 = (grid2[ca_cells] == 1).sum()
tree_pct2 = tree_cells2 / total_cells * 100

print(f"Tree coverage with p_tree=1.0: {tree_pct2:.1f}%")
print(f"Expected range: ~35-85% based on prob_map")
