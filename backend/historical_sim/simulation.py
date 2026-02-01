"""
Fire simulation logic and grid management.

This module handles:
- Loading cached masks
- Creating simulation grids with vegetation
- Fire spread logic with wind effects
- Monte Carlo simulations
"""

import numpy as np
from pathlib import Path

# Cell states
EMPTY = 0
TREE = 1
BURNING = 2
BURNT = 3

# Cache directory
CACHE_DIR = Path("cache")


def load_masks():
    """
    Load precomputed California and fire masks from cache.
    
    Returns:
        ca_mask: NxN array (1=inside CA, 0=outside)
        fire_mask: NxN array (2=fire start points)
        bounds: (minx, miny, maxx, maxy)
    """
    ca_file = CACHE_DIR / "ca_mask.npz"
    fire_file = CACHE_DIR / "fire_mask.npz"
    
    if not ca_file.exists() or not fire_file.exists():
        raise FileNotFoundError(
            "Cache files not found! Run preprocess.py first to generate them."
        )
    
    # Load CA mask
    ca_data = np.load(ca_file)
    ca_mask = ca_data['ca_mask']
    bounds = (ca_data['minx'], ca_data['miny'], ca_data['maxx'], ca_data['maxy'])
    
    # Load fire mask
    fire_data = np.load(fire_file)
    fire_mask = fire_data['fire_mask']
    
    return ca_mask, fire_mask, bounds


def create_grid(p_tree=0.6, seed=None):
    """
    Create a simulation grid with vegetation and initial fires.
    
    Args:
        p_tree: Probability of a tree in each cell (vegetation density)
        seed: Random seed for reproducibility
    
    Returns:
        grid: NxN array with EMPTY, TREE, and BURNING cells
        ca_mask: NxN array showing California boundary
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Load cached masks
    ca_mask, fire_mask, bounds = load_masks()
    N = ca_mask.shape[0]
    
    # Initialize empty grid
    grid = np.zeros((N, N), dtype=int)
    
    # Add vegetation only inside California
    inside_ca = ca_mask == 1
    n_cells = inside_ca.sum()
    
    grid[inside_ca] = np.random.choice(
        [EMPTY, TREE],
        size=n_cells,
        p=[1 - p_tree, p_tree]
    )
    
    # Add initial fire points
    grid[fire_mask == BURNING] = BURNING
    
    return grid, ca_mask


def get_wind_multiplier(direction, wind_dir, wind_strength):
    """
    Calculate fire spread probability multiplier based on wind.
    
    Args:
        direction: Tuple (di, dj) representing spread direction
        wind_dir: Tuple (di, dj) representing wind direction
        wind_strength: Wind strength multiplier (1.0 = no wind effect)
    
    Returns:
        Probability multiplier for this direction
    """
    if wind_dir == (0, 0):  # No wind
        return 1.0
    
    if direction == wind_dir:  # Downwind
        return wind_strength * 2
    elif direction == (-wind_dir[0], -wind_dir[1]):  # Upwind
        return 1.0 / (wind_strength * 2)
    else:  # Perpendicular
        return 1.0


def step(grid, ignition_prob, wind_dir=(0, 0), wind_strength=1.0):
    """
    Advance fire simulation by one time step.
    
    Args:
        grid: Current grid state
        ignition_prob: Base probability of fire spreading to adjacent tree
        wind_dir: Wind direction tuple (di, dj)
        wind_strength: Wind strength multiplier
    
    Returns:
        Updated grid after one time step
    """
    new_grid = grid.copy()
    burning_cells = np.argwhere(grid == BURNING)
    
    for i, j in burning_cells:
        # Burning cell becomes burnt
        new_grid[i, j] = BURNT
        
        # Try to spread to neighbors (up, down, left, right)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            
            # Check bounds
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                # Only spread to trees
                if grid[ni, nj] == TREE:
                    # Calculate spread probability with wind effect
                    wind_mult = get_wind_multiplier((di, dj), wind_dir, wind_strength)
                    spread_prob = min(1.0, ignition_prob * wind_mult)
                    
                    # Probabilistic spread
                    if np.random.rand() < spread_prob:
                        new_grid[ni, nj] = BURNING
    
    return new_grid


def run_simulation(grid, ignition_prob, wind_dir=(0, 0), wind_strength=1.0, max_steps=10000):
    """
    Run fire simulation until no burning cells remain.
    
    Args:
        grid: Initial grid state
        ignition_prob: Base fire spread probability
        wind_dir: Wind direction
        wind_strength: Wind strength multiplier
        max_steps: Maximum simulation steps (safety limit)
    
    Returns:
        Final grid state
    """
    current_grid = grid.copy()
    steps = 0
    
    while np.any(current_grid == BURNING) and steps < max_steps:
        current_grid = step(current_grid, ignition_prob, wind_dir, wind_strength)
        steps += 1
    
    return current_grid


def monte_carlo_simulation(n_runs, p_tree, ignition_prob, wind_dir=(0, 0), wind_strength=1.0, custom_fire_points=None):
    """
    Run Monte Carlo simulation to estimate burn probability.
    
    Args:
        n_runs: Number of simulation runs
        p_tree: Vegetation density
        ignition_prob: Base fire spread probability
        wind_dir: Wind direction
        wind_strength: Wind strength multiplier
        custom_fire_points: List of (y, x) tuples for custom fire starts, or None for historic starts
    
    Returns:
        burn_probability: NxN array with burn probability (0 to 1)
    """
    # Get grid shape from first run
    grid, _ = create_grid(p_tree=p_tree)
    burn_counts = np.zeros_like(grid, dtype=float)
    
    print(f"Running {n_runs} Monte Carlo simulations...")
    for run in range(n_runs):
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{n_runs}")
        
        # Create new random grid
        grid, _ = create_grid(p_tree=p_tree)
        
        # Use custom fire points if provided
        if custom_fire_points is not None:
            # Remove historic fire starts
            grid[grid == BURNING] = TREE
            # Add custom fire starts
            for y, x in custom_fire_points:
                if grid[y, x] == TREE:
                    grid[y, x] = BURNING
        
        # Run simulation
        final_grid = run_simulation(grid, ignition_prob, wind_dir, wind_strength)
        
        # Count burnt cells
        burn_counts += (final_grid == BURNT)
    
    # Calculate probability
    burn_probability = burn_counts / n_runs
    print("  Done!")
    
    return burn_probability