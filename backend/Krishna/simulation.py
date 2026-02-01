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


def step(grid, ignition_prob, wind_dir=(0, 0), wind_strength=1.0, retardant_zones=None):
    """
    Advance fire simulation by one time step.
    
    Args:
        grid: Current grid state
        ignition_prob: Base probability of fire spreading to adjacent tree
        wind_dir: Wind direction tuple (di, dj)
        wind_strength: Wind strength multiplier
        retardant_zones: List of (y, x, radius) tuples for retardant application, or None
    
    Returns:
        Updated grid after one time step
    """
    new_grid = grid.copy()
    burning_cells = np.argwhere(grid == BURNING)
    
    # Create retardant mask if zones are provided
    retardant_mask = None
    if retardant_zones:
        N = grid.shape[0]
        retardant_mask = np.ones((N, N))  # 1.0 = normal, 0.2 = retardant
        
        for cy, cx, radius in retardant_zones:
            # Apply retardant in circular area
            y_grid, x_grid = np.ogrid[:N, :N]
            distances = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
            retardant_mask[distances <= radius] = 0.2  # 80% reduction in spread probability
    
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
                    spread_prob = ignition_prob * wind_mult
                    
                    # Apply retardant effect if in retardant zone
                    if retardant_mask is not None:
                        spread_prob *= retardant_mask[ni, nj]
                    
                    # Clamp to [0, 1]
                    spread_prob = min(1.0, spread_prob)
                    
                    # Probabilistic spread
                    if np.random.rand() < spread_prob:
                        new_grid[ni, nj] = BURNING
    
    return new_grid


def run_simulation(grid, ignition_prob, wind_dir=(0, 0), wind_strength=1.0, retardant_zones=None, cleared_zones=None, max_steps=10000):
    """
    Run fire simulation until no burning cells remain.
    
    Args:
        grid: Initial grid state
        ignition_prob: Base fire spread probability
        wind_dir: Wind direction
        wind_strength: Wind strength multiplier
        retardant_zones: List of (y, x, radius) tuples for retardant zones
        cleared_zones: List of (y, x, width, height) tuples for cleared areas
        max_steps: Maximum simulation steps (safety limit)
    
    Returns:
        Final grid state
    """
    current_grid = grid.copy()
    
    # Apply cleared zones (remove trees)
    if cleared_zones:
        for cy, cx, w, h in cleared_zones:
            half_w = w // 2
            half_h = h // 2
            if half_w == 0 or half_h == 0:
                continue  # Skip invalid cleared zones
            for dy in range(-half_h, half_h + 1):
                for dx in range(-half_w, half_w + 1):
                    y, x = cy + dy, cx + dx
                    if 0 <= y < current_grid.shape[0] and 0 <= x < current_grid.shape[1]:
                        if (dx / half_w) ** 2 + (dy / half_h) ** 2 <= 1:
                            if current_grid[y, x] == TREE:
                                current_grid[y, x] = EMPTY
    
    steps = 0
    
    while np.any(current_grid == BURNING) and steps < max_steps:
        current_grid = step(current_grid, ignition_prob, wind_dir, wind_strength, retardant_zones)
        steps += 1
    
    return current_grid


def monte_carlo_simulation(n_runs, p_tree, ignition_prob, wind_dir=(0, 0), wind_strength=1.0, custom_fire_points=None, retardant_zones=None, cleared_zones=None):
    """
    Run Monte Carlo simulation to estimate burn probability.
    
    Args:
        n_runs: Number of simulation runs
        p_tree: Vegetation density
        ignition_prob: Base fire spread probability
        wind_dir: Wind direction
        wind_strength: Wind strength multiplier
        custom_fire_points: List of (y, x) tuples for custom fire starts, or None for historic starts
        retardant_zones: List of (y, x, radius) tuples for retardant zones
        cleared_zones: List of (y, x, width, height) tuples for cleared areas
    
    Returns:
        burn_probability: NxN array with burn probability (0 to 1)
    """
    # Get grid shape from first run
    grid, _ = create_grid(p_tree=p_tree)
    burn_counts = np.zeros_like(grid, dtype=float)
    
    print(f"Running {n_runs} Monte Carlo simulations...")
    print(f"  Grid shape: {grid.shape}")
    print(f"  Initial burning cells: {np.count_nonzero(grid == BURNING)}")
    print(f"  Initial trees: {np.count_nonzero(grid == TREE)}")
    
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
        
        # Count initial burning cells
        initial_burning = np.count_nonzero(grid == BURNING)
        
        # Run simulation with mitigation zones
        final_grid = run_simulation(grid, ignition_prob, wind_dir, wind_strength, retardant_zones, cleared_zones)
        
        # Count burnt cells
        burnt_cells = np.count_nonzero(final_grid == BURNT)
        burn_counts += (final_grid == BURNT)
        
        if run == 0:
            print(f"  First run: {initial_burning} initial fires -> {burnt_cells} burnt cells")
    
    # Calculate probability
    burn_probability = burn_counts / n_runs
    print(f"  Done! Total cells with burn probability > 0: {np.count_nonzero(burn_probability)}")
    
    return burn_probability