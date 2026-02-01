"""
Historical Fire Simulation Visualization.

Features:
- Pre-computed fire spread simulation based on historical data
- Time slider to scrub through the simulation
- Uses same seed as Krishna simulation for similar results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap

from Krishna.simulation import (
    create_grid, step as krishna_step,
    EMPTY, TREE, BURNING, BURNT
)


def historical_step(grid, ignition_prob, wind_dir=(0, 0), wind_strength=1.0):
    """
    Advance fire simulation by one time step with slight variation.
    Uses Krishna's step but adds ~5% randomness for slightly different results.
    """
    # Use Krishna's step function as base
    new_grid = krishna_step(grid, ignition_prob, wind_dir, wind_strength)
    
    # Add slight random variation (~5% chance to flip some burning/not-burning cells)
    burning_mask = new_grid == BURNING
    burning_cells = np.argwhere(burning_mask)
    
    # Randomly prevent ~5% of new burns
    if len(burning_cells) > 0:
        n_to_flip = max(1, int(len(burning_cells) * 0.05))
        if np.random.rand() < 0.3:  # Only sometimes apply variation
            indices_to_flip = np.random.choice(len(burning_cells), min(n_to_flip, len(burning_cells)), replace=False)
            for idx in indices_to_flip:
                y, x = burning_cells[idx]
                # Only flip if it was a tree (new burn), not an existing burn
                if grid[y, x] == TREE:
                    new_grid[y, x] = TREE  # Prevent this burn
    
    return new_grid


class HistoricalFireSimulation:
    """Historical fire simulation viewer with time slider."""
    
    def __init__(self):
        # Fixed parameters for historical simulation (same as Krishna defaults)
        self.p_tree = 0.64
        self.ignition_prob = 7
        self.wind_strength = 10
        self.wind_dir = (1, 0)
        
        # Use fixed seed for reproducibility (same base as Krishna would use)
        np.random.seed(42)
        
        # Initialize grid
        self.initial_grid, self.ca_mask = create_grid(p_tree=self.p_tree, seed=42)
        
        # Pre-compute all simulation frames
        self.frames = self._precompute_simulation()
        self.current_frame = 0
        
        # Setup plot
        self.setup_plot()
        self.setup_controls()
        
    def _precompute_simulation(self):
        """Pre-compute all simulation frames."""
        print("Pre-computing simulation frames...")
        frames = [self.initial_grid.copy()]
        grid = self.initial_grid.copy()
        
        max_steps = 10000
        step_count = 0
        
        while np.any(grid == BURNING) and step_count < max_steps:
            grid = historical_step(
                grid,
                self.ignition_prob,
                self.wind_dir,
                self.wind_strength
            )
            frames.append(grid.copy())
            step_count += 1
            
            if step_count % 50 == 0:
                print(f"  Computed {step_count} frames...")
        
        print(f"  Done! Total frames: {len(frames)}")
        return frames
        
    def setup_plot(self):
        """Create the main plot and colormap."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
        self.ax.axis("off")
        self.ax.set_title("Historical Simulation", fontsize=16, pad=10)
        
        # Colormap: grey=empty, green=tree, red=burning, brown=burnt
        self.cmap = ListedColormap(["lightgrey", "green", "red", "peru"])
        
        # Mask cells outside California (make them white/transparent)
        masked_grid = np.ma.masked_where(self.ca_mask == 0, self.frames[0])
        
        # Initial image
        self.im = self.ax.imshow(
            masked_grid, 
            cmap=self.cmap, 
            vmin=0, vmax=3, origin="lower"
        )
    
    def setup_controls(self):
        """Create time slider."""
        # Time slider axis
        ax_time = plt.axes([0.15, 0.05, 0.7, 0.03])
        
        # Create time slider
        self.time_slider = Slider(
            ax_time, "Time", 
            valmin=0, valmax=len(self.frames) - 1, 
            valinit=0, valstep=1
        )
        
        # Connect callback
        self.time_slider.on_changed(self.on_time_change)
    
    def on_time_change(self, val):
        """Handle time slider changes."""
        frame_idx = int(val)
        self.current_frame = frame_idx
        
        # Update display
        masked_grid = np.ma.masked_where(self.ca_mask == 0, self.frames[frame_idx])
        self.im.set_data(masked_grid)
        self.fig.canvas.draw_idle()
    
    def run(self):
        """Show the visualization."""
        plt.show()


def main():
    """Run the historical simulation."""
    print("\n" + "="*60)
    print("Historical Simulation")
    print("="*60)
    print("\nUse the time slider to scrub through the simulation.")
    print("Close the window to exit.")
    print("="*60 + "\n")
    
    sim = HistoricalFireSimulation()
    sim.run()


if __name__ == "__main__":
    main()
