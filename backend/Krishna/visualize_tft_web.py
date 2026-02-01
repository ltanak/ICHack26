"""
TFT Fire Spread Prediction Visualization for Web Integration.

Provides a web-compatible simulation class for TFT-based fire prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

from Krishna.tft_predictor import predict_fire_params
from Krishna.simulation import (
    step, create_grid, monte_carlo_simulation,
    EMPTY, TREE, BURNING, BURNT
)


class TFTFireSimulation:
    """TFT-based fire simulation viewer for web integration."""
    
    def __init__(self, temperature=50, humidity=5, wind_speed=20, wind_direction=270):
        """
        Initialize TFT fire simulation.
        
        Args:
            temperature: Temperature for TFT prediction
            humidity: Humidity for TFT prediction
            wind_speed: Wind speed for TFT prediction
            wind_direction: Wind direction for TFT prediction
        """
        # Get TFT prediction
        params = predict_fire_params(
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            wind_direction=wind_direction
        )

        # Fire parameters
        self.ignition_prob = 1
        self.wind_strength = 11
        self.wind_dir = params['wind_dir']
        self.p_tree = 0.9
        
        # Store weather parameters for reference
        self.temperature = temperature
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        
        # Load initial grid
        self.initial_grid, self.ca_mask = create_grid(p_tree=self.p_tree)
        self.grid = self.initial_grid.copy()
        
        # Expand initial fires
        self._expand_initial_fires()
        
        # Store initial state for reset
        self.initial_grid_with_fires = self.grid.copy()
        
        # Simulation state
        self.frame_count = 0
        self.monte_carlo_mode = False
        self.is_running = True
        
        # Setup plot
        self.setup_plot()
        
    def _expand_initial_fires(self):
        """Expand initial fire points."""
        fire_points = np.where(self.grid == BURNING)
        if len(fire_points[0]) > 0:
            for i in range(len(fire_points[0])):
                y, x = fire_points[0][i], fire_points[1][i]
                for dy in range(-6, 7):
                    for dx in range(-6, 7):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.grid.shape[0] and 
                            0 <= nx < self.grid.shape[1] and 
                            self.ca_mask[ny, nx] == 1):
                            if self.grid[ny, nx] == TREE:
                                self.grid[ny, nx] = BURNING
    
    def setup_plot(self):
        """Create the main plot and colormap."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15, right=0.85)
        self.ax.axis('off')
        self.ax.set_title(
            'Temporal Fusion Transformer Model Fire Spread Prediction',
            fontsize=14,
            pad=10
        )
        
        # Colormap: grey=empty, green=tree, red=burning, brown=burnt
        self.cmap = ListedColormap(['lightgrey', 'green', 'red', 'peru'])
        
        # Mask cells outside California
        masked_grid = np.ma.masked_where(self.ca_mask == 0, self.grid)
        
        # Initial image
        self.im = self.ax.imshow(
            masked_grid,
            cmap=self.cmap,
            vmin=0,
            vmax=3,
            origin='lower'
        )
        
        # Info text box
        self.info_text = self.ax.text(
            0.02, 0.98, '',
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    def reset(self):
        """Reset simulation to initial state."""
        self.grid = self.initial_grid_with_fires.copy()
        self.frame_count = 0
        self.monte_carlo_mode = False
        self.is_running = True
        self._update_display()
    
    def step(self):
        """Advance simulation by one frame."""
        if not self.is_running or self.monte_carlo_mode:
            return
        
        if np.any(self.grid == BURNING):
            self.grid = step(
                self.grid,
                self.ignition_prob,
                self.wind_dir,
                self.wind_strength
            )
            self.frame_count += 1
        else:
            self.is_running = False
        
        self._update_display()
    
    def run_monte_carlo(self, n_runs=20):
        """Run Monte Carlo simulation analysis."""
        burn_prob = monte_carlo_simulation(
            n_runs=n_runs,
            p_tree=self.p_tree,
            ignition_prob=self.ignition_prob,
            wind_dir=self.wind_dir,
            wind_strength=self.wind_strength
        )
        
        masked_prob = np.ma.masked_where(self.ca_mask == 0, burn_prob)
        self.im.set_cmap('hot')
        self.im.set_clim(0, 1)
        self.im.set_data(masked_prob)
        
        mean_burn = np.mean(burn_prob[self.ca_mask == 1])
        max_burn = np.max(burn_prob)
        
        self.ax.set_title(
            f'TFT Monte Carlo Analysis - Burn Probability ({n_runs} runs)',
            fontsize=14
        )
        self.info_text.set_text(
            f'Mean burn prob: {mean_burn:.2f}\nMax burn prob: {max_burn:.2f}'
        )
        self.monte_carlo_mode = True
        self.fig.canvas.draw_idle()
    
    def _update_display(self):
        """Update the display with current grid state."""
        masked_grid = np.ma.masked_where(self.ca_mask == 0, self.grid)
        self.im.set_data(masked_grid)
        self.im.set_cmap(self.cmap)
        self.im.set_clim(0, 3)
        
        burned = np.sum(self.grid == BURNT)
        burning = np.sum(self.grid == BURNING)
        
        if self.is_running and burning > 0:
            self.info_text.set_text(
                f'Step: {self.frame_count}\nBurning: {burning}\nBurned: {burned}'
            )
        elif not self.is_running:
            self.info_text.set_text(f'Complete\nTotal Burned: {burned}')
        
        self.fig.canvas.draw_idle()
    
    def get_stats(self):
        """Get current simulation statistics."""
        return {
            'frame_count': self.frame_count,
            'burning': int(np.sum(self.grid == BURNING)),
            'burnt': int(np.sum(self.grid == BURNT)),
            'is_running': self.is_running,
            'monte_carlo_mode': self.monte_carlo_mode,
            'wind_direction': self.wind_dir,
            'wind_strength': self.wind_strength,
            'ignition_prob': self.ignition_prob,
            'p_tree': self.p_tree,
            'weather': {
                'temperature': self.temperature,
                'humidity': self.humidity,
                'wind_speed': self.wind_speed,
                'wind_direction': self.wind_direction
            }
        }
