"""
Interactive fire simulation visualization.

Features:
- Real-time fire spread animation
- Interactive controls for parameters
- Single run vs Monte Carlo mode
- Wind direction and strength controls
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.colors import ListedColormap

from Krishna.simulation import (
    create_grid, step, monte_carlo_simulation,
    EMPTY, TREE, BURNING, BURNT
)

# Animation parameters
ANIMATION_INTERVAL = 100  # milliseconds
MONTE_CARLO_RUNS = 20

# Wind directions
WIND_DIRS = {
    "None": (0, 0),
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


class FireSimulation:
    """Interactive fire simulation with matplotlib."""
    
    def __init__(self):
        # Initial parameters
        self.p_tree = 0.6
        self.ignition_prob = 0.7
        self.wind_strength = 1.0
        self.wind_dir = (0, 0)
        self.mode = "Historic Run"
        self.monte_carlo_enabled = False  # Track if Monte Carlo is enabled
        
        # Custom mode fire points
        self.custom_fire_points = []
        self.custom_mode_paused = False  # Track if custom mode is paused
        
        # Mitigation strategies
        self.mitigation_mode = "None"  # "None", "Retardant", or "Clear"
        self.retardant_zones = []  # List of (y, x, radius) tuples
        self.cleared_zones = []  # List of (y, x, width, height) tuples for ellipses
        self.retardant_radius = 10  # Radius in grid cells
        self.retardant_multiplier = 0.2  # Reduces ignition prob to 20%
        self.clear_width = 18  # Width of cleared ellipse
        self.clear_height = 9  # Height of cleared ellipse
        
        # Initialize grid
        self.grid, self.ca_mask = create_grid(p_tree=self.p_tree)
        
        # Setup plot
        self.setup_plot()
        self.setup_controls()
        
    def setup_plot(self):
        """Create the main plot and colormap."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.3, left=0.15)
        self.ax.axis("off")
        self.ax.set_title("California Wildfire Simulation", fontsize=14, pad=10)
        
        # Colormap: grey=empty, green=tree, red=burning, brown=burnt
        self.cmap = ListedColormap(["lightgrey", "green", "red", "peru"])
        
        # Mask cells outside California (make them white/transparent)
        masked_grid = np.ma.masked_where(self.ca_mask == 0, self.grid)
        
        # Initial image
        self.im = self.ax.imshow(
            masked_grid, 
            cmap=self.cmap, 
            vmin=0, vmax=3, origin="lower"
        )
        # Set background color for masked areas
        #self.im.set_bad(color='white')
        
    def setup_controls(self):
        """Create sliders and buttons."""
        # Slider axes
        ax_p = plt.axes([0.25, 0.20, 0.6, 0.03])
        ax_ig = plt.axes([0.25, 0.15, 0.6, 0.03])
        ax_wind = plt.axes([0.25, 0.10, 0.6, 0.03])
        
        # Radio button axes
        ax_mode = plt.axes([0.02, 0.75, 0.12, 0.10])
        ax_wind_dir = plt.axes([0.02, 0.40, 0.12, 0.25])
        ax_mitigation = plt.axes([0.02, 0.25, 0.12, 0.12])  # Mitigation selector
        
        # Checkbox for Monte Carlo (above mode selector)
        ax_mc = plt.axes([0.02, 0.88, 0.12, 0.06])
        
        # Reset and Start buttons
        ax_reset = plt.axes([0.85, 0.85, 0.1, 0.05])
        ax_start = plt.axes([0.85, 0.78, 0.1, 0.05])
        
        # Create sliders
        self.p_slider = Slider(
            ax_p, "Vegetation Density", 
            valmin=0.1, valmax=0.9, 
            valinit=self.p_tree, valstep=0.02
        )
        
        self.ignition_slider = Slider(
            ax_ig, "Dryness (Ignition Prob)", 
            valmin=0.1, valmax=0.9, 
            valinit=self.ignition_prob, valstep=0.02
        )
        
        self.wind_slider = Slider(
            ax_wind, "Wind Strength", 
            valmin=1.0, valmax=9.0, 
            valinit=self.wind_strength, valstep=0.1
        )
        
        # Create radio buttons for mode
        self.mode_radio = RadioButtons(
            ax_mode, 
            ("Historic Run", "Custom Run")
        )
        
        # Create Monte Carlo checkbox
        from matplotlib.widgets import CheckButtons
        self.mc_check = CheckButtons(ax_mc, ["Monte Carlo"], [False])
        
        # Create wind direction radio buttons
        self.wind_radio = RadioButtons(
            ax_wind_dir,
            ("None", "Up", "Down", "Left", "Right")
        )
        
        # Create mitigation strategy radio buttons
        self.mitigation_radio = RadioButtons(
            ax_mitigation,
            ("None", "Retardant", "Clear Trees")
        )
        
        # Create reset button
        self.reset_btn = Button(ax_reset, "Reset")
        
        # Create start button (for Custom Run mode)
        self.start_btn = Button(ax_start, "Start")
        
        # Connect callbacks
        self.p_slider.on_changed(self.on_param_change)
        self.ignition_slider.on_changed(self.on_param_change)
        self.wind_slider.on_changed(self.on_param_change)
        self.wind_radio.on_clicked(self.on_wind_change)
        self.mitigation_radio.on_clicked(self.on_mitigation_change)
        self.mode_radio.on_clicked(self.on_mode_change)
        self.mc_check.on_clicked(self.on_mc_change)
        self.reset_btn.on_clicked(self.reset)
        self.start_btn.on_clicked(self.start_simulation)
        
        # Connect click event for custom fire placement and mitigation
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_param_change(self, val):
        """Handle parameter slider changes."""
        self.p_tree = self.p_slider.val
        self.ignition_prob = self.ignition_slider.val
        self.wind_strength = self.wind_slider.val
        self.reset(None)
        
    def on_wind_change(self, label):
        """Handle wind direction change."""
        self.wind_dir = WIND_DIRS[label]
        self.reset(None)
    
    def on_mc_change(self, label):
        """Handle Monte Carlo checkbox change."""
        self.monte_carlo_enabled = not self.monte_carlo_enabled
        self.reset(None)
    
    def on_mitigation_change(self, label):
        """Handle mitigation strategy change."""
        self.mitigation_mode = label
        print(f"Mitigation mode: {label}")
        if label == "Retardant":
            print(f"  Click to apply fire retardant (radius: {self.retardant_radius} cells)")
        elif label == "Clear Trees":
            print(f"  Click to clear vegetation (ellipse: {self.clear_width}x{self.clear_height} cells)")
    
    def on_mode_change(self, label):
        """Handle mode change between Historic Run and Custom Run."""
        self.mode = label
        if label == "Custom Run":
            # Clear custom fire points and pause when switching to custom mode
            self.custom_fire_points = []
            self.custom_mode_paused = True
        else:
            self.custom_mode_paused = False
        self.reset(None)
    
    def start_simulation(self, event):
        """Start the simulation in Custom Run mode."""
        if self.mode == "Custom Run" and self.custom_mode_paused:
            self.custom_mode_paused = False
            num_fires = len(self.custom_fire_points)
            print(f"Starting simulation with {num_fires} custom fire point(s)")
            self.ax.set_title(f"California Wildfire Simulation - Custom Run (Running)", fontsize=14)
            self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        """Handle mouse clicks to place fires or apply mitigation strategies."""
        # Only process clicks within the axes
        if event.inaxes != self.ax:
            return
        
        # Get click coordinates
        x, y = int(event.xdata), int(event.ydata)
        N = self.grid.shape[0]
        
        # Check if click is within grid bounds and inside California
        if not (0 <= x < N and 0 <= y < N and self.ca_mask[y, x] == 1):
            return
        
        # Handle mitigation strategies
        if self.mitigation_mode == "Retardant":
            # Add retardant zone
            self.retardant_zones.append((y, x, self.retardant_radius))
            self.visualize_retardant_zone(y, x)
            print(f"Retardant applied at ({y}, {x})")
            
        elif self.mitigation_mode == "Clear Trees":
            # Add cleared zone
            self.cleared_zones.append((y, x, self.clear_width, self.clear_height))
            self.apply_clearing(y, x)
            print(f"Trees cleared at ({y}, {x})")
            
        # Handle custom fire placement (only in Custom Run mode with no mitigation)
        elif self.mode == "Custom Run" and self.custom_mode_paused and self.mitigation_mode == "None":
            # Add fire point
            self.custom_fire_points.append((y, x))
            
            # Update grid to show the new fire
            if self.grid[y, x] == TREE:
                self.grid[y, x] = BURNING
                
                # Update display
                masked_grid = np.ma.masked_where(self.ca_mask == 0, self.grid)
                self.im.set_data(masked_grid)
                self.fig.canvas.draw_idle()
                
                print(f"Fire placed at ({y}, {x})")
    
    def visualize_retardant_zone(self, cy, cx):
        """Draw a circle to show retardant zone."""
        from matplotlib.patches import Circle
        # Remove old retardant circles
        for artist in self.ax.patches[:]:
            if isinstance(artist, Circle):
                artist.remove()
        
        # Draw all retardant zones
        for y, x, r in self.retardant_zones:
            circle = Circle((x, y), r, fill=False, edgecolor='cyan', linewidth=2, linestyle='--', alpha=0.7)
            self.ax.add_patch(circle)
        
        self.fig.canvas.draw_idle()
    
    def apply_clearing(self, cy, cx):
        """Clear vegetation in an elliptical area."""
        from matplotlib.patches import Ellipse
        
        # Clear trees in the grid
        half_w = self.clear_width // 2
        half_h = self.clear_height // 2
        
        for dy in range(-half_h, half_h + 1):
            for dx in range(-half_w, half_w + 1):
                y, x = cy + dy, cx + dx
                # Check if point is inside ellipse
                if 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]:
                    if (dx / half_w) ** 2 + (dy / half_h) ** 2 <= 1:
                        if self.grid[y, x] == TREE:
                            if np.random.rand() < 0.55:  # 60% chance to clear tree
                                self.grid[y, x] = EMPTY
        
        # Remove old ellipses
        for artist in self.ax.patches[:]:
            if isinstance(artist, Ellipse):
                artist.remove()
        
        # Draw all cleared zones
        for y, x, w, h in self.cleared_zones:
            ellipse = Ellipse((x, y), w, h, fill=False, edgecolor='orange', linewidth=2, linestyle='--', alpha=0.7)
            self.ax.add_patch(ellipse)
        
        # Update display
        masked_grid = np.ma.masked_where(self.ca_mask == 0, self.grid)
        self.im.set_data(masked_grid)
        self.fig.canvas.draw_idle()
        
    def reset(self, event):
        """Reset the simulation."""
        # Check if Monte Carlo is enabled
        if self.monte_carlo_enabled:
            # Run Monte Carlo simulation (keep mitigation zones)
            print(f"\nStarting Monte Carlo analysis ({self.mode})...")
            
            # Use custom fire points if in Custom mode, otherwise None for historic
            custom_fires = self.custom_fire_points if self.mode == "Custom Run" else None
            
            burn_prob = monte_carlo_simulation(
                n_runs=MONTE_CARLO_RUNS,
                p_tree=self.p_slider.val,
                ignition_prob=self.ignition_prob,
                wind_dir=self.wind_dir,
                wind_strength=self.wind_strength,
                custom_fire_points=custom_fires,
                retardant_zones=self.retardant_zones,
                cleared_zones=self.cleared_zones
            )
            
            # Mask cells outside California
            masked_prob = np.ma.masked_where(self.ca_mask == 0, burn_prob)
            
            # Display probability heatmap
            self.im.set_cmap("hot")
            self.im.set_clim(0, 1)
            self.im.set_data(masked_prob)
            
            mode_text = "Historic" if self.mode == "Historic Run" else "Custom"
            self.ax.set_title(
                f"Monte Carlo Analysis - {mode_text} Starts ({MONTE_CARLO_RUNS} runs)",
                fontsize=14
            )
            
        else:
            # Clear mitigation zones for single run modes
            self.retardant_zones = []
            self.cleared_zones = []
            
            # Remove visualization patches
            for artist in list(self.ax.patches):
                artist.remove()
            
            if self.mode == "Historic Run":
                # Create new grid with historic fire starts
                self.grid, _ = create_grid(p_tree=self.p_tree)
                
                # Mask cells outside California
                masked_grid = np.ma.masked_where(self.ca_mask == 0, self.grid)
                
                self.im.set_cmap(self.cmap)
                self.im.set_clim(0, 3)
                self.im.set_data(masked_grid)
                self.ax.set_title("California Wildfire Simulation - Historic Run", fontsize=14)
                
            elif self.mode == "Custom Run":
                # Create grid with only vegetation, no initial fires
                self.grid, _ = create_grid(p_tree=self.p_tree)
                
                # Remove historic fire starts
                self.grid[self.grid == BURNING] = TREE
                
                # Clear custom fire points and pause
                self.custom_fire_points = []
                self.custom_mode_paused = True
                
                # Mask cells outside California
                masked_grid = np.ma.masked_where(self.ca_mask == 0, self.grid)
                
                self.im.set_cmap(self.cmap)
                self.im.set_clim(0, 3)
                self.im.set_data(masked_grid)
                self.ax.set_title("California Wildfire Simulation - Custom Run (Click to place fires, then Start)", fontsize=14)
            
        self.fig.canvas.draw_idle()
        
    def update(self, frame):
        """Animation update function."""
        # Don't animate in Monte Carlo mode
        if self.monte_carlo_enabled:
            return [self.im]
        
        # Animate in Historic Run and Custom Run modes
        if self.mode in ["Historic Run", "Custom Run"]:
            # Don't animate if Custom Run is paused
            if self.mode == "Custom Run" and self.custom_mode_paused:
                return [self.im]
            
            # Check if fire is still burning
            if np.any(self.grid == BURNING):
                # Advance one step with retardant zones
                self.grid = step(
                    self.grid,
                    self.ignition_prob,
                    self.wind_dir,
                    self.wind_strength,
                    self.retardant_zones
                )
                # Update display with masked grid
                masked_grid = np.ma.masked_where(self.ca_mask == 0, self.grid)
                self.im.set_data(masked_grid)
        
        return [self.im]
    
    def run(self):
        """Start the animation."""
        self.ani = FuncAnimation(
            self.fig, 
            self.update, 
            interval=ANIMATION_INTERVAL,
            blit=False
        )
        plt.show()


def main():
    """Run the interactive simulation."""
    print("\n" + "="*60)
    print("California Wildfire Simulation with Mitigation Strategies")
    print("="*60)
    print("\nControls:")
    print("  - Vegetation Density: Tree coverage")
    print("  - Dryness: Fire spread probability")
    print("  - Wind Strength: Wind effect on spread")
    print("  - Wind Direction: Direction of wind")
    print("\nModes:")
    print("  - Historic Run: Uses real fire starts from GPKG data")
    print("  - Custom Run: Click to place fires, then click Start")
    print("\nMitigation Strategies (click on map):")
    print("  - None: No mitigation, normal behavior")
    print("  - Retardant: Apply fire retardant (cyan circles)")
    print("    → Reduces ignition probability by 80% in area")
    print("  - Clear Trees: Remove vegetation (orange ellipses)")
    print("    → Creates firebreaks by clearing trees")
    print("\nOptions:")
    print("  - Monte Carlo: Run 20 simulations, show burn probability")
    print("    (Works with both Historic and Custom starts!)")
    print("\nButtons:")
    print("  - Start: Begin simulation (Custom Run mode)")
    print("  - Reset: Restart simulation (clears mitigation zones)")
    print("\nWorkflow:")
    print("  1. Choose mode (Historic or Custom Run)")
    print("  2. (Custom only) Click to place fire points")
    print("  3. Select mitigation strategy (Retardant or Clear Trees)")
    print("  4. Click on map to apply mitigation")
    print("  5. (Optional) Check Monte Carlo for probability analysis")
    print("  6. Click 'Start' to begin simulation")
    print("  7. Watch how mitigation affects fire spread!")
    print("\nClose the window to exit.")
    print("="*60 + "\n")
    
    sim = FireSimulation()
    sim.run()


if __name__ == "__main__":
    main()