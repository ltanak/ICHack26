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
        
        self.update_display(0)
    
    def _load_california(self):
        """Load California as vertices."""
        try:
            print("Loading California...")
            states = gpd.read_file(
                "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
            )
            ca = states[states["name"] == "California"]
            verts = []
            for geom in ca.geometry:
                verts.extend(geom_to_verts(geom))
            
            # Calculate bounds
            all_coords = np.vstack(verts)
            minx, miny = all_coords.min(axis=0)
            maxx, maxy = all_coords.max(axis=0)
            w, h = maxx - minx, maxy - miny
            self.bounds = (minx - w*0.05, miny - h*0.05, maxx + w*0.05, maxy + h*0.05)
            return verts
        except Exception as e:
            print(f"Warning: {e}")
            self.bounds = (-124.5, 32.5, -114.0, 42.0)
            return []
    
    def _add_vegetation_background(self):
        """Add realistic vegetation density as background."""
        try:
            # Import the realistic vegetation functions
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from Krishna.simulation import load_masks, load_dense_areas, create_vegetation_probability_map
            
            # Extract year from folder name (e.g., "2017_Snapshot" -> 2017)
            year_str = self.folder_name.split('_')[0]
            year = int(year_str) if year_str.isdigit() else 2020
            
            # Load vegetation data
            ca_mask, _, bounds = load_masks()
            dense_coords, radius_deg = load_dense_areas(year=year, threshold=10.0)
            
            # Create probability map
            prob_map = create_vegetation_probability_map(
                ca_mask, bounds, dense_coords, radius_deg,
                base_prob=0.8, dense_prob=1.0, min_prob=0.8
            )
            
            # Display as background image
            minx, miny, maxx, maxy = bounds
            self.ax.imshow(
                prob_map,
                extent=(minx, maxx, miny, maxy),
                origin='lower',
                cmap='Greens',
                alpha=0.6,
                vmin=0,
                vmax=1,
                zorder=0
            )
            print(f"Added vegetation background for year {year}")
        except Exception as e:
            print(f"Could not load vegetation background: {e}")
    
    def _build_cache(self):
        """Build cumulative vertex cache for instant playback."""
        files = get_sorted_files(self.folder_path)
        self.cumulative_verts = []  # Pre-computed cumulative for each frame
        self.frame_dates = []
        
        accumulated = []
        for i, f in enumerate(files):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(files)}...")
            
            dt, p = parse_filename(f.name)
            date_str = dt.strftime('%B %d, %Y') + f" {p}" if dt else f.stem
            self.frame_dates.append(date_str)
            
            try:
                gdf = gpd.read_file(f, layer="perimeter")
                gdf = gdf[gdf.geometry.notna()]
                for geom in gdf.geometry:
                    accumulated.extend(geom_to_verts(geom))
            except:
                pass
            
            # Store a copy of accumulated state at this frame
            self.cumulative_verts.append(list(accumulated))
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump({'cumulative': self.cumulative_verts, 'dates': self.frame_dates}, f)
        print("Cache saved!")
    
    def _load_cache(self):
        with open(self.cache_file, 'rb') as f:
            data = pickle.load(f)
        self.cumulative_verts = data['cumulative']
        self.frame_dates = data['dates']
    
    def setup_plot(self):
        """Create the main plot and colormap."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
        self.ax.axis("off")
        self.ax.set_facecolor('white')
        
        self.ax.set_xlim(self.bounds[0], self.bounds[2])
        self.ax.set_ylim(self.bounds[1], self.bounds[3])
        
        # Add realistic vegetation background
        self._add_vegetation_background()
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
        # California (static)
        if self.ca_verts:
            ca_coll = PolyCollection(self.ca_verts, facecolor='none',
                                     edgecolor='darkgreen', linewidth=2, alpha=0.9)
            self.ax.add_collection(ca_coll)
        
        # Fire collection (updated each frame)
        self.fire_coll = PolyCollection([], facecolor='red',
                                        edgecolor='darkred', alpha=0.8)
        self.ax.add_collection(self.fire_coll)
        
        year = self.folder_name.split('_')[0]
        self.title = self.ax.set_title(f"California Wildfires - {year}", fontsize=14)
        
        # Controls
        self.slider = Slider(plt.axes([0.2, 0.06, 0.6, 0.03]), 'Frame',
                            0, len(self.cumulative_verts)-1, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider)
        
        self.play_btn = Button(plt.axes([0.2, 0.01, 0.12, 0.04]), 'Play')
        self.play_btn.on_clicked(self.toggle_play)
        
        Button(plt.axes([0.35, 0.01, 0.12, 0.04]), 'Reset').on_clicked(self.reset)
        Button(plt.axes([0.5, 0.01, 0.12, 0.04]), 'Faster').on_clicked(lambda e: self.speed(-20))
        Button(plt.axes([0.65, 0.01, 0.12, 0.04]), 'Slower').on_clicked(lambda e: self.speed(20))
        
        self.is_playing = False
        self.animation = None
        self.interval = ANIMATION_INTERVAL
    
    def update_display(self, idx):
        idx = int(idx) % len(self.cumulative_verts)
        self.current_frame = idx
        
        # Just use pre-computed cumulative data - instant!
        self.fire_coll.set_verts(self.cumulative_verts[idx])
        
        year = self.folder_name.split('_')[0]
        self.title.set_text(f"California Wildfires - {year}\n{self.frame_dates[idx]}")
        
        self.slider.eventson = False
        self.slider.set_val(idx)
        self.slider.eventson = True
        
        return [self.fire_coll, self.title]
    
    def on_slider(self, val):
        if not self.is_playing:
            self.update_display(int(val))
            self.fig.canvas.draw_idle()
    
    def toggle_play(self, event):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.label.set_text('Play')
            if self.animation:
                self.animation.event_source.stop()
        else:
            self.is_playing = True
            self.play_btn.label.set_text('Pause')
            self.start_anim()
    
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
