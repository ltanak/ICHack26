"""
Historical Fire Visualization - FAST version using PolyCollection.
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.collections import PolyCollection
from pathlib import Path
from datetime import datetime
import re
import pickle

ANIMATION_INTERVAL = 1  # Fast!


def parse_filename(filename):
    match = re.match(r'(\d{4})(\d{2})(\d{2})(AM|PM)\.gpkg', filename)
    if match:
        y, m, d, p = match.groups()
        return datetime(int(y), int(m), int(d)), p
    return None, None


def get_sorted_files(folder_path):
    files = list(folder_path.glob("*.gpkg"))
    def key(f):
        dt, p = parse_filename(f.name)
        return (dt or datetime.min, 0 if p == 'AM' else 1)
    return sorted(files, key=key)


def geom_to_verts(geom):
    """Convert geometry to vertex arrays."""
    verts = []
    if geom is None:
        return verts
    if geom.geom_type == 'Polygon':
        verts.append(np.array(geom.exterior.coords))
    elif geom.geom_type == 'MultiPolygon':
        for p in geom.geoms:
            verts.append(np.array(p.exterior.coords))
    return verts


def get_available_folders(base_path="Datasets/Snapshot"):
    base = Path(base_path)
    if not base.exists():
        return []
    return [f.name for f in sorted(base.iterdir()) 
            if f.is_dir() and f.name.endswith("_Snapshot")]


class HistoricalFireSimulation:
    def __init__(self, folder_name="2017_Snapshot", base_path="Datasets/Snapshot"):
        self.folder_name = folder_name
        self.folder_path = Path(base_path) / folder_name
        self.current_frame = -1
        
        # Cache paths
        cache_dir = Path("historical_sim/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"{folder_name}_cumulative.pkl"
        
        # Load California
        self.ca_verts = self._load_california()
        
        # Load or build cache
        if self.cache_file.exists():
            print("Loading cache...")
            self._load_cache()
        else:
            print("Building cache (one-time, may take a few minutes)...")
            self._build_cache()
        
        print(f"Ready: {len(self.cumulative_verts)} frames")
        self.setup_plot()
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
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15)
        self.ax.axis("off")
        self.ax.set_facecolor('white')
        
        self.ax.set_xlim(self.bounds[0], self.bounds[2])
        self.ax.set_ylim(self.bounds[1], self.bounds[3])
        
        # California (static)
        if self.ca_verts:
            ca_coll = PolyCollection(self.ca_verts, facecolor='green',
                                     edgecolor='darkgreen', alpha=0.8)
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
    
    def start_anim(self):
        def update(_):
            if self.is_playing:
                next_frame = self.current_frame + 1
                if next_frame >= len(self.cumulative_verts):
                    # Stop at end instead of looping
                    self.is_playing = False
                    self.play_btn.label.set_text('Play')
                    return
                self.update_display(next_frame)
                self.fig.canvas.draw_idle()
        self.animation = FuncAnimation(self.fig, update, interval=self.interval,
                                       blit=False, cache_frame_data=False)
        plt.draw()
    
    def reset(self, _):
        self.is_playing = False
        self.play_btn.label.set_text('Play')
        if self.animation:
            self.animation.event_source.stop()
        self.current_frame = -1
        self.update_display(0)
        self.fig.canvas.draw_idle()
    
    def speed(self, delta):
        self.interval = max(10, min(500, self.interval + delta))
        if self.is_playing and self.animation:
            self.animation.event_source.stop()
            self.start_anim()
    
    def run(self):
        plt.show()


def main():
    print("\n" + "="*50)
    print("California Wildfire Historical Visualization")
    print("="*50)
    
    available = get_available_folders()
    if not available:
        print("ERROR: No snapshot folders found!")
        return
    
    folder = "2017_Snapshot" if "2017_Snapshot" in available else available[0]
    sim = HistoricalFireSimulation(folder_name=folder)
    sim.run()


if __name__ == "__main__":
    main()
