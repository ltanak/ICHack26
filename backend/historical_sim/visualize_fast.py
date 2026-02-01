"""
Historical Fire Time Series Visualization - Fast playback using pre-rendered frames.

Run `pixi run historical-preprocess` first to generate the cache,
then `pixi run historical` for fast playback.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.image import imread
from pathlib import Path
import pickle
from typing import List

# Animation parameters  
ANIMATION_INTERVAL = 50  # milliseconds between frames (fast!)


def get_available_folders(base_path: str = "Datasets/Snapshot") -> List[str]:
    """Get list of available snapshot folders."""
    base = Path(base_path)
    folders = []
    for folder in sorted(base.iterdir()):
        if folder.is_dir() and folder.name.endswith("_Snapshot"):
            folders.append(folder.name)
    return folders


def check_cache_exists(folder_name: str) -> bool:
    """Check if preprocessed cache exists."""
    cache_dir = Path("historical_sim/cache") / folder_name
    metadata_file = cache_dir / "metadata.pkl"
    return metadata_file.exists()


def load_cache(folder_name: str):
    """Load preprocessed cache metadata."""
    cache_dir = Path("historical_sim/cache") / folder_name
    metadata_file = cache_dir / "metadata.pkl"
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    return metadata


class HistoricalFireSimulation:
    """Fast historical fire visualization using pre-rendered frames."""
    
    def __init__(self, folder_name: str = "2017_Snapshot"):
        self.folder_name = folder_name
        self.current_frame = 0
        
        # Load cache
        print(f"Loading cached frames for {folder_name}...")
        self.metadata = load_cache(folder_name)
        self.num_frames = self.metadata['num_frames']
        self.frame_data = self.metadata['frame_data']
        
        # Pre-load all images into memory for maximum speed
        print(f"Loading {self.num_frames} frames into memory...")
        self.images = []
        for i, (frame_path, date_str) in enumerate(self.frame_data):
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{self.num_frames}...")
            self.images.append(imread(frame_path))
        print("  Done!")
        
        # Setup plot
        self.setup_plot()
        self.update_display(0)
    
    def setup_plot(self):
        """Create the main plot with controls."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15)
        self.ax.axis("off")
        
        # Display first frame
        self.img_display = self.ax.imshow(self.images[0])
        
        # Add time slider
        ax_slider = plt.axes([0.2, 0.06, 0.6, 0.03])
        self.time_slider = Slider(
            ax_slider, 'Time',
            0, max(1, self.num_frames - 1),
            valinit=0, valstep=1
        )
        self.time_slider.on_changed(self.on_slider_change)
        
        # Add Play/Pause button
        ax_play = plt.axes([0.2, 0.01, 0.1, 0.04])
        self.play_btn = Button(ax_play, 'Play')
        self.play_btn.on_clicked(self.toggle_play)
        
        # Add Reset button
        ax_reset = plt.axes([0.35, 0.01, 0.1, 0.04])
        self.reset_btn = Button(ax_reset, 'Reset')
        self.reset_btn.on_clicked(self.reset)
        
        # Speed buttons
        ax_slower = plt.axes([0.55, 0.01, 0.1, 0.04])
        self.slower_btn = Button(ax_slower, 'Slower')
        self.slower_btn.on_clicked(self.slower)
        
        ax_faster = plt.axes([0.7, 0.01, 0.1, 0.04])
        self.faster_btn = Button(ax_faster, 'Faster')
        self.faster_btn.on_clicked(self.faster)
        
        # Animation state
        self.is_playing = False
        self.animation = None
        self.interval = ANIMATION_INTERVAL
    
    def update_display(self, frame_idx: int):
        """Update display - just swap the image (super fast!)."""
        frame_idx = int(frame_idx) % self.num_frames
        self.current_frame = frame_idx
        
        # Just update the image data - very fast!
        self.img_display.set_data(self.images[frame_idx])
        
        # Update slider without triggering callback
        if hasattr(self, 'time_slider'):
            self.time_slider.eventson = False
            self.time_slider.set_val(frame_idx)
            self.time_slider.eventson = True
        
        return [self.img_display]
    
    def on_slider_change(self, val):
        """Handle slider movement."""
        if not self.is_playing:
            self.update_display(int(val))
            self.fig.canvas.draw_idle()
    
    def toggle_play(self, event):
        """Toggle play/pause animation."""
        if self.is_playing:
            self.is_playing = False
            self.play_btn.label.set_text('Play')
            if self.animation:
                self.animation.event_source.stop()
        else:
            self.is_playing = True
            self.play_btn.label.set_text('Pause')
            self.start_animation()
    
    def start_animation(self):
        """Start the animation."""
        def anim_update(frame):
            if self.is_playing:
                next_frame = (self.current_frame + 1) % self.num_frames
                return self.update_display(next_frame)
            return [self.img_display]
        
        self.animation = FuncAnimation(
            self.fig, anim_update,
            interval=self.interval,
            blit=True,  # Use blitting for speed
            cache_frame_data=False
        )
        plt.draw()
    
    def reset(self, event):
        """Reset to first frame."""
        self.is_playing = False
        self.play_btn.label.set_text('Play')
        if self.animation:
            self.animation.event_source.stop()
        self.update_display(0)
        self.fig.canvas.draw_idle()
    
    def slower(self, event):
        """Slow down animation."""
        self.interval = min(500, self.interval + 25)
        if self.animation and self.is_playing:
            self.animation.event_source.stop()
            self.start_animation()
    
    def faster(self, event):
        """Speed up animation."""
        self.interval = max(10, self.interval - 25)
        if self.animation and self.is_playing:
            self.animation.event_source.stop()
            self.start_animation()
    
    def run(self):
        """Start the visualization."""
        plt.show()


def main():
    """Run the historical fire simulation."""
    print("\n" + "="*60)
    print("California Wildfire Historical Time Series")
    print("="*60)
    
    available = get_available_folders()
    if not available:
        print("ERROR: No snapshot folders found in Datasets/Snapshot/")
        return
    
    default_folder = "2017_Snapshot" if "2017_Snapshot" in available else available[0]
    
    # Check if cache exists
    if not check_cache_exists(default_folder):
        print(f"\nNo cache found for {default_folder}")
        print("Running preprocessing first (this only needs to be done once)...\n")
        from historical_sim.preprocess import preprocess_folder
        preprocess_folder(default_folder)
    
    print("\nStarting fast playback...")
    print("Controls: Play/Pause, Reset, Slower/Faster buttons")
    print("="*60 + "\n")
    
    sim = HistoricalFireSimulation(folder_name=default_folder)
    sim.run()


if __name__ == "__main__":
    main()
