"""
Preprocess historical fire data - renders all frames to images for fast playback.
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re
from typing import List, Tuple
import pickle
from PIL import Image
import io


def parse_filename(filename: str) -> Tuple[datetime, str]:
    """Parse snapshot filename to extract date and period (AM/PM)."""
    match = re.match(r'(\d{4})(\d{2})(\d{2})(AM|PM)\.gpkg', filename)
    if match:
        year, month, day, period = match.groups()
        dt = datetime(int(year), int(month), int(day))
        return dt, period
    raise ValueError(f"Cannot parse filename: {filename}")


def get_sorted_files(folder_path: Path) -> List[Path]:
    """Get all gpkg files sorted chronologically."""
    files = list(folder_path.glob("*.gpkg"))
    
    def sort_key(path):
        try:
            dt, period = parse_filename(path.name)
            return (dt, 0 if period == 'AM' else 1)
        except:
            return (datetime.min, 0)
    
    return sorted(files, key=sort_key)


def load_perimeters(gpkg_file: Path) -> gpd.GeoDataFrame:
    """Load fire perimeters from a gpkg file."""
    try:
        gdf = gpd.read_file(gpkg_file, layer="perimeter")
        gdf = gdf[gdf.geometry.notna()]
        return gdf
    except Exception as e:
        return gpd.GeoDataFrame()


def load_california_boundary():
    """Load California boundary from Natural Earth."""
    try:
        print("Loading California boundary...")
        states = gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
        )
        ca_gdf = states[states["name"] == "California"]
        return ca_gdf
    except Exception as e:
        print(f"Warning: Could not load California boundary: {e}")
        return None


def preprocess_folder(folder_name: str = "2017_Snapshot", 
                      base_path: str = "Datasets/Snapshot"):
    """Pre-render all frames to images for fast playback."""
    
    folder_path = Path(base_path) / folder_name
    cache_dir = Path("historical_sim/cache") / folder_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    metadata_file = cache_dir / "metadata.pkl"
    
    print(f"\n{'='*60}")
    print(f"Preprocessing: {folder_name}")
    print(f"{'='*60}")
    
    # Load files
    files = get_sorted_files(folder_path)
    print(f"Found {len(files)} snapshot files")
    
    if not files:
        print("ERROR: No files found!")
        return
    
    # Load California boundary
    ca_boundary = load_california_boundary()
    
    # Calculate bounds
    if ca_boundary is not None and len(ca_boundary) > 0:
        bounds = ca_boundary.total_bounds
        minx, miny, maxx, maxy = bounds
        width = maxx - minx
        height = maxy - miny
        buffer = 0.05
        bounds = (minx - width * buffer, 
                  miny - height * buffer,
                  maxx + width * buffer, 
                  maxy + height * buffer)
    else:
        bounds = (-124.5, 32.5, -114.0, 42.0)
    
    # Pre-load all perimeter data
    print("\nLoading perimeter data...")
    perimeter_cache = {}
    for i, f in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(files)} files...")
        perimeters = load_perimeters(f)
        if len(perimeters) > 0:
            perimeter_cache[i] = perimeters
    print(f"  Done loading {len(perimeter_cache)} frames with data")
    
    # Render frames
    print("\nRendering frames to images...")
    
    year = folder_name.split('_')[0]
    accumulated = []
    frame_data = []  # Store (image_path, date_str) for each frame
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    for i, f in enumerate(files):
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(f"  Rendering frame {i + 1}/{len(files)}...")
        
        # Accumulate perimeters
        if i in perimeter_cache:
            accumulated.append(perimeter_cache[i])
        
        # Clear and redraw
        ax.clear()
        ax.axis("off")
        ax.set_facecolor('white')
        
        minx, miny, maxx, maxy = bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        
        # Draw California
        if ca_boundary is not None:
            ca_boundary.plot(ax=ax, facecolor='green', 
                            edgecolor='darkgreen', alpha=0.8, linewidth=1)
        
        # Draw accumulated fires
        for perimeters in accumulated:
            if len(perimeters) > 0:
                perimeters.plot(ax=ax, facecolor='red', 
                               edgecolor='darkred', alpha=0.8, linewidth=1)
        
        # Get date string
        try:
            dt, period = parse_filename(f.name)
            date_str = dt.strftime('%B %d, %Y') + f" {period}"
        except:
            date_str = f.stem
        
        ax.set_title(f"California Wildfire Historical Data - {year}\n{date_str}", 
                     fontsize=14, pad=10)
        
        # Save frame as PNG
        frame_path = cache_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, bbox_inches='tight', facecolor='white', dpi=100)
        frame_data.append((str(frame_path), date_str))
    
    plt.close(fig)
    
    # Save metadata
    metadata = {
        'folder_name': folder_name,
        'num_frames': len(files),
        'frame_data': frame_data,
        'bounds': bounds,
        'files': [str(f) for f in files]
    }
    
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"Cached {len(files)} frames to: {cache_dir}")
    print(f"{'='*60}\n")


def get_available_folders(base_path: str = "Datasets/Snapshot") -> List[str]:
    """Get list of available snapshot folders."""
    base = Path(base_path)
    folders = []
    for folder in sorted(base.iterdir()):
        if folder.is_dir() and folder.name.endswith("_Snapshot"):
            folders.append(folder.name)
    return folders


if __name__ == "__main__":
    available = get_available_folders()
    if not available:
        print("ERROR: No snapshot folders found!")
    else:
        default_folder = "2017_Snapshot" if "2017_Snapshot" in available else available[0]
        preprocess_folder(default_folder)
