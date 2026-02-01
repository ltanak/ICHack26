
"""
Preprocess fire data and create cached masks for fast simulation.

This script:
1. Loads the actual California boundary from Natural Earth
2. Loads fire detection points from GPKG file
3. Rasterizes both into numpy masks
4. Saves masks to cache/ directory for fast loading

Run this once to generate cache files.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from pathlib import Path

# Constants
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

BURNING = 2


def create_ca_mask(ca_polygon, N=400):
    """
    Create a rasterized mask of California.
    
    Args:
        ca_polygon: Shapely polygon representing California boundary
        N: Grid resolution (NxN)
    
    Returns:
        ca_mask: NxN array where 1=inside CA, 0=outside
        bounds: (minx, miny, maxx, maxy) of the grid
    """
    minx, miny, maxx, maxy = ca_polygon.bounds
    
    # Create grid of points
    xs = np.linspace(minx, maxx, N)
    ys = np.linspace(miny, maxy, N)
    
    # Rasterize using simple point-in-polygon test
    ca_mask = np.zeros((N, N), dtype=int)
    
    print("Rasterizing California boundary...")
    for i, y in enumerate(ys):
        if i % 50 == 0:
            print(f"  Progress: {i}/{N}")
        for j, x in enumerate(xs):
            point = Point(x, y)
            if ca_polygon.contains(point):
                ca_mask[i, j] = 1
    
    print(f"  Done! {ca_mask.sum()} cells inside CA")
    return ca_mask, (minx, miny, maxx, maxy)


def create_fire_mask(fires_gdf, ca_mask, bounds, N=400):
    """
    Create a mask showing initial fire locations.
    
    Args:
        fires_gdf: GeoDataFrame with fire detection points
        ca_mask: California mask (to only mark fires inside CA)
        bounds: (minx, miny, maxx, maxy)
        N: Grid resolution
    
    Returns:
        fire_mask: NxN array where BURNING=2 at fire locations
    """
    minx, miny, maxx, maxy = bounds
    fire_mask = np.zeros((N, N), dtype=int)
    
    print("Rasterizing fire points...")
    fire_count = 0
    
    for _, row in fires_gdf.iterrows():
        geom = row['geometry']
        
        # Handle both Point and MultiPoint geometries
        points = []
        if geom.geom_type == 'Point':
            points = [geom]
        elif geom.geom_type == 'MultiPoint':
            points = list(geom.geoms)
        else:
            continue
        
        for point in points:
            lon, lat = point.x, point.y
            
            # Convert to grid indices
            x_idx = int((lon - minx) / (maxx - minx) * (N - 1))
            y_idx = int((lat - miny) / (maxy - miny) * (N - 1))
            
            # Only mark if within grid and inside CA
            if 0 <= x_idx < N and 0 <= y_idx < N:
                if ca_mask[y_idx, x_idx] == 1:
                    fire_mask[y_idx, x_idx] = BURNING
                    fire_count += 1
    
    print(f"  Done! {fire_count} fire points marked")
    return fire_mask


def run_preprocessing(gpkg_path, N=400):
    """
    Main preprocessing pipeline.
    
    Args:
        gpkg_path: Path to GPKG file with fire data
        N: Grid resolution (default 400x400)
    """
    print(f"\n{'='*60}")
    print(f"Starting preprocessing: {gpkg_path}")
    print(f"Grid resolution: {N}x{N}")
    print(f"{'='*60}\n")
    
    # 1. Load fire data
    print("Loading fire data from GPKG...")
    fires_gdf = gpd.read_file(gpkg_path, layer="newfirepix")
    print(f"  Loaded {len(fires_gdf)} fire detections")
    
    # 2. Load actual California boundary from Natural Earth
    print("\nLoading California boundary from Natural Earth...")
    states = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
    )
    ca_gdf = states[states["name"] == "California"]
    ca_polygon = ca_gdf.geometry.iloc[0]  # Get the actual polygon
    print(f"  Loaded California polygon")
    
    minx, miny, maxx, maxy = ca_polygon.bounds
    print(f"  Bounds: ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f})")
    
    # 3. Rasterize California
    ca_mask, bounds = create_ca_mask(ca_polygon, N)
    
    # 4. Rasterize fire points
    fire_mask = create_fire_mask(fires_gdf, ca_mask, bounds, N)
    
    # 5. Save to cache
    ca_cache_file = CACHE_DIR / "ca_mask.npz"
    fire_cache_file = CACHE_DIR / "fire_mask.npz"
    
    print(f"\nSaving cache files...")
    np.savez(ca_cache_file, ca_mask=ca_mask, minx=minx, miny=miny, maxx=maxx, maxy=maxy, N=N)
    print(f"  ✓ {ca_cache_file}")
    
    np.savez(fire_cache_file, fire_mask=fire_mask)
    print(f"  ✓ {fire_cache_file}")
    
    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Run preprocessing on the January 1, 2020 AM snapshot
    gpkg_file = "Datasets/Snapshot/2020_Snapshot/20200101AM.gpkg"
    run_preprocessing(gpkg_file, N=400)