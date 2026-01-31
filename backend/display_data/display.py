import geopandas as gpd
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
from image_editing.image import apply_transparency

# Read all gpkg files from Datasets 2017
# def read_

#gets the final state
def get_coords(year: int) -> list[int]:
    path = Path(f"Datasets/Snapshot/{year}_Snapshot/")
    gpkg_file = sorted(path.glob("*.gpkg"))[-1]
    gdf = gpd.read_file(gpkg_file, layer="perimeter")
    bounds = gdf.total_bounds
    min_long, min_lat, max_long, max_lat = bounds
# print(gdf.head())

    return [min_long, min_lat, max_long, max_lat]

def get_image_path(year: int) -> Path:
    output_dir = Path("Datasets/Images")
    return output_dir / f"{year}.png"

def save_image(year: int):
    path = Path(f"Datasets/Snapshot/{year}_Snapshot/")
    gpkg_file = sorted(path.glob("*.gpkg"))[-1]
    gdf = gpd.read_file(gpkg_file, layer="perimeter")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
    gdf.plot(ax=ax, alpha=0.7, edgecolor='darkred', facecolor='red', linewidth=2)

    # Get bounds - use exact data bounds without padding
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    # Set exact limits to data bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Remove axes, labels, and ticks
    ax.axis('off')

    # Save to image file in display_data/output
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{gpkg_file.stem}_spread.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"\nImage saved to: {output_path}")
    plt.close()


def overlay_image(year: int, satellite_csv: Path, out):
    path_img = get_image_path(year)
    return apply_transparency(path_img, satellite_csv)

# datasets_2017_dir = Path("Datasets/Snapshot/2020_Snapshot/")

print(f"Found {len(gpkg_files)} GPKG files in Datasets 2017:")
# Read the last file (final state)
# gpkg_file = gpkg_files[-1]
# print(f"\nReading: {gpkg_file.name}")
gdf = gpd.read_file(gpkg_file, layer="perimeter")
print(gdf.head())

# Plot and save as image
fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
gdf.plot(ax=ax, alpha=0.7, edgecolor='darkred', facecolor='red', linewidth=2)

# Get bounds - use exact data bounds without padding
bounds = gdf.total_bounds
minx, miny, maxx, maxy = bounds

# Set exact limits to data bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

# Remove axes, labels, and ticks
ax.axis('off')

# Print corner coordinates
print(f"\nCorner Coordinates:")
print(f"Bottom-left (minx, miny): ({minx}, {miny})")
print(f"Bottom-right (maxx, miny): ({maxx}, {miny})")
print(f"Top-left (minx, maxy): ({minx}, {maxy})")
print(f"Top-right (maxx, maxy): ({maxx}, {maxy})")

# Save to image file in display_data/output
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / f"{gpkg_file.stem}_spread.png"
plt.savefig(output_path, dpi=300, facecolor='white')
print(f"\nImage saved to: {output_path}")
plt.close()

