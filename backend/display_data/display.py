import geopandas as gpd
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
from image_editing.image import apply_transparency
import json
import geopandas as gpd
import requests
from shapely.geometry import box
import pandas as pd

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
    output_dir = Path("Datasets/images")
    return output_dir / f"{year}.png"

def get_satellite_path(year: int) -> Path:
    output_dir = Path("Datasets/satellite")
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


def overlay_image(year: int, satellite_path: Path) -> Path:
    """Overlay fire spread image on satellite image and return path to result."""
    fire_path = get_image_path(year)
    apply_transparency(satellite_path, fire_path, year=year)
    
    # Return path to the saved overlay
    overlay_path = Path("display_data/overlays") / f"{year}_overlay.png"
    return overlay_path

def get_info(year: int):
    path = Path(f"Datasets/Snapshot/{year}_Snapshot/")
    gpkg_file = sorted(path.glob("*.gpkg"))[-1]
    gdf = gpd.read_file(gpkg_file, layer="perimeter")
    
    # print(f"\nYear: {year}")
    # print(f"File: {gpkg_file.name}")
    # print(f"\nAll columns: {gdf.columns.tolist()}")
    # print(f"\nDataFrame shape: {gdf.shape}")
    # print(f"\n===== ACTUAL DATA VALUES =====")
    # print(gdf.head(10))
    # print(f"\n===== SAMPLE VALUES FROM FIRST ROW =====")
    first_row = gdf.iloc[0]
    hmap = {}
    for col in gdf.columns:
        hmap[col] = first_row[col]
    return hmap

def api_call(year: int):
    try:
        url = f"https://incidents.fire.ca.gov/umbraco/api/IncidentApi/GeoJsonList?year={year}&inactive=true"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
        return gdf
    except Exception as e:
        print(f"Error calling API: {e}")
        return None


def match_fires_by_date(year: int):
    """Match API fires with local GPKG fires and track fireID across all time snapshots.
    Returns the most relevant fire match with all info."""
    
    api_gdf = api_call(year)
    if api_gdf is None:
        print("Could not retrieve API data")
        return
    
    path = Path(f"Datasets/Snapshot/{year}_Snapshot/")
    gpkg_files = sorted(path.glob("*.gpkg"))
    
    if not gpkg_files:
        print(f"No GPKG files found for year {year}")
        return
    
    # local fire data from final snapshot only
    local_fires = []
    gpkg_file = gpkg_files[-1]  # Get the last file
    gdf = gpd.read_file(gpkg_file, layer="perimeter")
    for idx, row in gdf.iterrows():
        local_fires.append({
            'fireID': row['fireID'],
            'date': f"{row['tst_year']}-{row['tst_month']:02d}-{row['tst_day']:02d}",
            'clat': row['clat'],
            'clon': row['clon'],
            'farea': row['farea'],
            'gpkg_file': gpkg_file.name
        })
    
    local_df = pd.DataFrame(local_fires)
    
    # date column in API data
    api_gdf['StartDate_only'] = pd.to_datetime(api_gdf['Started']).dt.strftime('%Y-%m-%d')
    
    # match by date
    matches = []
    for api_idx, api_row in api_gdf.iterrows():
        api_date = api_row['StartDate_only']
        api_name = api_row['Name']
        api_county = api_row['County']
        api_lat = api_row['Latitude']
        api_lon = api_row['Longitude']
        
        # local fires with same date
        same_date_fires = local_df[local_df['date'] == api_date]
        
        if len(same_date_fires) > 0:
            # Find spatially closest fire
            same_date_fires = same_date_fires.copy().reset_index(drop=True)
            same_date_fires['distance'] = ((same_date_fires['clat'] - api_lat)**2 + 
                                          (same_date_fires['clon'] - api_lon)**2)**0.5
            closest_idx = same_date_fires['distance'].idxmin()
            closest = same_date_fires.iloc[closest_idx]
            distance_km = closest['distance'] * 111
            
            matches.append({
                'distance_km': distance_km,  # Keep for sorting, will be removed later
                'name': api_name,
                'county': api_county,
                'started': api_date,
                'acres_burned': api_row['AcresBurned'],
                'api_latitude': api_lat,
                'api_longitude': api_lon,
                'local_fireID': int(closest['fireID']),
                'local_area': closest['farea'],
                'local_latitude': closest['clat'],
                'local_longitude': closest['clon']
            })
    
    if not matches:
        print(f"No matched fires found for {year}")
        return None
    
    # Return the largest fire by acres burned (not smallest distance)
    best_match = max(matches, key=lambda x: x['acres_burned'])
    # Remove the internal distance_km field from final output
    del best_match['distance_km']
    
    # Now track this fireID across ALL snapshots to see fire progression
    matched_fireID = best_match['local_fireID']
    time_series = []
    for gpkg_file in gpkg_files:
        gdf = gpd.read_file(gpkg_file, layer="perimeter")
        fire_records = gdf[gdf['fireID'] == matched_fireID]
        
        if len(fire_records) > 0:
            for idx, row in fire_records.iterrows():
                snapshot_date = f"{row['tst_year']}-{row['tst_month']:02d}-{row['tst_day']:02d}"
                time_series.append({
                    'date': snapshot_date,
                    'filename': gpkg_file.name,
                    'area': row['farea'],
                    'latitude': row['clat'],
                    'longitude': row['clon'],
                    # 'geometry': row['geometry']
                })
    
    # best_match['time_series'] = time_series
    best_match['time_series'] = time_series
    
    # Add summary statistics
    if time_series:
        areas = [ts['area'] for ts in time_series]
        best_match['summary'] = {
            'total_snapshots': len(time_series),
            'min_area': min(areas),
            'max_area': max(areas),
            'first_recorded': time_series[0]['date'],
            'last_recorded': time_series[-1]['date']
        }
    
    del best_match["time_series"]
    return best_match



if __name__ == "__main__":
    # Match fires by date
    matches = match_fires_by_date(2020)
    # #     print(f"  {key}: {val}")
    # print(api_call(2020))
    # print(get_info(2020))
    print(matches)
    
    # Uncomment below to test API (may fail if endpoint is down)
    # result = api_call(2020)
    # print(result)

# datasets_2017_dir = Path("Datasets/Snapshot/2020_Snapshot/")

# print(f"Found {len(gpkg_files)} GPKG files in Datasets 2017:")
# # Read the last file (final state)
# # gpkg_file = gpkg_files[-1]
# # print(f"\nReading: {gpkg_file.name}")
# gdf = gpd.read_file(gpkg_file, layer="perimeter")
# print(gdf.head())

# # Plot and save as image
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
# gdf.plot(ax=ax, alpha=0.7, edgecolor='darkred', facecolor='red', linewidth=2)

# # Get bounds - use exact data bounds without padding
# bounds = gdf.total_bounds
# minx, miny, maxx, maxy = bounds

# # Set exact limits to data bounds
# ax.set_xlim(minx, maxx)
# ax.set_ylim(miny, maxy)

# # Remove axes, labels, and ticks
# ax.axis('off')

# # Print corner coordinates
# print(f"\nCorner Coordinates:")
# print(f"Bottom-left (minx, miny): ({minx}, {miny})")
# print(f"Bottom-right (maxx, miny): ({maxx}, {miny})")
# print(f"Top-left (minx, maxy): ({minx}, {maxy})")
# print(f"Top-right (maxx, maxy): ({maxx}, {maxy})")

# # Save to image file in display_data/output
# output_dir = Path(__file__).parent / "output"
# output_dir.mkdir(exist_ok=True)
# output_path = output_dir / f"{gpkg_file.stem}_spread.png"
# plt.savefig(output_path, dpi=300, facecolor='white')
# print(f"\nImage saved to: {output_path}")
# plt.close()

