"""
HRRR Weather Data Pipeline for Wildfire Spread Modeling

Downloads NOAA HRRR weather data within 50km buffer of active fire perimeters.
Uses 12-hour cadence (06Z/18Z) aligned with VIIRS AM/PM observations.

Output: Parquet files partitioned by year/date with weather features:
- wind_speed_ms, wind_direction_deg (from U/V at 10m)
- temperature_c (from TMP at 2m)  
- dewpoint_c (from DPT at 2m)
- relative_humidity_pct (from RH at 2m)
- precip_mm (from APCP)

Data source: AWS Open Data (noaa-hrrr-bdp-pds)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from herbie import Herbie
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

# Configuration
BUFFER_KM = 50  # Buffer around fire centroids
HOURS = [6, 18]  # 06Z and 18Z (late evening and late morning CA time)
HRRR_START_DATE = datetime(2016, 8, 1)  # HRRR archive starts here on AWS

# HRRR variable search patterns
VARIABLES = {
    'u10': ':UGRD:10 m above ground:anl:',
    'v10': ':VGRD:10 m above ground:anl:',
    'tmp2m': ':TMP:2 m above ground:anl:',
    'dpt2m': ':DPT:2 m above ground:anl:',
    'rh2m': ':RH:2 m above ground:anl:',   # Relative humidity
    'precip': ':APCP:surface:'
}


def load_fire_perimeters(perimeter_file, years=None, min_area_km2=100):
    """
    Load fire perimeter data and extract centroids with date ranges.
    
    Args:
        perimeter_file: Path to GeoPackage
        years: List of years to include
        min_area_km2: Minimum fire area in km² (default 100)
    
    Returns DataFrame with: fireID, year, clat, clon, start_date, end_date
    """
    gdf = gpd.read_file(perimeter_file)
    
    # Filter by years if specified
    if years:
        gdf = gdf[gdf['year'].isin(years)]
    
    # Filter by minimum area (for significant fires only)
    if min_area_km2 > 0:
        gdf = gdf[gdf['farea'] >= min_area_km2]
        print(f"Filtered to fires with area >= {min_area_km2} km²: {len(gdf)} fires")
    
    # Parse start/end dates
    def parse_date(date_str):
        # Format: YYYYMMDD[AM/PM]
        date_part = date_str[:8]
        return datetime.strptime(date_part, '%Y%m%d')
    
    fires = []
    for _, row in gdf.iterrows():
        try:
            start_date = parse_date(row['tst'])
            end_date = parse_date(row['ted'])
            
            # Only include fires after HRRR archive starts
            if end_date >= HRRR_START_DATE:
                fires.append({
                    'fireID': row['fireID'],
                    'year': row['year'],
                    'clat': row['clat'],
                    'clon': row['clon'],
                    'farea_km2': row['farea'],
                    'start_date': max(start_date, HRRR_START_DATE),
                    'end_date': end_date,
                    'duration_days': (end_date - start_date).days + 1
                })
        except:
            continue
    
    return pd.DataFrame(fires)


def get_unique_fire_dates(fires_df):
    """
    Get unique (date, hour) combinations needed across all fires.
    Returns dict mapping (date_str, hour) -> list of fire centroids active that day.
    """
    date_fires = {}
    
    for _, fire in fires_df.iterrows():
        current = fire['start_date']
        while current <= fire['end_date']:
            date_str = current.strftime('%Y-%m-%d')
            for hour in HOURS:
                key = (date_str, hour)
                if key not in date_fires:
                    date_fires[key] = []
                date_fires[key].append({
                    'fireID': fire['fireID'],
                    'clat': fire['clat'],
                    'clon': fire['clon']
                })
            current += timedelta(days=1)
    
    return date_fires


def download_hrrr_variables(date_str, hour):
    """
    Download all required HRRR variables for a given date/hour.
    Returns dict of xarray datasets or None if failed.
    """
    try:
        H = Herbie(
            date=f"{date_str} {hour:02d}:00",
            model="hrrr",
            product="sfc",
            fxx=0,
        )
        
        datasets = {}
        for var_name, search_pattern in VARIABLES.items():
            try:
                ds = H.xarray(search_pattern, remove_grib=True)
                datasets[var_name] = ds
            except Exception as e:
                # Precip might not always be available
                if var_name != 'precip':
                    print(f"Warning: Could not get {var_name} for {date_str} {hour:02d}Z: {e}")
        
        return datasets if datasets else None
        
    except Exception as e:
        print(f"Error downloading HRRR for {date_str} {hour:02d}Z: {e}")
        return None


def extract_fire_buffer_data(datasets, fire_centroids, timestamp, buffer_km=50):
    """
    Extract weather data within buffer_km of each fire centroid.
    
    Returns list of records with weather features.
    """
    records = []
    
    # Get coordinate grids (same for all variables)
    first_ds = list(datasets.values())[0]
    lats = first_ds.latitude.values
    lons = first_ds.longitude.values
    
    # Convert HRRR longitude (0-360) to standard (-180 to 180)
    lons_standard = np.where(lons > 180, lons - 360, lons)
    
    # Get variable arrays
    u10 = datasets.get('u10')
    v10 = datasets.get('v10')
    tmp2m = datasets.get('tmp2m')
    dpt2m = datasets.get('dpt2m')
    rh2m = datasets.get('rh2m')
    precip = datasets.get('precip')
    
    u_vals = list(u10.data_vars.values())[0].values if u10 else None
    v_vals = list(v10.data_vars.values())[0].values if v10 else None
    tmp_vals = list(tmp2m.data_vars.values())[0].values if tmp2m else None
    dpt_vals = list(dpt2m.data_vars.values())[0].values if dpt2m else None
    rh_vals = list(rh2m.data_vars.values())[0].values if rh2m else None
    precip_vals = list(precip.data_vars.values())[0].values if precip else None
    
    # Process each fire centroid
    for fire in fire_centroids:
        fire_lat = fire['clat']
        fire_lon = fire['clon']
        fire_id = fire['fireID']
        
        # Calculate approximate distance (in km) using simple formula
        # At mid-latitudes, 1 degree lat ≈ 111 km, 1 degree lon ≈ 85 km
        lat_diff_km = np.abs(lats - fire_lat) * 111
        lon_diff_km = np.abs(lons_standard - fire_lon) * 85
        distance_km = np.sqrt(lat_diff_km**2 + lon_diff_km**2)
        
        # Mask to points within buffer
        buffer_mask = distance_km <= buffer_km
        
        if not np.any(buffer_mask):
            continue
        
        # Extract data within buffer
        buffer_lats = lats[buffer_mask]
        buffer_lons = lons_standard[buffer_mask]
        
        buffer_u = u_vals[buffer_mask] if u_vals is not None else None
        buffer_v = v_vals[buffer_mask] if v_vals is not None else None
        buffer_tmp = tmp_vals[buffer_mask] if tmp_vals is not None else None
        buffer_dpt = dpt_vals[buffer_mask] if dpt_vals is not None else None
        buffer_rh = rh_vals[buffer_mask] if rh_vals is not None else None
        buffer_precip = precip_vals[buffer_mask] if precip_vals is not None else None
        
        # Calculate derived features
        if buffer_u is not None and buffer_v is not None:
            wind_speed = np.sqrt(buffer_u**2 + buffer_v**2)
            wind_direction = (270 - np.degrees(np.arctan2(buffer_v, buffer_u))) % 360
        else:
            wind_speed = np.full(len(buffer_lats), np.nan)
            wind_direction = np.full(len(buffer_lats), np.nan)
        
        # Convert temperature from Kelvin to Celsius
        if buffer_tmp is not None:
            temp_c = buffer_tmp - 273.15
        else:
            temp_c = np.full(len(buffer_lats), np.nan)
        
        if buffer_dpt is not None:
            dewpoint_c = buffer_dpt - 273.15
        else:
            dewpoint_c = np.full(len(buffer_lats), np.nan)
        
        # Relative humidity (already in %)
        if buffer_rh is None:
            buffer_rh = np.full(len(buffer_lats), np.nan)
        
        if buffer_precip is None:
            buffer_precip = np.full(len(buffer_lats), np.nan)
        
        # Create records
        for i in range(len(buffer_lats)):
            records.append({
                'timestamp': timestamp,
                'fireID': fire_id,
                'latitude': float(buffer_lats[i]),
                'longitude': float(buffer_lons[i]),
                'wind_speed_ms': float(wind_speed[i]),
                'wind_direction_deg': float(wind_direction[i]),
                'temperature_c': float(temp_c[i]),
                'dewpoint_c': float(dewpoint_c[i]),
                'relative_humidity_pct': float(buffer_rh[i]),
                'precip_mm': float(buffer_precip[i])
            })
    
    return records


def process_fire_weather(perimeter_file, output_dir, years=[2017, 2018, 2019, 2020], min_area_km2=100):
    """
    Main pipeline: download HRRR data for all fire events and save as Parquet.
    
    Args:
        perimeter_file: Path to fire perimeter GeoPackage
        output_dir: Output directory for Parquet files
        years: List of years to process
        min_area_km2: Minimum fire area filter (default 100 km²)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading fire perimeter data...")
    fires_df = load_fire_perimeters(perimeter_file, years=years, min_area_km2=min_area_km2)
    print(f"Found {len(fires_df)} fires in years {years} with area >= {min_area_km2} km²")
    
    # Get unique date/hour combinations
    print("Computing date/fire mapping...")
    date_fires = get_unique_fire_dates(fires_df)
    print(f"Total date/hour combinations to process: {len(date_fires)}")
    
    # Group by year for partitioned output
    all_records = []
    processed_count = 0
    
    # Sort by date for better caching
    sorted_keys = sorted(date_fires.keys())
    
    for date_str, hour in tqdm(sorted_keys, desc="Downloading HRRR data"):
        fire_centroids = date_fires[(date_str, hour)]
        
        # Download HRRR data
        datasets = download_hrrr_variables(date_str, hour)
        
        if datasets is None:
            continue
        
        # Extract timestamp
        timestamp = datetime.strptime(f"{date_str} {hour:02d}:00", '%Y-%m-%d %H:%M')
        
        # Extract data for fire buffers
        records = extract_fire_buffer_data(datasets, fire_centroids, timestamp, BUFFER_KM)
        all_records.extend(records)
        
        processed_count += 1
        
        # Close datasets
        for ds in datasets.values():
            ds.close()
        
        # Periodically save to avoid memory issues
        if len(all_records) > 1_000_000:
            save_records_to_parquet(all_records, output_dir)
            all_records = []
    
    # Save remaining records
    if all_records:
        save_records_to_parquet(all_records, output_dir)
    
    print(f"\nProcessed {processed_count} timestamps")
    return output_dir


def save_records_to_parquet(records, output_dir):
    """
    Save records to Parquet, partitioned by year and date.
    """
    if not records:
        return
    
    df = pd.DataFrame(records)
    df['year'] = df['timestamp'].dt.year
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    
    # Write partitioned parquet
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(
        table,
        root_path=output_dir,
        partition_cols=['year', 'date']
    )
    
    print(f"Saved {len(df)} records to Parquet")


def run_sanity_check(output_dir):
    """
    Run sanity check on the output data.
    """
    print("\n" + "="*50)
    print("SANITY CHECK")
    print("="*50)
    
    # Read all parquet files
    df = pd.read_parquet(output_dir)
    
    print(f"\nTotal rows: {len(df):,}")
    print(f"Unique timestamps: {df['timestamp'].nunique()}")
    print(f"Unique fires: {df['fireID'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Lat range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    print(f"Lon range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
    
    print("\nFeature statistics:")
    print(df[['wind_speed_ms', 'wind_direction_deg', 'temperature_c', 'dewpoint_c', 'relative_humidity_pct', 'precip_mm']].describe())
    
    print("\nSample data (head):")
    print(df.head(10))
    
    # File size
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, filenames in os.walk(output_dir)
        for f in filenames if f.endswith('.parquet')
    )
    print(f"\nTotal Parquet size: {total_size / 1024 / 1024:.1f} MB")


def estimate_dataset_size(perimeter_file, years=[2017, 2018, 2019, 2020]):
    """
    Estimate the final dataset size without downloading.
    """
    print("Estimating dataset size...")
    
    fires_df = load_fire_perimeters(perimeter_file, years=years)
    print(f"Fires in {years}: {len(fires_df)}")
    
    # Get unique date/hour combinations
    date_fires = get_unique_fire_dates(fires_df)
    
    total_timestamps = len(date_fires)
    avg_fires_per_timestamp = np.mean([len(v) for v in date_fires.values()])
    
    # Estimate grid points per fire (50km buffer at 3km resolution)
    # Circle area = pi * r^2, grid cells ≈ area / 9 km^2
    grid_points_per_fire = int(np.pi * (BUFFER_KM**2) / 9)
    
    estimated_rows = total_timestamps * avg_fires_per_timestamp * grid_points_per_fire
    
    # Estimate size: ~100 bytes per row in Parquet (compressed)
    estimated_size_mb = estimated_rows * 100 / 1024 / 1024
    
    print(f"\nEstimate:")
    print(f"  Total timestamps: {total_timestamps:,}")
    print(f"  Avg active fires per timestamp: {avg_fires_per_timestamp:.1f}")
    print(f"  Grid points per fire buffer: ~{grid_points_per_fire:,}")
    print(f"  Estimated total rows: ~{estimated_rows:,.0f}")
    print(f"  Estimated Parquet size: ~{estimated_size_mb:.0f} MB")
    
    return estimated_rows, estimated_size_mb


if __name__ == "__main__":
    # Paths (use absolute paths based on script location)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PERIMETER_FILE = os.path.join(SCRIPT_DIR, "../Snapshot/Finalperimeter_2012-2020.gpkg")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "hrrr_fire_weather")
    
    # Configuration
    YEARS = [2017, 2018, 2019, 2020]
    MIN_AREA_KM2 = 100  # Only fires >= 100 km²
    
    print("="*60)
    print("HRRR Weather Data Download for Wildfire Spread Modeling")
    print("="*60)
    print(f"Years: {YEARS}")
    print(f"Minimum fire area: {MIN_AREA_KM2} km²")
    print(f"Time steps: 06Z and 18Z (12-hour cadence)")
    print(f"Buffer: {BUFFER_KM} km around fire centroids")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # Process the data
    process_fire_weather(
        PERIMETER_FILE, 
        OUTPUT_DIR, 
        years=YEARS, 
        min_area_km2=MIN_AREA_KM2
    )
    
    # Run sanity check
    run_sanity_check(OUTPUT_DIR)
