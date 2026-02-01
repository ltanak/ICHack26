"""
Fire Snapshot Processing Module

Loads fire perimeter snapshots from GeoPackage files and computes
time-varying target variables (area change, spread rate, perimeter growth).
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm


def parse_snapshot_filename(filename: str) -> Tuple[datetime, str]:
    """
    Parse snapshot filename to extract date and AM/PM period.
    
    Args:
        filename: Filename like "20170815AM.gpkg"
        
    Returns:
        (datetime, period): Parsed date and "AM"/"PM"
    """
    basename = Path(filename).stem  # "20170815AM"
    date_str = basename[:8]  # "20170815"
    period = basename[8:]     # "AM" or "PM"
    
    date = datetime.strptime(date_str, "%Y%m%d")
    return date, period


def create_snapshot_key(date: datetime, period: str) -> str:
    """
    Create a standardized snapshot key for joining.
    
    Args:
        date: Snapshot date
        period: "AM" or "PM"
        
    Returns:
        Key like "20170815AM"
    """
    return f"{date.strftime('%Y%m%d')}{period}"


def load_single_snapshot(filepath: Path) -> Optional[gpd.GeoDataFrame]:
    """
    Load a single fire snapshot GeoPackage.
    
    Args:
        filepath: Path to .gpkg file
        
    Returns:
        GeoDataFrame with fire perimeters, or None if empty/invalid
    """
    try:
        gdf = gpd.read_file(filepath)
        if gdf.empty:
            return None
        
        # Ensure CRS is set (assume WGS84 if missing)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        
        # Add snapshot metadata
        date, period = parse_snapshot_filename(filepath.name)
        gdf['snapshot_date'] = date
        gdf['snapshot_period'] = period
        gdf['snapshot_key'] = create_snapshot_key(date, period)
        
        return gdf
        
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def load_fire_snapshots(
    snapshot_dir: Path,
    years: List[int],
    min_area_km2: float = 1.0,
    fire_id_col: str = 'fireID'
) -> gpd.GeoDataFrame:
    """
    Load all fire snapshots for specified years.
    
    Args:
        snapshot_dir: Base directory containing year folders
        years: List of years to process (e.g., [2017, 2018, 2019, 2020])
        min_area_km2: Minimum fire area to include (filters small fires)
        fire_id_col: Column name containing fire identifier
        
    Returns:
        Consolidated GeoDataFrame with all snapshots
    """
    snapshot_dir = Path(snapshot_dir)
    all_snapshots = []
    
    for year in years:
        year_dir = snapshot_dir / f"{year}_Snapshot"
        if not year_dir.exists():
            print(f"Warning: Directory not found: {year_dir}")
            continue
            
        gpkg_files = sorted(year_dir.glob("*.gpkg"))
        print(f"Loading {len(gpkg_files)} snapshots from {year}...")
        
        for filepath in tqdm(gpkg_files, desc=f"Year {year}"):
            gdf = load_single_snapshot(filepath)
            if gdf is not None and not gdf.empty:
                all_snapshots.append(gdf)
    
    if not all_snapshots:
        raise ValueError("No valid snapshots found!")
    
    # Concatenate all snapshots
    combined = gpd.GeoDataFrame(pd.concat(all_snapshots, ignore_index=True))
    
    # Ensure geometry column is set
    if 'geometry' in combined.columns:
        combined = combined.set_geometry('geometry')
    
    # Ensure CRS consistency
    combined = combined.to_crs("EPSG:4326")
    
    print(f"Loaded {len(combined)} total snapshot records")
    return combined


def compute_fire_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute geometric metrics for each fire perimeter.
    
    Args:
        gdf: GeoDataFrame with fire perimeters
        
    Returns:
        GeoDataFrame with added columns:
            - area_km2: Polygon area in km²
            - perimeter_km: Perimeter length in km
            - centroid_lat: Centroid latitude
            - centroid_lon: Centroid longitude
            - compactness: Isoperimetric quotient (4π × area / perimeter²)
    """
    # Project to equal-area CRS for accurate measurements (CA Albers)
    gdf_projected = gdf.to_crs("ESRI:102003")
    
    # Compute area in km²
    gdf['area_km2'] = gdf_projected.geometry.area / 1e6
    
    # Compute perimeter in km
    gdf['perimeter_km'] = gdf_projected.geometry.length / 1e3
    
    # Compute centroids (in original WGS84)
    centroids = gdf.geometry.centroid
    gdf['centroid_lon'] = centroids.x
    gdf['centroid_lat'] = centroids.y
    
    # Compute compactness (isoperimetric quotient)
    # Ranges from 0 to 1, where 1 is a perfect circle
    gdf['compactness'] = (4 * np.pi * gdf['area_km2']) / (gdf['perimeter_km'] ** 2)
    gdf['compactness'] = gdf['compactness'].clip(0, 1)
    
    return gdf


def create_snapshot_timeseries(
    gdf: gpd.GeoDataFrame,
    fire_id_col: str = 'fireID'
) -> pd.DataFrame:
    """
    Convert fire snapshots into a time series dataframe with derived targets.
    
    This function:
    1. Groups snapshots by fire ID
    2. Sorts by timestamp
    3. Computes time-varying targets (area_change, spread_rate)
    4. Assigns sequential time indices
    
    Args:
        gdf: GeoDataFrame with fire metrics computed
        fire_id_col: Column containing fire identifier
        
    Returns:
        DataFrame with columns:
            - fire_uid: Unique fire identifier
            - time_idx: Sequential timestep index (0, 1, 2, ...)
            - snapshot_key: Original snapshot key for joining
            - timestamp: Full datetime
            - hour_of_day: 0=AM, 1=PM
            - day_of_year: Julian day
            - area_km2: Current area
            - area_change: Target - area difference from previous timestep
            - spread_rate: Area change normalized by time
            - perimeter_km: Current perimeter
            - perimeter_change: Perimeter difference
            - centroid_lat, centroid_lon: Fire centroid
            - compactness: Shape compactness
    """
    # Create unique fire identifier
    if fire_id_col in gdf.columns:
        gdf['fire_uid'] = gdf[fire_id_col].astype(str)
    else:
        # Try to extract from other columns
        if 'fire_uid' not in gdf.columns:
            raise ValueError(f"Fire ID column '{fire_id_col}' not found. "
                           f"Available columns: {gdf.columns.tolist()}")
    
    # Create full timestamp from date + period
    def make_timestamp(row):
        hour = 6 if row['snapshot_period'] == 'AM' else 18
        return row['snapshot_date'].replace(hour=hour)
    
    gdf['timestamp'] = gdf.apply(make_timestamp, axis=1)
    
    # Sort by fire and timestamp
    gdf = gdf.sort_values(['fire_uid', 'timestamp'])
    
    # Process each fire separately
    timeseries_records = []
    
    for fire_uid, fire_group in tqdm(gdf.groupby('fire_uid'), desc="Building time series"):
        fire_group = fire_group.sort_values('timestamp').reset_index(drop=True)
        
        for i, row in fire_group.iterrows():
            record = {
                'fire_uid': fire_uid,
                'time_idx': i,
                'snapshot_key': row['snapshot_key'],
                'timestamp': row['timestamp'],
                'hour_of_day': 0 if row['snapshot_period'] == 'AM' else 1,
                'day_of_year': row['timestamp'].timetuple().tm_yday,
                'month': row['timestamp'].month,
                'year': row['timestamp'].year,
                'area_km2': row['area_km2'],
                'perimeter_km': row['perimeter_km'],
                'centroid_lat': row['centroid_lat'],
                'centroid_lon': row['centroid_lon'],
                'compactness': row['compactness'],
            }
            
            # Compute changes from previous timestep
            if i > 0:
                prev_row = fire_group.iloc[i - 1]
                record['area_change'] = row['area_km2'] - prev_row['area_km2']
                record['perimeter_change'] = row['perimeter_km'] - prev_row['perimeter_km']
                
                # Time difference in hours (should be ~12 for consecutive)
                time_diff_hours = (row['timestamp'] - prev_row['timestamp']).total_seconds() / 3600
                if time_diff_hours > 0:
                    record['spread_rate'] = record['area_change'] / (time_diff_hours / 12)  # per 12hr
                else:
                    record['spread_rate'] = 0.0
            else:
                # First observation - no change
                record['area_change'] = 0.0
                record['perimeter_change'] = 0.0
                record['spread_rate'] = 0.0
            
            timeseries_records.append(record)
    
    df = pd.DataFrame(timeseries_records)
    
    # Add derived time features
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    
    print(f"Created time series with {len(df)} records for {df['fire_uid'].nunique()} fires")
    return df


def get_fire_date_range(gdf: gpd.GeoDataFrame, fire_id_col: str = 'fireID') -> pd.DataFrame:
    """
    Get the date range for each fire.
    
    Args:
        gdf: GeoDataFrame with snapshots
        fire_id_col: Fire ID column name
        
    Returns:
        DataFrame with fire_uid, start_date, end_date, duration_days
    """
    ranges = []
    
    for fire_id, group in gdf.groupby(fire_id_col):
        dates = pd.to_datetime(group['snapshot_date'])
        ranges.append({
            'fire_uid': str(fire_id),
            'start_date': dates.min(),
            'end_date': dates.max(),
            'duration_days': (dates.max() - dates.min()).days + 1,
            'n_snapshots': len(group)
        })
    
    return pd.DataFrame(ranges)
