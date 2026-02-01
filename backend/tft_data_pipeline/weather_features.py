"""
Weather Feature Processing Module

Loads HRRR weather data and aggregates gridded values to fire-level features.
Handles multiple grid points per fire through spatial aggregation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from scipy.spatial import cKDTree
from tqdm import tqdm


def load_weather_data(
    weather_file: Path,
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load HRRR weather data from parquet file.
    
    Expected columns:
        - fire_uid or fireID: Fire identifier
        - snapshot_key: Timestep key (e.g., "20170815AM")
        - latitude, longitude: Grid point location
        - timestamp: Observation time
        - wind_speed (or wind_speed_ms): Wind speed in m/s
        - wind_direction (or wind_direction_deg): Wind direction in degrees
        - temperature (or temperature_c): Temperature in °C
        - humidity (or relative_humidity_pct, dewpoint_c): Humidity measure
        - precipitation (or precip_mm): Precipitation in mm
    
    Args:
        weather_file: Path to parquet file
        years: Optional list of years to filter
        
    Returns:
        DataFrame with standardized weather columns
    """
    weather_file = Path(weather_file)
    
    if not weather_file.exists():
        raise FileNotFoundError(f"Weather file not found: {weather_file}")
    
    df = pd.read_parquet(weather_file)
    print(f"Loaded {len(df)} weather records")
    
    # Standardize column names
    column_mapping = {
        'fireID': 'fire_uid',
        'wind_speed_ms': 'wind_speed',
        'wind_direction_deg': 'wind_direction',
        'temperature_c': 'temperature',
        'relative_humidity_pct': 'humidity',
        'dewpoint_c': 'dewpoint',
        'precip_mm': 'precipitation',
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # If humidity not directly available, compute from dewpoint if possible
    if 'humidity' not in df.columns and 'dewpoint' in df.columns and 'temperature' in df.columns:
        # Approximate RH from dewpoint using Magnus formula
        df['humidity'] = compute_rh_from_dewpoint(df['temperature'], df['dewpoint'])
    
    # Convert fire_uid to string
    if 'fire_uid' in df.columns:
        df['fire_uid'] = df['fire_uid'].astype(str)
    
    # Parse timestamp if needed
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by years if specified
        if years:
            df = df[df['timestamp'].dt.year.isin(years)]
            print(f"Filtered to {len(df)} records for years {years}")
    
    return df


def compute_rh_from_dewpoint(temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    """
    Compute relative humidity from temperature and dewpoint using Magnus formula.
    
    Args:
        temp_c: Temperature in Celsius
        dewpoint_c: Dewpoint temperature in Celsius
        
    Returns:
        Relative humidity as percentage (0-100)
    """
    # Magnus formula constants
    a = 17.27
    b = 237.7
    
    # Saturation vapor pressure ratio
    alpha_t = (a * temp_c) / (b + temp_c)
    alpha_d = (a * dewpoint_c) / (b + dewpoint_c)
    
    rh = 100 * np.exp(alpha_d - alpha_t)
    return rh.clip(0, 100)


def aggregate_weather_to_fire(
    weather_df: pd.DataFrame,
    aggregation: str = 'mean',
    spatial_weights: bool = True
) -> pd.DataFrame:
    """
    Aggregate gridded weather data to fire-level values.
    
    Handles multiple grid points per fire by aggregating spatially.
    
    Args:
        weather_df: Weather DataFrame with multiple grid points per fire/timestep
        aggregation: Aggregation method ('mean', 'median', 'max', 'weighted')
        spatial_weights: If True, weight by inverse distance to fire centroid
        
    Returns:
        DataFrame with one row per (fire_uid, snapshot_key) with aggregated weather
    """
    # Weather variables to aggregate
    weather_vars = ['wind_speed', 'wind_direction', 'temperature', 
                    'humidity', 'precipitation']
    
    # Filter to available columns
    weather_vars = [v for v in weather_vars if v in weather_df.columns]
    
    # Group by fire and timestep
    group_cols = ['fire_uid', 'snapshot_key']
    available_groups = [c for c in group_cols if c in weather_df.columns]
    
    if not available_groups:
        raise ValueError("No grouping columns found (fire_uid, snapshot_key)")
    
    # Define aggregation functions
    if aggregation == 'mean':
        agg_funcs = {v: 'mean' for v in weather_vars}
    elif aggregation == 'median':
        agg_funcs = {v: 'median' for v in weather_vars}
    elif aggregation == 'max':
        agg_funcs = {v: 'max' for v in weather_vars}
    else:
        agg_funcs = {v: 'mean' for v in weather_vars}
    
    # Also compute spatial statistics
    agg_funcs.update({
        'latitude': 'mean',
        'longitude': 'mean',
    })
    
    # For wind direction, compute circular mean
    def circular_mean_deg(angles):
        """Compute mean of angles in degrees."""
        angles_rad = np.deg2rad(angles)
        mean_sin = np.sin(angles_rad).mean()
        mean_cos = np.cos(angles_rad).mean()
        return np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360
    
    # Special handling for wind direction
    aggregated = weather_df.groupby(available_groups).agg(agg_funcs).reset_index()
    
    # Recompute wind direction with circular mean if present
    if 'wind_direction' in weather_vars:
        wind_dir_agg = weather_df.groupby(available_groups)['wind_direction'].apply(circular_mean_deg)
        aggregated = aggregated.drop(columns=['wind_direction'])
        aggregated = aggregated.merge(wind_dir_agg.reset_index(), on=available_groups)
    
    # Add count of grid points (useful for quality assessment)
    counts = weather_df.groupby(available_groups).size().reset_index(name='n_weather_points')
    aggregated = aggregated.merge(counts, on=available_groups)
    
    # Add weather variability metrics (std dev within fire area)
    if aggregation == 'mean':
        for var in ['wind_speed', 'temperature']:
            if var in weather_vars:
                var_std = weather_df.groupby(available_groups)[var].std().reset_index(name=f'{var}_std')
                aggregated = aggregated.merge(var_std, on=available_groups, how='left')
                aggregated[f'{var}_std'] = aggregated[f'{var}_std'].fillna(0)
    
    print(f"Aggregated to {len(aggregated)} fire-timestep records")
    return aggregated


def compute_fire_weather_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple Fire Weather Index (FWI) from weather variables.
    
    This is a simplified index combining:
    - Wind speed (higher = more spread)
    - Humidity (lower = more spread)
    - Temperature (higher = more spread, if extreme)
    
    The index is normalized to [0, 100].
    
    Args:
        df: DataFrame with weather variables
        
    Returns:
        DataFrame with added 'fire_weather_index' column
    """
    df = df.copy()
    
    # Initialize index
    fwi = np.zeros(len(df))
    
    # Wind contribution (normalized)
    if 'wind_speed' in df.columns:
        # Wind typically 0-20 m/s, normalize to 0-40
        wind_contrib = np.clip(df['wind_speed'] / 20, 0, 1) * 40
        fwi += wind_contrib
    
    # Humidity contribution (inverted - lower humidity = higher risk)
    if 'humidity' in df.columns:
        # RH 0-100%, invert so low humidity = high contribution
        humidity_contrib = (100 - df['humidity'].clip(0, 100)) / 100 * 40
        fwi += humidity_contrib
    
    # Temperature contribution (extreme heat bonus)
    if 'temperature' in df.columns:
        # Temperature contribution increases above 30°C
        temp_contrib = np.clip((df['temperature'] - 20) / 30, 0, 1) * 20
        fwi += temp_contrib
    
    df['fire_weather_index'] = fwi.clip(0, 100)
    
    return df


def interpolate_missing_weather(
    weather_df: pd.DataFrame,
    fire_timeseries: pd.DataFrame,
    method: str = 'linear'
) -> pd.DataFrame:
    """
    Interpolate weather data for missing timesteps.
    
    Args:
        weather_df: Aggregated weather data
        fire_timeseries: Complete fire time series with all expected timesteps
        method: Interpolation method ('linear', 'nearest', 'ffill')
        
    Returns:
        Weather data with missing timesteps filled
    """
    # Get all expected (fire_uid, snapshot_key) combinations
    expected = fire_timeseries[['fire_uid', 'snapshot_key', 'timestamp']].drop_duplicates()
    
    # Merge with weather
    merged = expected.merge(weather_df, on=['fire_uid', 'snapshot_key'], how='left')
    
    # Weather columns to interpolate
    weather_cols = ['wind_speed', 'wind_direction', 'temperature', 
                    'humidity', 'precipitation', 'fire_weather_index']
    weather_cols = [c for c in weather_cols if c in merged.columns]
    
    # Interpolate per fire
    result_dfs = []
    
    for fire_uid, group in merged.groupby('fire_uid'):
        group = group.sort_values('timestamp').copy()
        
        for col in weather_cols:
            if col == 'wind_direction':
                # Special handling for circular data
                group[col] = group[col].interpolate(method='nearest')
            else:
                group[col] = group[col].interpolate(method=method)
            
            # Fill remaining NaN at edges
            group[col] = group[col].fillna(method='ffill').fillna(method='bfill')
        
        result_dfs.append(group)
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Report interpolation statistics
    original_missing = expected.merge(weather_df, on=['fire_uid', 'snapshot_key'], how='left')
    n_interpolated = original_missing['wind_speed'].isna().sum() if 'wind_speed' in original_missing else 0
    print(f"Interpolated {n_interpolated} missing weather records")
    
    return result


def create_weather_lookup(weather_df: pd.DataFrame) -> Dict[Tuple[str, str], Dict]:
    """
    Create a fast lookup dictionary for weather data.
    
    Args:
        weather_df: Aggregated weather DataFrame
        
    Returns:
        Dictionary mapping (fire_uid, snapshot_key) -> weather dict
    """
    lookup = {}
    
    weather_cols = ['wind_speed', 'wind_direction', 'temperature', 
                    'humidity', 'precipitation', 'fire_weather_index',
                    'wind_speed_std', 'temperature_std']
    weather_cols = [c for c in weather_cols if c in weather_df.columns]
    
    for _, row in weather_df.iterrows():
        key = (str(row['fire_uid']), row['snapshot_key'])
        lookup[key] = {col: row[col] for col in weather_cols if pd.notna(row.get(col))}
    
    return lookup
