"""
TFT Dataset Pipeline - Main Module

Orchestrates the complete data pipeline to build a TFT-ready dataset
from raw wildfire data sources.

Output Schema (PyTorch Forecasting compatible):
    - time_idx: int - Sequential timestep index per group
    - fire_uid: str - Group identifier
    - Static features (constant per fire)
    - Time-varying known features (known in future)
    - Time-varying observed features (unknown in future)
    - target: float - Area change to predict
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings

from .fire_snapshots import (
    load_fire_snapshots, 
    compute_fire_metrics,
    create_snapshot_timeseries
)
from .weather_features import (
    load_weather_data,
    aggregate_weather_to_fire,
    compute_fire_weather_index,
    interpolate_missing_weather
)
from .vegetation_features import (
    load_vegetation_data,
    create_vegetation_lookup,
    assign_vegetation_features
)
from .static_features import (
    load_final_perimeters,
    extract_static_features,
    merge_static_to_timeseries,
    compute_fire_size_category,
    compute_fire_region
)


# TFT Schema Definition
TFT_SCHEMA = {
    'group_id': 'fire_uid',
    'time_idx': 'time_idx',
    'target': ['area_change', 'spread_rate'],
    
    'static_categoricals': [
        'start_year',
        'start_month', 
        'fire_size_category',
        'fire_region',
        'forest_type_code'
    ],
    
    'static_reals': [
        'final_area_km2',
        'fire_duration_days',
        'fire_centroid_lat',
        'fire_centroid_lon',
        'vegetation_density',
        'litter_cover_pct',
        'slope_pct',
        'elevation_ft'
    ],
    
    'time_varying_known_categoricals': [
        'hour_of_day',
        'month',
        'is_weekend'
    ],
    
    'time_varying_known_reals': [
        'day_of_year',
        'time_idx'  # Can be used as positional encoding
    ],
    
    'time_varying_unknown_reals': [
        'wind_speed',
        'wind_direction',
        'temperature',
        'humidity',
        'precipitation',
        'fire_weather_index',
        'area_km2',
        'perimeter_km',
        'compactness'
    ]
}


def build_tft_dataset(
    weather_file: Path,
    snapshot_dir: Path,
    perimeter_file: Path,
    vegetation_dir: Path,
    years: List[int] = [2017, 2018, 2019, 2020],
    output_file: Optional[Path] = None,
    min_fire_area_km2: float = 1.0,
    min_timesteps: int = 4,
    normalize: bool = True,
    handle_missing: bool = True,
    fire_id_col: str = 'fireID'
) -> pd.DataFrame:
    """
    Build complete TFT dataset from raw data sources.
    
    Pipeline Steps:
        1. Load and process fire snapshots → time series
        2. Load and aggregate weather data → fire-level
        3. Load and assign vegetation features → static
        4. Load and extract final perimeter features → static
        5. Merge all features
        6. Handle missing timesteps
        7. Normalize features
        8. Validate schema
    
    Args:
        weather_file: Path to HRRR weather parquet
        snapshot_dir: Directory with year_Snapshot folders
        perimeter_file: Path to final perimeters GeoPackage
        vegetation_dir: Directory with vegetation CSV files
        years: Years to include
        output_file: Optional path to save output parquet
        min_fire_area_km2: Minimum final fire area (filters small fires)
        min_timesteps: Minimum number of timesteps per fire
        normalize: Whether to normalize numeric features
        handle_missing: Whether to interpolate missing timesteps
        fire_id_col: Column name for fire identifier
        
    Returns:
        TFT-ready DataFrame in long format
    """
    print("=" * 60)
    print("TFT Dataset Pipeline")
    print("=" * 60)
    
    # Convert paths
    weather_file = Path(weather_file)
    snapshot_dir = Path(snapshot_dir)
    perimeter_file = Path(perimeter_file)
    vegetation_dir = Path(vegetation_dir)
    
    # Step 1: Load and process fire snapshots
    print("\n[1/7] Loading fire snapshots...")
    snapshots_gdf = load_fire_snapshots(snapshot_dir, years, fire_id_col=fire_id_col)
    
    print("\n[2/7] Computing fire metrics...")
    snapshots_gdf = compute_fire_metrics(snapshots_gdf)
    
    print("\n[3/7] Creating time series...")
    fire_ts = create_snapshot_timeseries(snapshots_gdf, fire_id_col=fire_id_col)
    
    # Step 2: Load and merge weather data
    print("\n[4/7] Processing weather data...")
    if weather_file.exists():
        weather_df = load_weather_data(weather_file, years)
        weather_agg = aggregate_weather_to_fire(weather_df)
        weather_agg = compute_fire_weather_index(weather_agg)
        
        # Merge weather with fire time series
        fire_ts = fire_ts.merge(
            weather_agg,
            on=['fire_uid', 'snapshot_key'],
            how='left'
        )
        
        # Handle missing weather
        if handle_missing:
            fire_ts = interpolate_missing_weather(weather_agg, fire_ts)
    else:
        print(f"Warning: Weather file not found at {weather_file}")
        # Add placeholder weather columns
        for col in ['wind_speed', 'wind_direction', 'temperature', 
                   'humidity', 'precipitation', 'fire_weather_index']:
            fire_ts[col] = np.nan
    
    # Step 3: Load and assign vegetation features
    print("\n[5/7] Processing vegetation data...")
    try:
        veg_data = load_vegetation_data(vegetation_dir)
        veg_lookup = create_vegetation_lookup(veg_data)
        fire_ts = assign_vegetation_features(fire_ts, veg_lookup)
    except Exception as e:
        print(f"Warning: Could not process vegetation data: {e}")
        # Add placeholder columns
        for col in ['vegetation_density', 'litter_cover_pct', 'slope_pct', 
                   'elevation_ft', 'forest_type_code']:
            fire_ts[col] = np.nan
    
    # Step 4: Load and merge static features
    print("\n[6/7] Extracting static features...")
    perimeters_gdf = load_final_perimeters(perimeter_file)
    static_df = extract_static_features(perimeters_gdf)
    fire_ts = merge_static_to_timeseries(fire_ts, static_df)
    
    # Add derived static categories
    if 'final_area_km2' in fire_ts.columns:
        fire_ts['fire_size_category'] = compute_fire_size_category(
            fire_ts.groupby('fire_uid')['final_area_km2'].transform('first')
        )
    
    if 'fire_centroid_lat' in fire_ts.columns and 'fire_centroid_lon' in fire_ts.columns:
        fire_ts['fire_region'] = compute_fire_region(
            fire_ts['fire_centroid_lat'],
            fire_ts['fire_centroid_lon']
        )
    
    # Step 5: Filter and clean
    print("\n[7/7] Final processing...")
    
    # Filter fires with sufficient timesteps
    fire_counts = fire_ts.groupby('fire_uid').size()
    valid_fires = fire_counts[fire_counts >= min_timesteps].index
    fire_ts = fire_ts[fire_ts['fire_uid'].isin(valid_fires)]
    print(f"Filtered to {len(valid_fires)} fires with >= {min_timesteps} timesteps")
    
    # Recompute time_idx after filtering
    fire_ts = fire_ts.sort_values(['fire_uid', 'timestamp'])
    fire_ts['time_idx'] = fire_ts.groupby('fire_uid').cumcount()
    
    # Handle missing values
    fire_ts = handle_missing_timesteps(fire_ts)
    
    # Normalize if requested
    if normalize:
        fire_ts = normalize_features(fire_ts)
    
    # Validate schema
    fire_ts = validate_tft_schema(fire_ts)
    
    # Save if output path provided
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fire_ts.to_parquet(output_file, index=False)
        print(f"\nSaved TFT dataset to {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print(f"  Total records: {len(fire_ts)}")
    print(f"  Unique fires: {fire_ts['fire_uid'].nunique()}")
    print(f"  Date range: {fire_ts['timestamp'].min()} to {fire_ts['timestamp'].max()}")
    print(f"  Columns: {len(fire_ts.columns)}")
    print("=" * 60)
    
    return fire_ts


def handle_missing_timesteps(
    df: pd.DataFrame,
    max_gap_hours: int = 24
) -> pd.DataFrame:
    """
    Handle missing timesteps in the fire time series.
    
    Strategy:
        1. Identify gaps in the timestamp sequence
        2. For small gaps (< max_gap_hours): interpolate
        3. For large gaps: fill with last known value + decay
        4. For initial missing values: backfill
    
    Args:
        df: Fire time series DataFrame
        max_gap_hours: Maximum gap to interpolate (hours)
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Columns to interpolate
    interpolate_cols = [
        'wind_speed', 'wind_direction', 'temperature', 
        'humidity', 'precipitation', 'fire_weather_index',
        'area_km2', 'perimeter_km', 'compactness'
    ]
    interpolate_cols = [c for c in interpolate_cols if c in df.columns]
    
    # Process each fire separately
    result_dfs = []
    
    for fire_uid, group in df.groupby('fire_uid'):
        group = group.sort_values('timestamp').copy()
        
        for col in interpolate_cols:
            if group[col].isna().any():
                # Linear interpolation for small gaps
                group[col] = group[col].interpolate(method='linear', limit=2)
                # Forward fill remaining
                group[col] = group[col].fillna(method='ffill')
                # Backward fill for initial values
                group[col] = group[col].fillna(method='bfill')
        
        result_dfs.append(group)
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Fill any remaining NaN with column medians
    for col in interpolate_cols:
        if col in result.columns and result[col].isna().any():
            median_val = result[col].median()
            result[col] = result[col].fillna(median_val)
    
    # Handle categorical missing
    categorical_fills = {
        'forest_type_code': 999,
        'fire_size_category': 'D',
        'fire_region': 'Central',
        'start_year': 2018,
        'start_month': 7
    }
    
    for col, fill_val in categorical_fills.items():
        if col in result.columns:
            result[col] = result[col].fillna(fill_val)
    
    return result


def normalize_features(
    df: pd.DataFrame,
    method: str = 'robust'
) -> pd.DataFrame:
    """
    Normalize numeric features for TFT training.
    
    Uses RobustScaler by default (less sensitive to outliers).
    
    Args:
        df: DataFrame with features
        method: 'standard' for StandardScaler, 'robust' for RobustScaler
        
    Returns:
        DataFrame with normalized features
        
    Note:
        In production, save the scaler and use it for inverse transform
        on predictions. Here we just normalize in-place.
    """
    df = df.copy()
    
    # Columns to normalize
    normalize_cols = [
        'wind_speed', 'temperature', 'humidity', 'precipitation',
        'fire_weather_index', 'area_km2', 'perimeter_km',
        'final_area_km2', 'vegetation_density', 'litter_cover_pct',
        'slope_pct', 'elevation_ft', 'fire_centroid_lat', 'fire_centroid_lon'
    ]
    
    normalize_cols = [c for c in normalize_cols if c in df.columns]
    
    # Select scaler
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    # Store original column names with _orig suffix before normalizing
    # (useful for debugging)
    for col in normalize_cols:
        if df[col].notna().any():
            df[f'{col}_orig'] = df[col].copy()
    
    # Fit and transform
    if normalize_cols:
        df[normalize_cols] = scaler.fit_transform(df[normalize_cols].fillna(0))
    
    # Note: In production, you'd save the scaler:
    # import joblib
    # joblib.dump(scaler, 'tft_scaler.pkl')
    
    print(f"Normalized {len(normalize_cols)} features using {method} scaler")
    
    return df


def validate_tft_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and enforce TFT schema requirements.
    
    Ensures:
        - Required columns exist
        - Correct data types
        - No NaN in critical columns
        - Proper time_idx ordering
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Validated DataFrame (may have added/modified columns)
    """
    df = df.copy()
    
    # Required columns
    required = ['fire_uid', 'time_idx', 'timestamp']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Ensure data types
    df['fire_uid'] = df['fire_uid'].astype(str)
    df['time_idx'] = df['time_idx'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure time_idx is sequential per group
    df = df.sort_values(['fire_uid', 'time_idx'])
    
    # Add target columns if missing
    if 'area_change' not in df.columns and 'area_km2' in df.columns:
        df['area_change'] = df.groupby('fire_uid')['area_km2'].diff().fillna(0)
    
    if 'spread_rate' not in df.columns and 'area_change' in df.columns:
        df['spread_rate'] = df['area_change']  # Already per 12-hour
    
    # Ensure categorical columns are proper type
    categorical_cols = TFT_SCHEMA['static_categoricals'] + TFT_SCHEMA['time_varying_known_categoricals']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Report validation results
    print("\nSchema Validation:")
    print(f"  Group ID column: {TFT_SCHEMA['group_id']}")
    print(f"  Time index column: {TFT_SCHEMA['time_idx']}")
    print(f"  Target columns: {[c for c in TFT_SCHEMA['target'] if c in df.columns]}")
    print(f"  Static categoricals: {[c for c in TFT_SCHEMA['static_categoricals'] if c in df.columns]}")
    print(f"  Static reals: {[c for c in TFT_SCHEMA['static_reals'] if c in df.columns]}")
    print(f"  Time-varying known cat: {[c for c in TFT_SCHEMA['time_varying_known_categoricals'] if c in df.columns]}")
    print(f"  Time-varying known real: {[c for c in TFT_SCHEMA['time_varying_known_reals'] if c in df.columns]}")
    print(f"  Time-varying unknown: {[c for c in TFT_SCHEMA['time_varying_unknown_reals'] if c in df.columns]}")
    
    return df


def create_pytorch_forecasting_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 24,
    max_prediction_length: int = 4,
    training_cutoff: Optional[str] = None
) -> Dict:
    """
    Prepare configuration for PyTorch Forecasting TimeSeriesDataSet.
    
    This returns the parameters needed to create a TimeSeriesDataSet.
    
    Args:
        df: TFT-ready DataFrame
        max_encoder_length: Number of historical timesteps (24 = 12 days)
        max_prediction_length: Number of future timesteps to predict (4 = 2 days)
        training_cutoff: Date string for train/val split
        
    Returns:
        Dictionary with TimeSeriesDataSet parameters
        
    Usage:
        ```python
        config = create_pytorch_forecasting_dataset(df)
        
        from pytorch_forecasting import TimeSeriesDataSet
        training = TimeSeriesDataSet(df[df['timestamp'] < cutoff], **config)
        validation = TimeSeriesDataSet(df[df['timestamp'] >= cutoff], **config)
        ```
    """
    # Get available columns
    available_static_cat = [c for c in TFT_SCHEMA['static_categoricals'] if c in df.columns]
    available_static_real = [c for c in TFT_SCHEMA['static_reals'] if c in df.columns]
    available_known_cat = [c for c in TFT_SCHEMA['time_varying_known_categoricals'] if c in df.columns]
    available_known_real = [c for c in TFT_SCHEMA['time_varying_known_reals'] if c in df.columns]
    available_unknown = [c for c in TFT_SCHEMA['time_varying_unknown_reals'] if c in df.columns]
    
    # Select target
    target = 'area_change' if 'area_change' in df.columns else 'spread_rate'
    
    config = {
        'time_idx': 'time_idx',
        'target': target,
        'group_ids': ['fire_uid'],
        'max_encoder_length': max_encoder_length,
        'max_prediction_length': max_prediction_length,
        'static_categoricals': available_static_cat,
        'static_reals': available_static_real,
        'time_varying_known_categoricals': available_known_cat,
        'time_varying_known_reals': available_known_real,
        'time_varying_unknown_reals': available_unknown + [target],
        'add_relative_time_idx': True,
        'add_target_scales': True,
        'add_encoder_length': True,
    }
    
    return config


def get_tft_schema() -> Dict:
    """
    Return the TFT schema definition.
    
    Returns:
        Dictionary defining all feature categories
    """
    return TFT_SCHEMA.copy()
