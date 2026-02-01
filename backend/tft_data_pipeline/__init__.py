"""
Temporal Fusion Transformer (TFT) Data Pipeline for Wildfire Spread Prediction

This module transforms raw wildfire datasets into a TFT-ready long-format dataframe
compatible with PyTorch Forecasting's TemporalFusionTransformer.

Key Components:
    - Fire snapshot processing with area/perimeter calculations
    - Weather data aggregation (multiple grid points per fire)
    - Vegetation feature extraction via spatial joins
    - Missing timestep handling and interpolation
    - Feature normalization and scaling
"""

from .fire_snapshots import (
    load_fire_snapshots,
    compute_fire_metrics,
    create_snapshot_timeseries,
)

from .weather_features import (
    load_weather_data,
    aggregate_weather_to_fire,
    compute_fire_weather_index,
)

from .vegetation_features import (
    load_vegetation_data,
    create_vegetation_lookup,
    assign_vegetation_features,
)

from .static_features import (
    load_final_perimeters,
    extract_static_features,
)

from .pipeline import (
    build_tft_dataset,
    normalize_features,
    handle_missing_timesteps,
    validate_tft_schema,
)

__all__ = [
    # Fire snapshots
    'load_fire_snapshots',
    'compute_fire_metrics',
    'create_snapshot_timeseries',
    # Weather
    'load_weather_data',
    'aggregate_weather_to_fire',
    'compute_fire_weather_index',
    # Vegetation
    'load_vegetation_data',
    'create_vegetation_lookup',
    'assign_vegetation_features',
    # Static
    'load_final_perimeters',
    'extract_static_features',
    # Pipeline
    'build_tft_dataset',
    'normalize_features',
    'handle_missing_timesteps',
    'validate_tft_schema',
]
