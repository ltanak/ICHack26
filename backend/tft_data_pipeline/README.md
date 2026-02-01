# Temporal Fusion Transformer (TFT) Data Pipeline for Wildfire Spread Prediction

## Overview

This module transforms raw wildfire datasets into a TFT-ready long-format dataframe compatible with PyTorch Forecasting's `TemporalFusionTransformer`.

---

## Data Sources

| Dataset | Type | Cadence | Source File |
|---------|------|---------|-------------|
| Weather | Time-varying, gridded | 12-hourly | `hrrr_weather_corrected.parquet` |
| Fire Snapshots | Time-varying target | 12-hourly | `{year}_Snapshot/{YYYYMMDD}{AM\|PM}.gpkg` |
| Final Perimeters | Static | - | `Finalperimeter_2012-2020.gpkg` |
| Vegetation | Static, plot-based | - | `CA_*.csv` files |

---

## TFT Variable Schema

### Target Variables (derived from fire snapshots)
| Variable | Description | Unit |
|----------|-------------|------|
| `area_km2` | Fire perimeter area at timestep t | km² |
| `area_change` | **Primary target** - area difference from previous timestep | km² |
| `spread_rate` | Area change per 12-hour period | km²/12hr |
| `perimeter_km` | Fire perimeter length | km |

### Static Covariates (time-invariant per fire)

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `fire_uid` | Categorical | Unique fire identifier | Final perimeters |
| `start_year` | Categorical | Year fire started | Final perimeters |
| `start_month` | Categorical | Month fire started | Final perimeters |
| `final_area_km2` | Real | Final fire perimeter area | Final perimeters |
| `fire_duration_days` | Real | Total fire duration | Final perimeters |
| `vegetation_density` | Real | Mean vegetation density (0-100) | Vegetation |
| `litter_cover_pct` | Real | Litter ground cover % | Vegetation |
| `forest_type_code` | Categorical | Dominant FIA forest type | Vegetation |
| `slope_pct` | Real | Mean terrain slope | Vegetation |
| `elevation_ft` | Real | Mean elevation | Vegetation |
| `fire_size_category` | Categorical | NWCG size class (A-G) | Derived |
| `fire_region` | Categorical | CA region (N/C/S) | Derived |

### Time-Varying Known Inputs (future-known at prediction time)
| Variable | Type | Description |
|----------|------|-------------|
| `time_idx` | Real | Sequential timestep index (0, 1, 2, ...) |
| `hour_of_day` | Categorical | 0=AM (06:00), 1=PM (18:00) |
| `day_of_year` | Real | Julian day (1-365) |
| `month` | Categorical | Month of year (1-12) |
| `is_weekend` | Categorical | Weekend indicator (0/1) |

### Time-Varying Observed Inputs (not known in future)
| Variable | Description | Unit | Source |
|----------|-------------|------|--------|
| `wind_speed` | Wind speed at 10m | m/s | HRRR |
| `wind_direction` | Wind direction | degrees | HRRR |
| `temperature` | Temperature at 2m | °C | HRRR |
| `humidity` | Relative humidity | % | HRRR |
| `precipitation` | Accumulated precipitation | mm | HRRR |
| `fire_weather_index` | Simplified FWI (0-100) | - | Derived |
| `area_km2` | Current fire area | km² | Snapshots |
| `perimeter_km` | Current perimeter | km | Snapshots |
| `compactness` | Shape compactness (0-1) | - | Derived |

---

## Join Strategy

### 1. Fire Snapshots → Time Series
```
Primary Key: (fire_uid, snapshot_key)
- snapshot_key = YYYYMMDD{AM|PM} (e.g., "20170815AM")
- Compute area, perimeter from polygon geometry
- Sort by timestamp, assign sequential time_idx
```

### 2. Weather → Fire Time Series
```
Join: fire_uid + snapshot_key → snapshot_key
Aggregation: Multiple grid points per fire
- Mean for wind_speed, temperature, humidity, precipitation
- Circular mean for wind_direction
- Add spatial std dev as uncertainty features
```

### 3. Vegetation → Fire (Static)
```
Spatial Join: K-nearest neighbor (K=5) within 10km
- Match plot locations to fire centroids
- Aggregate by mean (numeric) or mode (categorical)
```

### 4. Final Perimeters → Fire (Static)
```
Join: fire_uid
- Extract final_area, duration, centroid
```

---

## Final Tabular Schema

```
┌─────────────┬─────────────────────────────────────────────────────────────┐
│ Column Type │ Columns                                                      │
├─────────────┼─────────────────────────────────────────────────────────────┤
│ Group ID    │ fire_uid                                                     │
│ Time Index  │ time_idx                                                     │
│ Timestamp   │ timestamp                                                    │
├─────────────┼─────────────────────────────────────────────────────────────┤
│ Static Cat  │ start_year, start_month, fire_size_category, fire_region,   │
│             │ forest_type_code                                             │
├─────────────┼─────────────────────────────────────────────────────────────┤
│ Static Real │ final_area_km2, fire_duration_days, vegetation_density,     │
│             │ litter_cover_pct, slope_pct, elevation_ft,                   │
│             │ fire_centroid_lat, fire_centroid_lon                         │
├─────────────┼─────────────────────────────────────────────────────────────┤
│ Known Cat   │ hour_of_day, month, is_weekend                               │
├─────────────┼─────────────────────────────────────────────────────────────┤
│ Known Real  │ day_of_year, time_idx                                        │
├─────────────┼─────────────────────────────────────────────────────────────┤
│ Unknown Real│ wind_speed, wind_direction, temperature, humidity,          │
│             │ precipitation, fire_weather_index, area_km2, perimeter_km,  │
│             │ compactness                                                  │
├─────────────┼─────────────────────────────────────────────────────────────┤
│ Target      │ area_change, spread_rate                                     │
└─────────────┴─────────────────────────────────────────────────────────────┘
```

---

## Handling Special Cases

### Multiple Grid Points per Fire
- **Problem**: HRRR weather is gridded at ~3km; fires span multiple cells
- **Solution**: Aggregate by mean (with circular mean for wind direction)
- **Extras**: Include spatial std dev as uncertainty features

### Missing Timesteps
- **Detection**: Gaps in 12-hour sequence
- **Small gaps (< 24hr)**: Linear interpolation
- **Large gaps**: Forward-fill with decay
- **Edge cases**: Backward-fill for initial NaN

### Normalization and Scaling
- **Method**: RobustScaler (handles outliers from extreme fire events)
- **Scope**: Per-column across entire dataset
- **Excluded**: Categorical columns, target (handled by TFT's GroupNormalizer)

---

## Quick Start

### 1. Build Dataset
```python
from tft_data_pipeline import build_tft_dataset

df = build_tft_dataset(
    weather_file="Datasets/Weather/hrrr_weather_corrected.parquet",
    snapshot_dir="Datasets/Snapshot",
    perimeter_file="Datasets/Snapshot/Finalperimeter_2012-2020.gpkg",
    vegetation_dir="Datasets/vegetation",
    years=[2017, 2018, 2019, 2020],
    output_file="output/tft_wildfire_dataset.parquet"
)
```

### 2. Create PyTorch Forecasting Dataset
```python
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

training = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="area_change",
    group_ids=["fire_uid"],
    
    # Static features
    static_categoricals=["start_year", "start_month", "fire_region", "forest_type_code"],
    static_reals=["final_area_km2", "vegetation_density", "litter_cover_pct", 
                  "slope_pct", "elevation_ft"],
    
    # Time-varying known (calendar features)
    time_varying_known_categoricals=["hour_of_day", "month", "is_weekend"],
    time_varying_known_reals=["day_of_year"],
    
    # Time-varying unknown (weather + fire state)
    time_varying_unknown_reals=[
        "wind_speed", "wind_direction", "temperature", "humidity",
        "precipitation", "fire_weather_index", "area_km2", "spread_rate"
    ],
    
    # Sequence lengths
    max_encoder_length=24,    # 12 days lookback (24 × 12hr)
    max_prediction_length=4,  # 2 days forecast (4 × 12hr)
    
    # Normalization
    target_normalizer=GroupNormalizer(groups=["fire_uid"], transformation="softplus"),
    
    # Allow gaps in time series
    allow_missing_timesteps=True,
)

# Create dataloaders
train_dataloader = training.to_dataloader(train=True, batch_size=64)

# Build and train model
tft = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    learning_rate=0.001,
)
```

### 3. Run Complete Pipeline
```bash
# Build dataset
python -m tft_data_pipeline.build_tft_dataset

# Train model
python -m tft_data_pipeline.train_tft_model --data output/tft_wildfire_dataset.parquet
```

---

## Module Structure

```
tft_data_pipeline/
├── __init__.py              # Public API
├── fire_snapshots.py        # Fire perimeter processing
├── weather_features.py      # HRRR weather aggregation
├── vegetation_features.py   # FIA vegetation spatial join
├── static_features.py       # Final perimeter features
├── pipeline.py              # Main orchestration
├── build_tft_dataset.py     # Example: build dataset
├── train_tft_model.py       # Example: train TFT
└── README.md                # This documentation
```

---

## Output Format

The final dataframe is in **long format** with one row per (fire, timestep):

```
fire_uid | time_idx | timestamp           | area_km2 | area_change | wind_speed | ...
---------|----------|---------------------|----------|-------------|------------|----
fire_001 | 0        | 2017-08-15 06:00:00 | 10.5     | 0.0         | 5.2        | ...
fire_001 | 1        | 2017-08-15 18:00:00 | 15.2     | 4.7         | 7.1        | ...
fire_001 | 2        | 2017-08-16 06:00:00 | 22.8     | 7.6         | 8.3        | ...
fire_002 | 0        | 2017-09-01 06:00:00 | 5.0      | 0.0         | 3.1        | ...
...
```

---

## Dependencies

```
pandas>=1.3.0
numpy>=1.20.0
geopandas>=0.10.0
scipy>=1.7.0
scikit-learn>=1.0.0
pyarrow>=6.0.0
tqdm>=4.60.0

# For training (optional)
pytorch-forecasting>=0.10.0
pytorch-lightning>=1.5.0
torch>=1.9.0
```
