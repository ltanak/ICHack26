#!/usr/bin/env python
"""
Example: Build TFT Dataset for Wildfire Spread Prediction

This script demonstrates how to use the tft_data_pipeline module to create
a Temporal Fusion Transformer-ready dataset from raw wildfire data.

Usage:
    python build_tft_dataset.py

Output:
    - tft_wildfire_dataset.parquet: Complete TFT-ready dataset
    - tft_training_config.json: PyTorch Forecasting configuration
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tft_data_pipeline import build_tft_dataset, get_tft_schema
from tft_data_pipeline.pipeline import create_pytorch_forecasting_dataset
import json


def main():
    """Build TFT dataset from raw wildfire data."""
    
    # Define data paths (relative to backend directory)
    base_dir = Path(__file__).parent.parent
    
    config = {
        'weather_file': base_dir / 'Datasets/Weather/hrrr_weather_corrected.parquet',
        'snapshot_dir': base_dir / 'Datasets/Snapshot',
        'perimeter_file': base_dir / 'Datasets/Snapshot/Finalperimeter_2012-2020.gpkg',
        'vegetation_dir': base_dir / 'Datasets/vegetation',
        'output_file': base_dir / 'output/tft_wildfire_dataset.parquet',
        'years': [2017, 2018, 2019, 2020],
        'min_fire_area_km2': 1.0,
        'min_timesteps': 4,
        'normalize': True,
        'handle_missing': True,
    }
    
    print("Building TFT Dataset...")
    print(f"Weather file: {config['weather_file']}")
    print(f"Snapshot dir: {config['snapshot_dir']}")
    print(f"Years: {config['years']}")
    print()
    
    # Build dataset
    df = build_tft_dataset(**config)
    
    # Get PyTorch Forecasting configuration
    tft_config = create_pytorch_forecasting_dataset(
        df,
        max_encoder_length=24,  # 12 days lookback (24 * 12hr = 12 days)
        max_prediction_length=4  # 2 days forecast (4 * 12hr = 2 days)
    )
    
    # Save configuration
    config_path = base_dir / 'output/tft_training_config.json'
    with open(config_path, 'w') as f:
        json.dump(tft_config, f, indent=2)
    print(f"\nSaved TFT config to {config_path}")
    
    # Print schema summary
    schema = get_tft_schema()
    print("\n" + "=" * 60)
    print("TFT SCHEMA SUMMARY")
    print("=" * 60)
    
    print("\n📌 Group Identifier:")
    print(f"   {schema['group_id']}")
    
    print("\n📌 Time Index:")
    print(f"   {schema['time_idx']}")
    
    print("\n🎯 Target Variables:")
    for t in schema['target']:
        if t in df.columns:
            print(f"   ✓ {t}")
    
    print("\n📊 Static Categorical Features:")
    for feat in schema['static_categoricals']:
        status = "✓" if feat in df.columns else "✗"
        print(f"   {status} {feat}")
    
    print("\n📊 Static Real Features:")
    for feat in schema['static_reals']:
        status = "✓" if feat in df.columns else "✗"
        print(f"   {status} {feat}")
    
    print("\n⏰ Time-Varying Known Categorical:")
    for feat in schema['time_varying_known_categoricals']:
        status = "✓" if feat in df.columns else "✗"
        print(f"   {status} {feat}")
    
    print("\n⏰ Time-Varying Known Real:")
    for feat in schema['time_varying_known_reals']:
        status = "✓" if feat in df.columns else "✗"
        print(f"   {status} {feat}")
    
    print("\n🌡️ Time-Varying Unknown (Observed):")
    for feat in schema['time_varying_unknown_reals']:
        status = "✓" if feat in df.columns else "✗"
        print(f"   {status} {feat}")
    
    # Show sample data
    print("\n" + "=" * 60)
    print("SAMPLE DATA (first 5 rows)")
    print("=" * 60)
    print(df.head().to_string())
    
    # Show data statistics
    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    print(f"Total records: {len(df):,}")
    print(f"Unique fires: {df['fire_uid'].nunique():,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Mean timesteps per fire: {df.groupby('fire_uid').size().mean():.1f}")
    print(f"Max timesteps per fire: {df.groupby('fire_uid').size().max()}")
    
    if 'area_change' in df.columns:
        print(f"\nTarget (area_change) statistics:")
        print(f"  Mean: {df['area_change'].mean():.2f} km²")
        print(f"  Std:  {df['area_change'].std():.2f} km²")
        print(f"  Min:  {df['area_change'].min():.2f} km²")
        print(f"  Max:  {df['area_change'].max():.2f} km²")
    
    print("\n✅ TFT dataset ready for training!")
    print(f"   Output file: {config['output_file']}")
    
    return df


if __name__ == '__main__':
    main()
