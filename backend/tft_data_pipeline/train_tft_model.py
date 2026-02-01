#!/usr/bin/env python
"""
Example: Train Temporal Fusion Transformer for Wildfire Spread Prediction

This script demonstrates how to train a TFT model using PyTorch Forecasting
with the prepared wildfire dataset.

Prerequisites:
    pip install pytorch-forecasting pytorch-lightning

Usage:
    python train_tft_model.py --data output/tft_wildfire_dataset.parquet
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# PyTorch imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# PyTorch Forecasting imports
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss, RMSE, MAE
from pytorch_forecasting.data import GroupNormalizer


def load_dataset(data_path: Path, config_path: Path = None) -> pd.DataFrame:
    """Load TFT dataset from parquet file."""
    df = pd.read_parquet(data_path)
    
    # Ensure proper types
    df['fire_uid'] = df['fire_uid'].astype(str)
    df['time_idx'] = df['time_idx'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert categorical columns
    categorical_cols = ['start_year', 'start_month', 'fire_size_category', 
                       'fire_region', 'forest_type_code', 'hour_of_day', 
                       'month', 'is_weekend']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).astype('category')
    
    print(f"Loaded dataset: {len(df)} records, {df['fire_uid'].nunique()} fires")
    return df


def create_dataloaders(
    df: pd.DataFrame,
    max_encoder_length: int = 24,
    max_prediction_length: int = 4,
    batch_size: int = 64,
    train_val_split: float = 0.8
):
    """
    Create training and validation dataloaders.
    
    Args:
        df: TFT-ready DataFrame
        max_encoder_length: Historical timesteps (24 = 12 days at 12hr cadence)
        max_prediction_length: Future timesteps to predict (4 = 2 days)
        batch_size: Batch size for training
        train_val_split: Fraction for training
        
    Returns:
        training_dataloader, validation_dataloader, training_dataset
    """
    # Split by time (use last X% of each fire for validation)
    df = df.sort_values(['fire_uid', 'time_idx'])
    
    # Get cutoff time_idx per fire
    training_cutoff = df.groupby('fire_uid')['time_idx'].transform(
        lambda x: int(x.max() * train_val_split)
    )
    
    # Determine available columns for each category
    static_categoricals = []
    for col in ['start_year', 'start_month', 'fire_size_category', 'fire_region', 'forest_type_code']:
        if col in df.columns and df[col].notna().any():
            static_categoricals.append(col)
    
    static_reals = []
    for col in ['final_area_km2', 'fire_duration_days', 'vegetation_density', 
                'litter_cover_pct', 'slope_pct', 'elevation_ft']:
        if col in df.columns and df[col].notna().any():
            static_reals.append(col)
    
    time_varying_known_categoricals = []
    for col in ['hour_of_day', 'month', 'is_weekend']:
        if col in df.columns:
            time_varying_known_categoricals.append(col)
    
    time_varying_known_reals = []
    for col in ['day_of_year']:
        if col in df.columns:
            time_varying_known_reals.append(col)
    
    time_varying_unknown_reals = []
    for col in ['wind_speed', 'wind_direction', 'temperature', 'humidity', 
                'precipitation', 'fire_weather_index', 'area_km2', 
                'perimeter_km', 'compactness']:
        if col in df.columns and df[col].notna().any():
            time_varying_unknown_reals.append(col)
    
    # Select target
    target = 'area_change' if 'area_change' in df.columns else 'spread_rate'
    
    # Create training dataset
    training_data = df[df['time_idx'] <= training_cutoff]
    
    # Ensure we have enough data
    min_samples_per_group = max_encoder_length + max_prediction_length
    fire_counts = training_data.groupby('fire_uid').size()
    valid_fires = fire_counts[fire_counts >= min_samples_per_group].index
    training_data = training_data[training_data['fire_uid'].isin(valid_fires)]
    
    print(f"Training fires with sufficient data: {len(valid_fires)}")
    print(f"Training records: {len(training_data)}")
    
    # Create TimeSeriesDataSet
    training = TimeSeriesDataSet(
        training_data,
        time_idx='time_idx',
        target=target,
        group_ids=['fire_uid'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals + [target],
        target_normalizer=GroupNormalizer(
            groups=['fire_uid'],
            transformation='softplus'
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    
    # Create validation dataset
    validation_data = df[df['time_idx'] > training_cutoff]
    validation_data = validation_data[validation_data['fire_uid'].isin(valid_fires)]
    
    validation = TimeSeriesDataSet.from_dataset(
        training,
        validation_data,
        predict=True,
        stop_randomization=True
    )
    
    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0
    )
    
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size * 2,
        num_workers=0
    )
    
    print(f"Created dataloaders: {len(train_dataloader)} train batches, {len(val_dataloader)} val batches")
    
    return train_dataloader, val_dataloader, training


def create_tft_model(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """
    Create TFT model from training dataset configuration.
    
    Args:
        training_dataset: PyTorch Forecasting TimeSeriesDataSet
        
    Returns:
        Configured TemporalFusionTransformer model
    """
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        # Architecture parameters
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        
        # Output configuration
        output_size=7,  # 7 quantiles for probabilistic forecast
        loss=QuantileLoss(),
        
        # Optimization
        learning_rate=0.001,
        reduce_on_plateau_patience=3,
        
        # Logging
        log_interval=10,
        log_val_interval=1,
    )
    
    print(f"Created TFT model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def train_model(
    model: TemporalFusionTransformer,
    train_dataloader,
    val_dataloader,
    max_epochs: int = 50,
    output_dir: Path = Path('output/tft_model')
):
    """
    Train the TFT model.
    
    Args:
        model: TFT model
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        max_epochs: Maximum training epochs
        output_dir: Directory for checkpoints and logs
        
    Returns:
        Trained model, trainer
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=5,
        verbose=True,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename='tft-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir / 'logs',
        name='tft_wildfire'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
        logger=logger,
        enable_progress_bar=True
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nBest model: {best_model_path}")
    
    return model, trainer


def evaluate_model(model, val_dataloader, training_dataset):
    """Evaluate model and print metrics."""
    
    # Get predictions
    predictions = model.predict(val_dataloader, return_y=True)
    
    # Calculate metrics
    actuals = predictions.y[0]
    preds = predictions.output[:, :, 3]  # Median prediction (quantile 0.5)
    
    rmse = torch.sqrt(torch.mean((actuals - preds) ** 2)).item()
    mae = torch.mean(torch.abs(actuals - preds)).item()
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    
    # Feature importance (from attention weights)
    interpretation = model.interpret_output(predictions.output, reduction='sum')
    
    print("\nFeature Importance (Encoder):")
    for name, importance in sorted(
        interpretation['encoder_variables'].items(),
        key=lambda x: -x[1]
    )[:10]:
        print(f"  {name}: {importance:.4f}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Train TFT for wildfire prediction')
    parser.add_argument('--data', type=str, default='output/tft_wildfire_dataset.parquet',
                       help='Path to TFT dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to TFT config JSON')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--encoder-length', type=int, default=24,
                       help='Encoder length (historical timesteps)')
    parser.add_argument('--prediction-length', type=int, default=4,
                       help='Prediction length (future timesteps)')
    parser.add_argument('--output-dir', type=str, default='output/tft_model',
                       help='Output directory for model')
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Run build_tft_dataset.py first to create the dataset.")
        return
    
    df = load_dataset(data_path)
    
    # Create dataloaders
    train_dl, val_dl, training_dataset = create_dataloaders(
        df,
        max_encoder_length=args.encoder_length,
        max_prediction_length=args.prediction_length,
        batch_size=args.batch_size
    )
    
    # Create model
    model = create_tft_model(training_dataset)
    
    # Train
    model, trainer = train_model(
        model,
        train_dl,
        val_dl,
        max_epochs=args.epochs,
        output_dir=Path(args.output_dir)
    )
    
    # Evaluate
    predictions = evaluate_model(model, val_dl, training_dataset)
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
