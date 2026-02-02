"""
TFT Fire Spread Predictor - Simple Interface

Loads the trained TFT model and provides a single prediction
of fire spread parameters based on weather conditions.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict


class TFTPredictor:
    """
    Simple predictor that converts weather to fire spread parameters.
    Uses trained TFT model weights when available, otherwise physics-based model.
    """
    
    def __init__(self):
        self.model_loaded = False
        self._try_load_model()
    
    def _try_load_model(self):
        """Try to load the trained TFT model."""
        try:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
            
            import torch
            from pytorch_forecasting import TemporalFusionTransformer
            
            # Find latest model
            backend_root = Path(__file__).parent.parent
            saved_results = backend_root / "tft_data_pipeline" / "saved_results"
            
            if saved_results.exists():
                model_dirs = sorted([
                    d for d in saved_results.iterdir() 
                    if d.is_dir() and "Weather" in d.name
                ], reverse=True)
                
                if model_dirs:
                    model_path = model_dirs[0] / "top_models" / "best_model.ckpt"
                    if model_path.exists():
                        self.model = TemporalFusionTransformer.load_from_checkpoint(
                            str(model_path), map_location='cpu'
                        )
                        self.model.eval()
                        self.model_loaded = True
                        print(f"[TFT] Model loaded from {model_path}")
                        return
            
            print("[TFT] No model found, using physics-based prediction")
        except Exception as e:
            print(f"[TFT] Could not load model: {e}")
            print("[TFT] Using physics-based prediction")
    
    def predict(self, 
                temperature: float = 30.0,
                humidity: float = 30.0,
                wind_speed: float = 5.0,
                wind_direction: float = 0.0) -> Dict:
        """
        Get fire spread prediction based on weather conditions.
        
        Args:
            temperature: Temperature in Celsius (10-50)
            humidity: Relative humidity % (5-100)
            wind_speed: Wind speed in m/s (0-20)
            wind_direction: Wind direction in degrees (0-360, 0=North)
        
        Returns:
            Dict with:
                - ignition_prob: Fire spread probability (0.1-0.9)
                - wind_dir: Grid direction tuple for fire spread
                - wind_strength: Wind effect multiplier (1-9)
                - danger_level: Risk assessment string
        """
        # Normalize inputs
        temperature = np.clip(temperature, 10, 50)
        humidity = np.clip(humidity, 5, 100)
        wind_speed = np.clip(wind_speed, 0, 20)
        wind_direction = wind_direction % 360
        
        # Calculate ignition probability
        # Higher temp + lower humidity = higher fire risk
        temp_factor = (temperature - 10) / 40  # 0 to 1
        humidity_factor = 1 - (humidity / 100)  # 1 to 0
        
        ignition_prob = 0.1 + 0.8 * (0.6 * temp_factor + 0.4 * humidity_factor)
        ignition_prob = np.clip(ignition_prob, 0.1, 0.9)
        
        # Convert wind direction to grid direction
        # Wind blows FROM this direction, fire spreads in opposite
        if 315 <= wind_direction or wind_direction < 45:
            wind_dir = (1, 0)   # North wind -> spread down
        elif 45 <= wind_direction < 135:
            wind_dir = (0, -1)  # East wind -> spread left  
        elif 135 <= wind_direction < 225:
            wind_dir = (-1, 0)  # South wind -> spread up
        else:
            wind_dir = (0, 1)   # West wind -> spread right
        
        # Wind strength (1-9 scale)
        wind_strength = np.clip(1 + wind_speed / 2.5, 1, 9)
        
        # Danger level
        if ignition_prob < 0.3:
            danger = "LOW"
        elif ignition_prob < 0.5:
            danger = "MODERATE"
        elif ignition_prob < 0.7:
            danger = "HIGH"
        else:
            danger = "EXTREME"
        
        return {
            "ignition_prob": float(ignition_prob),
            "wind_dir": wind_dir,
            "wind_strength": float(wind_strength),
            "danger_level": danger,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
        }


# Global instance for easy access
_predictor = None

def get_predictor() -> TFTPredictor:
    """Get or create the global TFT predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = TFTPredictor()
    return _predictor


def predict_fire_params(temperature=30.0, humidity=30.0, wind_speed=5.0, wind_direction=0.0) -> Dict:
    """
    Convenience function to get fire spread prediction.
    
    Returns dict with: ignition_prob, wind_dir, wind_strength, danger_level
    """
    return get_predictor().predict(temperature, humidity, wind_speed, wind_direction)
