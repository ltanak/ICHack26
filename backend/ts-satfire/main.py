import sys
import os
from pathlib import Path

# Add TS-SatFire to path
ts_satfire_path = Path(__file__).parent / "TS-SatFire"
sys.path.insert(0, str(ts_satfire_path))
os.chdir(ts_satfire_path)

# Now import TS-SatFire modules
import run_spatial_model

def main():
    # Set up arguments for the model
    sys.argv = [
        "run_spatial_temp_model.py",
        "-m", "swinunetr",
        "-mode", "af",
        "-b", "32",
        "-r", "1",
        "-lr", "0.001",
        "-nh", "96",
        "-ed", "768",
        "-nc", "13",
        "-ts", "12",
        "-it", "1",
        "-test"
    ]
    
    print("DEBUG: sys.argv =", sys.argv)
    
    # Run the model
    run_spatial_model.main()

if __name__ == "__main__":
    main()