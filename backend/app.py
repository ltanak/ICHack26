from flask import Flask, request, jsonify, send_file
from typing import List
import json
from flask_cors import CORS
from satellite import get_satellite_image
from claude.claude import sendPrompt, get_historical_summary
from utils import MapPoint
from display_data.display import get_coords, get_image_path, save_image, overlay_image, get_snapshots
from pathlib import Path
from markupsafe import escape
from typing import List
from collections import deque
import io
import base64
import time
import matplotlib.pyplot as plt
from Krishna.visualize import FireSimulation
from Krishna.visualize_tft_web import TFTFireSimulation
import uuid
import numpy as np
# Import your simulation functions
from Krishna.simulation import (
    create_grid, step, monte_carlo_simulation, load_masks,
    EMPTY, TREE, BURNING, BURNT
)
from utils import MapPoint


app = Flask(__name__)
CORS(app)


# GET mark locations
@app.route('/points', methods=['GET'])
def markLocations() -> List[MapPoint]:
    if request.method == 'GET':
        with open('resources/wildfires.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(data)

            return jsonify(data)

YEAR = 2017

# POST selected point
@app.route('/summary/<int:year>', methods=['GET'])
def getSummary(year: int):
    global YEAR
    res = get_historical_summary(year=year)
    YEAR = year

    # For now just return Lorum Ipsum to save API calls
    # Wait to simulate api response time
    return jsonify(res)
    # return sendPrompt(prompt)


# Cache for satellite images by year
satellite_cache = {}
# Cache for overlay images by (year, snapshot)
overlay_cache = {}
# Track generated images and delete after N newer images have been served
cleanup_queue = deque()
CLEANUP_LAG = 10

@app.route('/satellite/', methods=['GET'])
def getSatelliteImage():
    global YEAR
    try:
        # get year from the request
        year = request.args.get('year', default=YEAR, type=int)
        snapshot = request.args.get('snapshot', default=None, type=str)

        # Serve cached overlay if available
        cache_key = (year, snapshot)
        if cache_key in overlay_cache:
            cached_overlay, cached_fire = overlay_cache[cache_key]
            if cached_overlay.exists():
                return send_file(cached_overlay, mimetype='image/png')

        # Check if satellite image is cached
        if year not in satellite_cache:
            # base imagee
            coords = get_coords(year)

            out_file, width_px, height_px = get_satellite_image(
                min_lon=coords[0],
                min_lat=coords[1],
                max_lon=coords[2],
                max_lat=coords[3],
                out_file=f"Datasets/satellites/{year}.png"
            )
            satellite_cache[year] = Path(out_file)
        else:
            out_file = satellite_cache[year]

        # overlayed image path
        overlay_path, fire_image_path = overlay_image(year, out_file, snapshot)
        print(f"Serving overlay: {overlay_path}")

        if not overlay_path or not overlay_path.exists():
            return jsonify({'error': 'Failed to generate overlay'}), 500

        # Cache the overlay for reuse
        overlay_cache[cache_key] = (overlay_path, fire_image_path)

        # NOTE: Cleanup disabled for debugging - images will persist on disk
        # Enqueue images for delayed cleanup
        # cleanup_queue.append((cache_key, overlay_path, fire_image_path))

        # Delete images that are older than CLEANUP_LAG
        # while len(cleanup_queue) > CLEANUP_LAG:
        #     old_key, old_overlay, old_fire = cleanup_queue.popleft()
        #     overlay_cache.pop(old_key, None)
        #     for path in (old_overlay, old_fire):
        #         try:
        #             if path and path.exists():
        #                 path.unlink()
        #         except Exception as cleanup_error:
        #             print(f"Cleanup error for {path}: {cleanup_error}")

        return send_file(overlay_path, mimetype='image/png')
    except Exception as e:
        print(f"Error in getSatelliteImage: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/snapshots', methods=['GET'])
def getSnapshots():
    """Get list of all available snapshot files for a year."""
    year = request.args.get('year', default=2017, type=int)
    snapshots = get_snapshots(year)
    return jsonify({'year': year, 'snapshots': snapshots, 'count': len(snapshots)})

if __name__ == '__main__':
    app.run(debug=False, port=5001)


@app.route('/simulation/frame', methods=['GET'])
def get_simulation_frame():
    """Get current simulation frame as base64 image"""
    # Get or create global simulation instance
    if not hasattr(get_simulation_frame, 'sim'):
        get_simulation_frame.sim = FireSimulation()

    sim = get_simulation_frame.sim

    # Capture current matplotlib figure as image
    buf = io.BytesIO()
    sim.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)

    # Convert to base64
    img_base64 = base64.b64encode(buf.read()).decode()

    return jsonify({
        'image': f'data:image/png;base64,{img_base64}',
        'mode': sim.mode,
        'p_tree': sim.p_tree,
        'ignition_prob': sim.ignition_prob,
        'wind_strength': sim.wind_strength
    })




@app.route('/simulation/step', methods=['POST'])
def simulation_step():
    """Advance simulation by one step"""
    sim = get_simulation_frame.sim
    sim.update(None)
    return get_simulation_frame()




# Store simulation state (in production, use Redis or sessions)
simulation_state = {}

# Wind directions mapping
WIND_DIRS = {
    "None": (0, 0),
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


def grid_to_json(grid, ca_mask):
    """Convert numpy grid to JSON-serializable format."""
    return {
        'grid': grid.tolist(),
        'mask': ca_mask.tolist(),
        'shape': grid.shape
    }


@app.route('/api/init', methods=['POST'])
def init_simulation():
    """
    Initialize a new simulation.
    
    Request body:
    {
        "p_tree": 0.6,
        "mode": "historic" or "custom",
        "custom_fires": [[y1, x1], [y2, x2], ...] (optional, for custom mode)
    }
    
    Returns:
    {
        "session_id": "uuid",
        "grid": [[...]],
        "mask": [[...]],
        "shape": [N, N]
    }
    """
    data = request.json
    p_tree = data.get('p_tree', 0.6)
    mode = data.get('mode', 'historic')
    custom_fires = data.get('custom_fires', [])
    
    # Create grid
    grid, ca_mask = create_grid(p_tree=p_tree)
    
    # For custom mode, remove historic fires
    if mode == 'custom':
        grid[grid == BURNING] = TREE
        # Add custom fire points if provided
        for y, x in custom_fires:
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                if ca_mask[y, x] == 1 and grid[y, x] == TREE:
                    grid[y, x] = BURNING
    
    # Create session
    session_id = str(uuid.uuid4())
    simulation_state[session_id] = {
        'grid': grid,
        'ca_mask': ca_mask
    }
    
    return jsonify({
        'session_id': session_id,
        **grid_to_json(grid, ca_mask)
    })


@app.route('/api/step', methods=['POST'])
def step_simulation():
    """
    Advance simulation by one step.
    
    Request body:
    {
        "session_id": "uuid",
        "ignition_prob": 0.7,
        "wind_dir": "None" | "Up" | "Down" | "Left" | "Right",
        "wind_strength": 1.0
    }
    
    Returns:
    {
        "grid": [[...]],
        "has_burning": true/false
    }
    """
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in simulation_state:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    ignition_prob = data.get('ignition_prob', 0.7)
    wind_dir_name = data.get('wind_dir', 'None')
    wind_strength = data.get('wind_strength', 1.0)
    
    wind_dir = WIND_DIRS.get(wind_dir_name, (0, 0))
    
    # Get current grid
    current_grid = simulation_state[session_id]['grid']
    ca_mask = simulation_state[session_id]['ca_mask']
    
    # Advance one step
    new_grid = step(current_grid, ignition_prob, wind_dir, wind_strength)
    
    # Update state
    simulation_state[session_id]['grid'] = new_grid
    
    # Check if still burning
    has_burning = np.any(new_grid == BURNING)
    
    return jsonify({
        'grid': new_grid.tolist(),
        'has_burning': bool(has_burning)
    })


@app.route('/api/add-fire', methods=['POST'])
def add_fire():
    """
    Add a fire point to the grid (custom mode).
    
    Request body:
    {
        "session_id": "uuid",
        "y": 50,
        "x": 50
    }
    
    Returns:
    {
        "grid": [[...]],
        "success": true/false
    }
    """
    data = request.json
    session_id = data.get('session_id')
    y = data.get('y')
    x = data.get('x')
    
    if session_id not in simulation_state:
        print("AH")
        return jsonify({'error': 'Invalid session ID'}), 400
    
    grid = simulation_state[session_id]['grid']
    ca_mask = simulation_state[session_id]['ca_mask']
    
    # Validate coordinates
    if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
        if ca_mask[y, x] == 1 and grid[y, x] == TREE:
            grid[y, x] = BURNING
            print("SUCCESS")
            success = True
        else:
            print("FAIL - 1")
            success = False
    else:
        print("FAIL - 2")
        success = False
    
    return jsonify({
        'grid': grid.tolist(),
        'success': success
    })


@app.route('/api/monte-carlo', methods=['POST'])
def run_monte_carlo():
    """
    Run Monte Carlo simulation.
    
    Request body:
    {
        "n_runs": 20,
        "p_tree": 0.6,
        "ignition_prob": 0.7,
        "wind_dir": "None",
        "wind_strength": 1.0,
        "mode": "historic" or "custom",
        "custom_fires": [[y1, x1], ...] (optional, for custom mode),
        "retardant_zones": [[y, x, radius], ...] (optional),
        "cleared_zones": [[y, x, width, height], ...] (optional)
    }
    
    Returns:
    {
        "probability": [[...]],
        "mask": [[...]]
    }
    """
    data = request.json
    n_runs = data.get('n_runs', 20)
    p_tree = data.get('p_tree', 0.8)
    ignition_prob = data.get('ignition_prob', 0.8)
    wind_dir_name = data.get('wind_dir', 'None')
    wind_strength = data.get('wind_strength', 1.0)
    mode = data.get('mode', 'historic')
    custom_fires = data.get('custom_fires', [])
    retardant_zones = data.get('retardant_zones', [])
    cleared_zones = data.get('cleared_zones', [])
    
    wind_dir = WIND_DIRS.get(wind_dir_name, (0, 0))
    
    # Determine custom fire points
    custom_fire_points = custom_fires if mode == 'custom' else None
    
    # Convert mitigation zones to tuples (format expected by simulation)
    retardant_zones_tuples = [tuple(z) for z in retardant_zones] if retardant_zones else None
    cleared_zones_tuples = [tuple(z) for z in cleared_zones] if cleared_zones else None
    
    print(f"Monte Carlo params: n_runs={n_runs}, p_tree={p_tree}, ignition_prob={ignition_prob}, mode={mode}")
    print(f"Custom fire points: {custom_fire_points}")
    print(f"Retardant zones: {retardant_zones_tuples}")
    print(f"Cleared zones: {cleared_zones_tuples}")
    
    # Run Monte Carlo with mitigation zones
    burn_prob = monte_carlo_simulation(
        n_runs=n_runs,
        p_tree=p_tree,
        ignition_prob=ignition_prob,
        wind_dir=wind_dir,
        wind_strength=wind_strength,
        custom_fire_points=custom_fire_points,
        retardant_zones=retardant_zones_tuples,
        cleared_zones=cleared_zones_tuples
    )
    
    print(f"Burn probability stats: min={burn_prob.min()}, max={burn_prob.max()}, mean={burn_prob.mean()}")
    print(f"Non-zero cells: {np.count_nonzero(burn_prob)}")
    
    # Get mask (ca_mask is first return value)
    ca_mask, fire_mask, _ = load_masks()
    
    print(f"CA mask stats: sum={ca_mask.sum()}, shape={ca_mask.shape}")
    print(f"Fire mask stats: sum={(fire_mask == BURNING).sum()}, shape={fire_mask.shape}")
    
    # Convert to lists for JSON serialization
    prob_list = burn_prob.tolist()
    mask_list = ca_mask.tolist()
    
    # Verify data before sending
    print(f"Probability list type: {type(prob_list)}, len: {len(prob_list)}")
    print(f"First row sample (first 10 values): {prob_list[0][:10]}")
    print(f"Sample non-zero values from row 200: {[v for v in prob_list[200][100:150] if v > 0][:5]}")
    
    return jsonify({
        'probability': prob_list,
        'mask': mask_list
    })


@app.route('/api/get-masks', methods=['GET'])
def get_masks():
    """
    Get California and fire masks.
    
    Returns:
    {
        "ca_mask": [[...]],
        "shape": [N, N]
    }
    """
    ca_mask, _, _ = load_masks()
    
    return jsonify({
        'ca_mask': ca_mask.tolist(),
        'shape': ca_mask.shape
    })


# TFT Simulation endpoints
@app.route('/tft/init', methods=['POST'])
def init_tft_simulation():
    """
    Initialize a new TFT fire simulation.
    
    Request body:
    {
        "temperature": 50,
        "humidity": 5,
        "wind_speed": 20,
        "wind_direction": 270
    }
    
    Returns:
    {
        "session_id": "uuid",
        "image": "base64_encoded_image",
        "stats": {...}
    }
    """
    data = request.json or {}
    
    # Get weather parameters
    temperature = data.get('temperature', 50)
    humidity = data.get('humidity', 5)
    wind_speed = data.get('wind_speed', 20)
    wind_direction = data.get('wind_direction', 270)
    
    # Create TFT simulation
    sim = TFTFireSimulation(
        temperature=temperature,
        humidity=humidity,
        wind_speed=wind_speed,
        wind_direction=wind_direction
    )
    
    # Store in global state
    session_id = str(uuid.uuid4())
    if not hasattr(init_tft_simulation, 'sessions'):
        init_tft_simulation.sessions = {}
    init_tft_simulation.sessions[session_id] = sim
    
    # Capture current frame
    buf = io.BytesIO()
    sim.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    
    return jsonify({
        'session_id': session_id,
        'image': f'data:image/png;base64,{img_base64}',
        'stats': sim.get_stats()
    })


@app.route('/tft/step/<session_id>', methods=['POST'])
def tft_step(session_id):
    """
    Advance TFT simulation by one step.
    
    Returns:
    {
        "image": "base64_encoded_image",
        "stats": {...}
    }
    """
    if not hasattr(init_tft_simulation, 'sessions'):
        return jsonify({'error': 'Session not found'}), 404
    
    if session_id not in init_tft_simulation.sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    sim = init_tft_simulation.sessions[session_id]
    sim.step()
    
    # Capture current frame
    buf = io.BytesIO()
    sim.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    
    return jsonify({
        'image': f'data:image/png;base64,{img_base64}',
        'stats': sim.get_stats()
    })


@app.route('/tft/reset/<session_id>', methods=['POST'])
def tft_reset(session_id):
    """
    Reset TFT simulation to initial state.
    
    Returns:
    {
        "image": "base64_encoded_image",
        "stats": {...}
    }
    """
    if not hasattr(init_tft_simulation, 'sessions'):
        return jsonify({'error': 'Session not found'}), 404
    
    if session_id not in init_tft_simulation.sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    sim = init_tft_simulation.sessions[session_id]
    sim.reset()
    
    # Capture current frame
    buf = io.BytesIO()
    sim.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    
    return jsonify({
        'image': f'data:image/png;base64,{img_base64}',
        'stats': sim.get_stats()
    })


@app.route('/tft/monte-carlo/<session_id>', methods=['POST'])
def tft_monte_carlo(session_id):
    """
    Run Monte Carlo analysis on TFT simulation.
    
    Request body:
    {
        "n_runs": 20
    }
    
    Returns:
    {
        "image": "base64_encoded_image",
        "stats": {...}
    }
    """
    if not hasattr(init_tft_simulation, 'sessions'):
        return jsonify({'error': 'Session not found'}), 404
    
    if session_id not in init_tft_simulation.sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.json or {}
    n_runs = data.get('n_runs', 20)
    
    sim = init_tft_simulation.sessions[session_id]
    sim.run_monte_carlo(n_runs=n_runs)
    
    # Capture current frame
    buf = io.BytesIO()
    sim.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    
    return jsonify({
        'image': f'data:image/png;base64,{img_base64}',
        'stats': sim.get_stats()
    })


@app.route('/tft/frame/<session_id>', methods=['GET'])
def tft_get_frame(session_id):
    """
    Get current TFT simulation frame and stats.
    
    Returns:
    {
        "image": "base64_encoded_image",
        "stats": {...}
    }
    """
    if not hasattr(init_tft_simulation, 'sessions'):
        return jsonify({'error': 'Session not found'}), 404
    
    if session_id not in init_tft_simulation.sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    sim = init_tft_simulation.sessions[session_id]
    
    # Capture current frame
    buf = io.BytesIO()
    sim.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    
    return jsonify({
        'image': f'data:image/png;base64,{img_base64}',
        'stats': sim.get_stats()
    })
