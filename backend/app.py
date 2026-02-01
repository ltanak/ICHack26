from flask import Flask, request, jsonify, send_file
from typing import List
import json
from flask_cors import CORS
from satellite import get_satellite_image
from claude.claude import sendPrompt
from utils import MapPoint
from display_data.display import get_coords, get_image_path, save_image, overlay_image
from pathlib import Path
from markupsafe import escape
from typing import List
import io
import base64
import matplotlib.pyplot as plt
from Krishna.visualize import FireSimulation

from utils import MapPoint


app = Flask(__name__)
CORS(app)


# GET mark locations
@app.route('/points', methods=['GET'])
def markLocations() -> List[MapPoint]:
    if request.method == 'GET':
        with open('resources/wildfires.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

            return jsonify(data)


# POST selected point
@app.route('/summary', methods=['GET'])
def getSummary(point: MapPoint):
    prompt = "PLACE_HOLDER"

    return sendPrompt(prompt)


@app.route('/satellite', methods=['GET'])
def getSatelliteImage():
    if request.method == 'GET':

        # get year from the request
        year = 2017

        # base imagee
        coords = get_coords(year)

        out_file, width_px, height_px = get_satellite_image(
            min_lon=coords[0],
            min_lat=coords[1],
            max_lon=coords[2],
            max_lat=coords[3],
            out_file=f"Datasets/satellites/{year}.png"
        )

        # overlayed image path
        overlay_path = overlay_image(year, Path(out_file))
        print(overlay_path)

        return send_file(overlay_path, mimetype='image/png')


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
