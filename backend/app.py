from flask import Flask, request, jsonify, send_file
from typing import List
import json
from flask_cors import CORS
from satellite import get_satellite_image
from claude import sendPrompt
from utils import MapPoint
from display_data.display import get_coords, get_image_path, save_image, overlay_image
from pathlib import Path
from markupsafe import escape
from typing import List

from claude import sendPrompt
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
        year = 1

        coords = get_coords(year)

        out_file, width_px, height_px = get_satellite_image(
            min_lon=coords[0],
            min_lat=coords[1],
            max_lon=coords[2],
            max_lat=coords[3],
            out_file=f"Datasets/satellite/{year}.png"
        )

        # overlayed image path
        overlay_path = overlay_image(year, Path(out_file))

        # overlay image path
        data = {
            "overlay_path": str(overlay_path),
            "width_px": width_px,
            "height_px": height_px
        }

        return send_file(overlay_path, mimetype='image/png')

