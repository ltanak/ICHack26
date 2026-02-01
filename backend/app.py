from flask import Flask, request, jsonify, send_file
from typing import List
import json
from flask_cors import CORS
from satellite import get_satellite_image
# from claude import sendPrompt
from utils import MapPoint
from display_data.display import get_coords, get_image_path, save_image, overlay_image
from pathlib import Path
from markupsafe import escape
from typing import List
import time

# from claude import sendPrompt
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
@app.route('/summary/<path:point>', methods=['GET'])
def getSummary(point: MapPoint):
    prompt = "PLACE_HOLDER"
    print("reached")

    # For now just return Lorum Ipsum to save API calls
    # Wait to simulate api response time
    time.sleep(2)
    return jsonify("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean porttitor magna eget tempus tristique. Suspendisse commodo arcu lacus, id iaculis elit aliquam non. Sed interdum dignissim turpis, vitae cursus neque volutpat eget. Aliquam vitae efficitur mauris. Integer ullamcorper, diam vitae pharetra tristique, lorem lorem lacinia diam, a cursus libero metus a felis. Sed luctus venenatis pellentesque. Donec sed nunc eget nunc placerat lobortis. Donec fringilla leo dolor, quis faucibus enim imperdiet sed. In fermentum libero ipsum, eget convallis eros gravida et. Aliquam ex massa, bibendum non rutrum vel, posuere mollis ante. Nam congue laoreet dictum. Nulla quis neque imperdiet, fringilla lacus ac, iaculis mi. Nunc laoreet lacinia feugiat. Aliquam nec rhoncus nisi, a malesuada velit.")
    # return sendPrompt(prompt)


@app.route('/satellite', methods=['GET'])
def getSatelliteImage():
    if request.method == 'GET':

        # get year from the request
        year = 1

        # base imagee
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

