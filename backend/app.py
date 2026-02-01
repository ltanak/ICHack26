from flask import Flask, request, jsonify, send_file
from typing import List
import json
from flask_cors import CORS
from satellite import get_satellite_image
from claude.claude import sendPrompt
from utils import MapPoint
from display_data.display import get_coords, get_image_path, save_image, overlay_image, get_snapshots
from pathlib import Path
from markupsafe import escape
from typing import List
from collections import deque

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


# Cache for satellite images by year
satellite_cache = {}
# Cache for overlay images by (year, snapshot)
overlay_cache = {}
# Track generated images and delete after N newer images have been served
cleanup_queue = deque()
CLEANUP_LAG = 10

@app.route('/satellite', methods=['GET'])
def getSatelliteImage():
    try:
        # get year from the request
        year = request.args.get('year', default=2017, type=int)
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

