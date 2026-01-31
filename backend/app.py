from flask import Flask, request, jsonify
from markupsafe import escape
from typing import List
import json
from flask_cors import CORS


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
def getSummary(point: MapPoint):
    prompt = "PLACE_HOLDER"

    return sendPrompt(prompt)

# request a summary for it 
 