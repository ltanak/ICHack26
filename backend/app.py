from flask import Flask, request, jsonify
from markupsafe import escape
from typing import List

from claude import sendPrompt
from utils import MapPoint

app = Flask(__name__)



# GET mark locations
@app.route('/points', methods=['GET'])
def markLocations() -> List[MapPoint]:
    if request.method == 'GET':
        data = [{
            "name": "California",
            "markerOffset": 15,
            "coordinates": [-119.4179, 36.7783]
        }]
    
        return jsonify(data)


# POST selected point
def getSummary(point: MapPoint):
    prompt = "PLACE_HOLDER"

    return sendPrompt(prompt)

# request a summary for it 
 