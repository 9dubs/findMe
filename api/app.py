from flask import Flask, render_template, request, jsonify
import pandas as pd
import cv2
import numpy as np
import base64
import json

from utils import computeD

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    if request.method == 'POST':
        inp = request.get_data()
        data = json.loads(inp)
        imageString = base64.b64decode(data["img"])
        nparr = np.fromstring(imageString, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

        dis, result = computeD(img)

        data = {
            "distance": dis,
            "name": result
        }

        return jsonify(data)

if __name__ == '__main__':
    app.run()