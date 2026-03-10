"""
API Endpoint for Hand Sign Detection
=====================================
Flask server with a /predict endpoint that accepts hand sign images
as base64-encoded JSON and returns the detected sign.

Usage:
    python api.py

Endpoints:
    POST /predict  - Detect hand sign from image
    GET  /health   - Health check
    GET  /         - API documentation page
"""

import os
import base64
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, render_template_string

from sign_detector import detect_sign, get_finger_states, Sign

app = Flask(__name__)

# --- Model setup ---
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model():
    """Download the HandLandmarker model if not present."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"  Saved to {MODEL_PATH}")


# Download model and initialize at startup
ensure_model()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(options)


API_DOCS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Hand Sign Detection API</title>
</head>
<body>
    <h1>Hand Sign Detection API</h1>
    <p>POST /predict - Send a base64-encoded image to detect hand signs.</p>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    """API documentation page with interactive upload."""
    return render_template_string(API_DOCS_HTML)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the hand sign from a base64-encoded image.

    Request body (JSON):
        { "image": "<base64_encoded_image_string>" }

    Returns:
        JSON with detected sign, confidence, handedness, and finger states.
    """
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field in JSON body"}), 400

    try:
        image_data = base64.b64decode(data["image"])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not decode image"}), 400

    except Exception as e:
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

    # Process with MediaPipe Tasks API
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(mp_image)

    if not results.hand_landmarks:
        return jsonify({
            "sign": Sign.NONE.value,
            "confidence": 0.0,
            "handedness": None,
            "fingers": None,
            "message": "No hand detected in image",
        })

    landmarks = results.hand_landmarks[0]
    handedness = "Right"
    if results.handedness:
        handedness = results.handedness[0][0].category_name

    detection = detect_sign(landmarks, handedness)
    finger_states = get_finger_states(landmarks, handedness)

    return jsonify({
        "sign": detection.sign.value,
        "confidence": round(detection.confidence, 3),
        "handedness": detection.handedness,
        "fingers": finger_states,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
