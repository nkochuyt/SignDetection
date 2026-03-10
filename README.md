# Hand Sign Detection: Sopra Steria Demo

Real-time hand gesture detection using MediaPipe and OpenCV, with a small Flask API on the side.

Built as a take-home demo task. I didn't have a lot of time, so I kept the scope tight and focused on getting something working end-to-end rather than going broad.

---

## What's in here

| File | What it does |
|---|---|
| `main.py` | Runs real-time detection via webcam with a HUD overlay |
| `sign_detector.py` | Core logic: landmark analysis and gesture classification |
| `visualizer.py` | Draws the overlay (sign badge, finger state panel, FPS) |
| `api.py` | Flask API with a `/predict` endpoint for image-based prediction |
| `requirements.txt` | Dependencies |

---

## How I approached it

MediaPipe gives you 21 3D landmarks per hand out of the box, no training needed, which made it a practical choice given the time constraint.

The detection itself is rule-based: for each finger I compare the distance from the fingertip to the wrist versus the PIP joint to the wrist. If the tip is farther, the finger is extended. The thumb is handled separately since it moves laterally, so I compare x-coordinates instead, adjusted for handedness.

Each gesture maps to a fixed pattern of extended/curled fingers:
- **Open Hand**: all five extended
- **Peace**: index + middle only
- **Shaka**: thumb + pinky only
- **Thumbs Up**: thumb only, pointing up

Wave detection tracks the wrist's x-position over a sliding window and counts direction reversals above a minimum amplitude threshold.

The API wraps the same detector and accepts a base64 image, returning the detected sign and finger states.

---

## Run it

```bash
pip install -r requirements.txt

# Webcam detection
python main.py

# API server (http://localhost:5000)
python api.py
```

Press `Q` to quit the webcam view.

---
