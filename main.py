"""
Real-Time Hand Sign Detection
==============================
Runs the webcam feed and detects hand signs in real time using the
MediaPipe Tasks API (HandLandmarker).

Usage:
    python main.py

Controls:
    Q - Quit the application
"""

import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp

from sign_detector import detect_sign, get_finger_states, WaveDetector, Sign
from visualizer import (
    draw_sign_badge,
    draw_finger_status,
    draw_wave_indicator,
    draw_hand_landmarks_custom,
    draw_fps,
    draw_instructions,
)

# --- Model download ---
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
    print(f"  URL: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"  Saved to {MODEL_PATH}")


def main():
    ensure_model()

    # --- Initialize MediaPipe HandLandmarker (Tasks API) ---
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # We use VIDEO mode for frame-by-frame processing with tracking
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    landmarker = HandLandmarker.create_from_options(options)

    # --- Wave Detector ---
    wave_detector = WaveDetector()

    # --- Open Camera ---
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check your camera connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 50)
    print("  Hand Sign Detection - Live Camera")
    print("=" * 50)
    print("Show your hand to the camera!")
    print("Detectable signs: Open Hand, Peace, Shaka, Thumbs Up")
    print("Bonus: Wave your hand for wave detection!")
    print("Press Q to quit.")
    print("=" * 50)

    prev_time = time.time()
    fps = 0
    frame_timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB and create MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect with increasing timestamp (required for VIDEO mode)
        frame_timestamp_ms += 33  # ~30fps
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # --- Process Detections ---
        if results.hand_landmarks:
            for idx, hand_lms in enumerate(results.hand_landmarks):
                # Get handedness
                handedness = "Right"
                if results.handedness and idx < len(results.handedness):
                    handedness = results.handedness[idx][0].category_name

                # Detect sign
                result = detect_sign(hand_lms, handedness)

                # Detect wave
                wave_detector.update(hand_lms)
                if wave_detector.is_waving:
                    result.sign = Sign.WAVING
                    result.confidence = 0.85

                # Get finger states for display
                finger_states = get_finger_states(hand_lms, handedness)

                # --- Draw Overlays ---
                draw_hand_landmarks_custom(frame, hand_lms, result.sign)
                draw_sign_badge(frame, result, position=(20, 30))
                draw_finger_status(frame, finger_states, position=(20, 140))
                draw_wave_indicator(frame, wave_detector.is_waving)
        else:
            wave_detector.reset()

        # --- FPS Calculation ---
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time

        draw_fps(frame, fps)
        draw_instructions(frame)

        # --- Display ---
        cv2.imshow("Hand Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Application closed.")


if __name__ == "__main__":
    main()
