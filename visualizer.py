"""
Visualization Module
====================
Provides drawing utilities for hand sign detection overlays.
Creates a polished HUD-style display on the camera feed.

Uses custom drawing (no mp.solutions.drawing_utils dependency).
"""

import cv2
import numpy as np
from sign_detector import Sign, DetectionResult, HAND_CONNECTIONS


# --- Color Palette (BGR) ---
COLORS = {
    Sign.OPEN_HAND: (0, 200, 100),    # Green
    Sign.PEACE: (230, 160, 50),        # Blue/Teal
    Sign.SURFING: (0, 180, 255),       # Orange
    Sign.THUMBS_UP: (50, 200, 230),    # Yellow
    Sign.WAVING: (200, 100, 255),      # Pink/Magenta
    Sign.NONE: (180, 180, 180),        # Gray
}

BG_COLOR = (20, 20, 20)
ACCENT = (255, 200, 50)


def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1, alpha=0.7):
    """Draw a rounded rectangle with optional transparency."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_hand_landmarks_custom(frame, landmarks, sign: Sign):
    """
    Draw hand landmarks and connections directly using OpenCV.
    No dependency on mp.solutions.drawing_utils.

    Parameters
    ----------
    frame : np.ndarray
        The BGR image to draw on.
    landmarks : list
        List of 21 landmarks with .x, .y attributes (normalized 0-1).
    sign : Sign
        The detected sign (used for color coding).
    """
    h, w = frame.shape[:2]
    color = COLORS.get(sign, COLORS[Sign.NONE])

    # Convert normalized landmarks to pixel coordinates
    points = []
    for lm in landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

    # Draw connections (skeleton lines)
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx],
                 (255, 255, 255), 1, cv2.LINE_AA)

    # Draw landmark dots
    for i, (px, py) in enumerate(points):
        # Fingertips get a slightly larger dot
        radius = 5 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)


def draw_sign_badge(frame, result: DetectionResult, position=(20, 30)):
    """Draw a stylish badge showing the detected sign."""
    sign = result.sign
    color = COLORS.get(sign, COLORS[Sign.NONE])
    text = sign.value
    confidence = result.confidence

    x, y = position

    # Badge background
    badge_w = 340
    badge_h = 90
    draw_rounded_rect(frame, (x, y), (x + badge_w, y + badge_h),
                      BG_COLOR, radius=12, alpha=0.85)

    # Accent bar on the left
    cv2.rectangle(frame, (x, y + 8), (x + 5, y + badge_h - 8), color, -1)

    # Sign name
    cv2.putText(frame, text, (x + 18, y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

    # Confidence bar
    bar_x = x + 18
    bar_y = y + 55
    bar_w = badge_w - 36
    bar_h = 16

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)

    fill_w = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                  color, -1)

    conf_text = f"{confidence * 100:.0f}%"
    cv2.putText(frame, conf_text, (bar_x + bar_w + 5, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)


def draw_finger_status(frame, finger_states, position=(20, 140)):
    """Draw a mini panel showing which fingers are extended/curled."""
    x, y = position
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_keys = ["thumb", "index", "middle", "ring", "pinky"]

    panel_w = 200
    panel_h = 30 + len(finger_names) * 28
    draw_rounded_rect(frame, (x, y), (x + panel_w, y + panel_h),
                      BG_COLOR, radius=10, alpha=0.8)

    cv2.putText(frame, "FINGERS", (x + 12, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACCENT, 1, cv2.LINE_AA)

    for i, (name, key) in enumerate(zip(finger_names, finger_keys)):
        fy = y + 45 + i * 28
        is_up = finger_states.get(key, False)

        indicator_color = (0, 220, 100) if is_up else (60, 60, 80)
        cv2.circle(frame, (x + 20, fy), 6, indicator_color, -1)

        label = f"{name}: {'UP' if is_up else 'DOWN'}"
        text_color = (200, 200, 200) if is_up else (100, 100, 100)
        cv2.putText(frame, label, (x + 35, fy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)


def draw_wave_indicator(frame, is_waving):
    """Draw a waving indicator at the top-right corner."""
    h, w = frame.shape[:2]
    if is_waving:
        x = w - 180
        y = 30
        draw_rounded_rect(frame, (x, y), (x + 160, y + 50),
                          (40, 20, 60), radius=10, alpha=0.85)
        cv2.putText(frame, "WAVING!", (x + 15, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 255), 2, cv2.LINE_AA)


def draw_fps(frame, fps):
    """Display FPS counter."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.0f}"
    cv2.putText(frame, text, (w - 130, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1, cv2.LINE_AA)


def draw_instructions(frame):
    """Draw instruction text at the bottom."""
    h, w = frame.shape[:2]
    instructions = "Show: Open Hand | Peace | Shaka | Thumbs Up | Wave  |  Q to quit"
    text_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    tx = (w - text_size[0]) // 2
    ty = h - 12

    draw_rounded_rect(frame, (0, h - 35), (w, h), BG_COLOR, radius=0, alpha=0.7)
    cv2.putText(frame, instructions, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
