"""
Hand Sign Detection Module
==========================
Detects hand signs (Open Hand, Peace, Surfing/Shaka, Thumbs Up) using
MediaPipe Hands landmarks. Provides both real-time camera detection and
single-image prediction.

Compatible with MediaPipe >= 0.10.14 (Tasks API).
"""

import math
from enum import Enum
from dataclasses import dataclass


class Sign(Enum):
    """Enumeration of detectable hand signs."""
    NONE = "No Sign"
    OPEN_HAND = "Open Hand"
    PEACE = "Peace"
    SURFING = "Shaka"
    THUMBS_UP = "Thumbs Up"
    WAVING = "Waving"


@dataclass
class DetectionResult:
    """Holds the result of a sign detection."""
    sign: Sign
    confidence: float
    handedness: str
    landmarks: list


# MediaPipe Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Hand skeleton connections for drawing
HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
]


def _distance(lm, i, j):
    """Euclidean distance between two landmarks."""
    return math.sqrt(
        (lm[i].x - lm[j].x) ** 2 +
        (lm[i].y - lm[j].y) ** 2 +
        (lm[i].z - lm[j].z) ** 2
    )


def is_finger_extended(landmarks, finger_tip, finger_pip, finger_mcp, wrist):
    """
    Check if a finger is extended by comparing the distance from
    the tip to the wrist vs the PIP joint to the wrist.
    """
    tip_to_wrist = _distance(landmarks, finger_tip, wrist)
    pip_to_wrist = _distance(landmarks, finger_pip, wrist)
    return tip_to_wrist > pip_to_wrist


def is_thumb_extended(landmarks, handedness):
    """
    Check if thumb is extended using lateral (x-axis) comparison.
    Accounts for handedness (mirrored in camera).
    """
    thumb_tip_x = landmarks[THUMB_TIP].x
    thumb_ip_x = landmarks[THUMB_IP].x

    if handedness == "Right":
        return thumb_tip_x > thumb_ip_x
    else:
        return thumb_tip_x < thumb_ip_x


def get_finger_states(landmarks, handedness):
    """
    Returns a dict with the extension state of each finger.
    True = extended, False = curled.
    """
    return {
        "thumb": is_thumb_extended(landmarks, handedness),
        "index": is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP, WRIST),
        "middle": is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, WRIST),
        "ring": is_finger_extended(landmarks, RING_TIP, RING_PIP, RING_MCP, WRIST),
        "pinky": is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP, WRIST),
    }


def detect_sign(landmarks, handedness="Right") -> DetectionResult:
    """
    Detect which hand sign is being shown based on finger states.

    Parameters
    ----------
    landmarks : list
        The 21 hand landmarks from MediaPipe.
    handedness : str
        "Left" or "Right" hand classification.

    Returns
    -------
    DetectionResult
        The detected sign with confidence score.
    """
    fingers = get_finger_states(landmarks, handedness)

    thumb = fingers["thumb"]
    index = fingers["index"]
    middle = fingers["middle"]
    ring = fingers["ring"]
    pinky = fingers["pinky"]

    # --- Open Hand: all 5 fingers extended ---
    if all([thumb, index, middle, ring, pinky]):
        spread = _distance(landmarks, INDEX_TIP, PINKY_TIP)
        palm_width = _distance(landmarks, INDEX_MCP, PINKY_MCP)
        confidence = min(1.0, spread / (palm_width + 1e-6))
        return DetectionResult(Sign.OPEN_HAND, confidence, handedness, landmarks)

    # --- Peace Sign: index + middle extended, others curled ---
    if index and middle and not thumb and not ring and not pinky:
        separation = _distance(landmarks, INDEX_TIP, MIDDLE_TIP)
        finger_len = _distance(landmarks, INDEX_MCP, INDEX_TIP)
        confidence = min(1.0, 0.7 + 0.3 * (separation / (finger_len + 1e-6)))
        return DetectionResult(Sign.PEACE, confidence, handedness, landmarks)

    # --- Surfing / Shaka: thumb + pinky extended, others curled ---
    if thumb and pinky and not index and not middle and not ring:
        confidence = 0.9
        return DetectionResult(Sign.SURFING, confidence, handedness, landmarks)

    # --- Thumbs Up: only thumb extended, all others curled ---
    if thumb and not index and not middle and not ring and not pinky:
        thumb_up = landmarks[THUMB_TIP].y < landmarks[THUMB_MCP].y
        confidence = 0.9 if thumb_up else 0.5
        return DetectionResult(Sign.THUMBS_UP, confidence, handedness, landmarks)

    return DetectionResult(Sign.NONE, 0.0, handedness, landmarks)


class WaveDetector:
    """Detect waving by tracking wrist x-position direction changes."""

    def __init__(self, threshold=5, window=20, min_move=0.005):
        self.threshold = threshold  
        self.window = window      
        self.min_move = min_move  
        self.positions = []
        self.is_waving = False

    def update(self, landmarks) -> tuple[bool, int]:
        self.positions.append(landmarks[0].x)
        if len(self.positions) > self.window:
            self.positions.pop(0)

        # Count direction reversals
        reversals = 0
        for i in range(2, len(self.positions)):
            prev_dir = self.positions[i-1] - self.positions[i-2]
            curr_dir = self.positions[i] - self.positions[i-1]
            if prev_dir * curr_dir < 0 and abs(curr_dir) > self.min_move:
                reversals += 1

        self.is_waving = reversals >= self.threshold
        return self.is_waving, reversals

    def reset(self):
        self.positions.clear()
        self.is_waving = False
