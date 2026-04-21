#!/usr/bin/env python3
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional vision dependency
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "vision" / "resource" / "lane.png"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "vision" / "lane_result.png"


def require_cv2():
    if cv2 is None:
        raise RuntimeError("Lane perception requires OpenCV. Install opencv-python before using this module.")
    return cv2

CROP_TOP   = 0.45  # crop top 45%
CROP_LEFT  = 0.40  # crop left 40%
CROP_RIGHT = 0.40  # crop right 40%

def pre_process(image):
    cv = require_cv2()
    """
    Pre-process the lane image to enhance lane markings.
    Including blur, convert BGR to HSV, color thresholding,
    morphological operations
    """
    # blur image
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    # convert BGR to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # color thresholding
    lower_bound = np.array([15, 50, 50])
    upper_bound = np.array([40, 255, 255])
    mask = cv.inRange(hsv, lower_bound, upper_bound)
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask

def find_lane_contours(mask, image):
    cv = require_cv2()
    """
    Find contours in the binary mask and filter them based on area.
    Returns a list of contours that likely correspond to lane markings.
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    good_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 30:
            continue
        good_contours.append(cnt)
    output = image.copy()
    cv.drawContours(output, good_contours, -1, (0, 255, 0), -1)  # -1 = filled
    return output

if __name__ == "__main__":
    cv = require_cv2()
    img = cv.imread(str(DEFAULT_IMAGE_PATH))
    if img is None:
        raise FileNotFoundError(f"Could not read {DEFAULT_IMAGE_PATH}")
    # Crop to floor area only — removes walls/signs from detection
    h, w = img.shape[:2]
    crop_y = int(h * CROP_TOP)
    crop_x0 = int(w * CROP_LEFT)
    crop_x1 = int(w * (1 - CROP_RIGHT))
    cropped = img[crop_y:, crop_x0:crop_x1]
    mask = pre_process(cropped)
    output = find_lane_contours(mask, cropped)

    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(DEFAULT_OUTPUT_PATH), output)
    print(f"Saved {DEFAULT_OUTPUT_PATH}")
