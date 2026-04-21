#!/usr/bin/env python3
import glob
import numpy as np
import os
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional vision dependency
    cv2 = None


def require_cv2():
    if cv2 is None:
        raise RuntimeError("Distance perception requires OpenCV. Install opencv-python before using this module.")
    return cv2

# =========================
# User-adjustable settings
# =========================

# Checkerboard is 6x8 vertices, side width 25 mm
CHECKERBOARD_ROWS = 6
CHECKERBOARD_COLS = 8
SQUARE_SIZE_METERS = 0.025

PROJECT_ROOT = Path(__file__).resolve().parents[4]
VISION_DATA_DIR = PROJECT_ROOT / "data" / "vision"
CALIB_DIR = str(VISION_DATA_DIR / "calibration")
KNOWN_IMAGE_PATH = str(VISION_DATA_DIR / "resource" / "cone_x40cm.png")
UNKNOWN_IMAGE_PATH = str(VISION_DATA_DIR / "resource" / "cone_unknown.png")
CALIBRATION_RESULTS_PATH = PROJECT_ROOT / "outputs" / "vision" / "calibration_results.npz"

# Known forward distance of the cone in cone_x40cm.png
KNOWN_X_METERS = 0.40

# y-axis convention:image left = positive y_car
LEFT_POSITIVE = True


# =========================
# Mouse click helper
# =========================

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def get_click_point(image, window_name="Click point"):
    """
    Show image and let user click one pixel.
    Returns (u, v) pixel coordinates.
    """
    global clicked_point
    cv = require_cv2()
    clicked_point = None
    last_printed_point = None

    display = image.copy()
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(window_name, mouse_callback)

    print(f"\nClick the required point in window: {window_name}")
    print("Press 'r' to reset click, 'c' to confirm, 'q' or ESC to quit.")

    while True:
        temp = display.copy()

        if clicked_point is not None:
            cv.circle(temp, clicked_point, 6, (0, 0, 255), -1)
            cv.putText(
                temp,
                f"{clicked_point}",
                (clicked_point[0] + 10, clicked_point[1] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            if clicked_point != last_printed_point:
                print(f"Selected point: {clicked_point}. Press 'c' to confirm.")
                last_printed_point = clicked_point

        cv.imshow(window_name, temp)
        key = cv.waitKey(20) & 0xFF

        if key == ord('r'):
            clicked_point = None
            last_printed_point = None
            print("Selection reset.")
        elif key == ord('q') or key == 27:
            cv.destroyWindow(window_name)
            raise KeyboardInterrupt("User quit point selection.")
        elif key == ord('c'):
            if clicked_point is not None:
                break

    cv.destroyWindow(window_name)
    return clicked_point


# =========================
# Camera calibration
# =========================

def calibrate_camera(calib_dir):
    """
    Calibrate camera from checkerboard images.
    Returns:
        K: intrinsic matrix
        dist: distortion coefficients
    """
    pattern_size = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)

    # Prepare world coordinates of checkerboard corners
    # z = 0 plane
    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_METERS

    objpoints = []
    imgpoints = []

    image_paths = sorted(glob.glob(os.path.join(calib_dir, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in {calib_dir}")

    gray_shape = None
    used_images = 0

    print("\n=== Calibration ===")
    for path in image_paths:
        img = cv.imread(path)
        if img is None:
            print(f"Skipping unreadable file: {path}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        found, corners = cv.findChessboardCorners(gray, pattern_size, None)

        if found:
            criteria = (
                cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
            corners_refined = cv.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            objpoints.append(objp)
            imgpoints.append(corners_refined)
            used_images += 1
            print(f"Used: {path}")
        else:
            print(f"Chessboard NOT found: {path}")

    if used_images < 3:
        raise RuntimeError("Too few valid checkerboard images for calibration.")

    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        gray_shape,
        None,
        None
    )

    print(f"\nCalibration RMS error: {ret}")
    print("Intrinsic matrix K:")
    print(K)
    print("Distortion coefficients:")
    print(dist.ravel())

    return K, dist


# =========================
# Geometry
# =========================

def undistort_pixel(u, v, K, dist):
    """
    Undistort a single pixel and return ideal pixel coordinates.
    """
    cv = require_cv2()
    pts = np.array([[[u, v]]], dtype=np.float32)
    undist = cv.undistortPoints(pts, K, dist, P=K)
    uu, vv = undist[0, 0]
    return float(uu), float(vv)


def estimate_camera_height(known_pixel, K, dist, known_x):
    """
    Estimate camera mounting height H from the known image.
    
    Assumption used:
        x_car = H * fy / (v - cy)
    so:
        H = known_x * (v - cy) / fy

    We take absolute value to keep height positive, since sign
    convention can vary depending on axis definition.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u, v = known_pixel
    u_u, v_u = undistort_pixel(u, v, K, dist)

    denom = (v_u - cy)
    if abs(denom) < 1e-6:
        raise ZeroDivisionError("Known point too close to principal point vertically.")

    H = abs(known_x * denom / fy)

    print("\n=== Height Estimation ===")
    print(f"Known pixel (raw): {(u, v)}")
    print(f"Known pixel (undistorted): {(u_u, v_u)}")
    print(f"Estimated camera height H: {H:.6f} m")

    return H


def pixel_to_car(u, v, K, dist, H, left_positive=True):
    """
    Convert image pixel to (x_car, y_car), assuming the point is on the ground.

    Model:
        x_car = H * fy / (v - cy)
        y_car = (u - cx) * x_car / fx

    Because sign conventions differ, enforce:
    - x_car as positive forward
    - y sign optionally flipped with left_positive flag
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u_u, v_u = undistort_pixel(u, v, K, dist)

    dv = (v_u - cy)
    if abs(dv) < 1e-6:
        raise ZeroDivisionError("Pixel too close to horizon/principal row for stable distance estimate.")

    # Magnitude of forward distance
    x_car = abs(H * fy / dv)

    # Lateral distance
    y_car = (u_u - cx) * x_car / fx

    # Optional convention: image right = car right, which often means left is negative
    # If left positive, flip sign here
    if left_positive:
        y_car = -y_car

    return float(x_car), float(y_car)


# =========================
# Visualization
# =========================

def draw_point_and_text(image, point, text):
    cv = require_cv2()
    out = image.copy()
    cv.circle(out, point, 6, (0, 0, 255), -1)
    cv.putText(
        out,
        text,
        (point[0] + 10, point[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )
    return out


# =========================
# Main
# =========================

def main():
    cv = require_cv2()
    print("Starting distance measurement pipeline...")

    # 1) Camera calibration
    K, dist = calibrate_camera(CALIB_DIR)

    # 2) Load known image and click lower-right corner of nearest red cone
    known_img = cv.imread(KNOWN_IMAGE_PATH)
    if known_img is None:
        raise FileNotFoundError(f"Could not read {KNOWN_IMAGE_PATH}")

    print("\nFor the known image, click the LOWER-RIGHT CORNER of the nearest red cone.")
    known_pixel = get_click_point(known_img, "Known cone: click lower-right corner")

    # 3) Estimate camera height
    H = estimate_camera_height(known_pixel, K, dist, KNOWN_X_METERS)

    # 4) Load unknown image and click corresponding cone point
    unknown_img = cv.imread(UNKNOWN_IMAGE_PATH)
    if unknown_img is None:
        raise FileNotFoundError(f"Could not read {UNKNOWN_IMAGE_PATH}")

    print("\nFor the unknown image, click the LOWER-RIGHT CORNER of the nearest red cone.")
    unknown_pixel = get_click_point(unknown_img, "Unknown cone: click lower-right corner")

    # 5) Convert unknown pixel to real-world car coordinates
    x_car, y_car = pixel_to_car(
        unknown_pixel[0],
        unknown_pixel[1],
        K,
        dist,
        H,
        left_positive=LEFT_POSITIVE
    )

    print("\n=== Final Result ===")
    print(f"Unknown cone pixel: {unknown_pixel}")
    print(f"Estimated x_car = {x_car:.4f} m")
    print(f"Estimated y_car = {y_car:.4f} m")

    # 6) Show annotated images
    known_vis = draw_point_and_text(
        known_img,
        known_pixel,
        f"known x = {KNOWN_X_METERS:.2f} m"
    )
    unknown_vis = draw_point_and_text(
        unknown_img,
        unknown_pixel,
        f"x={x_car:.3f} m, y={y_car:.3f} m"
    )

    cv.namedWindow("Known Image Result", cv.WINDOW_NORMAL)
    cv.namedWindow("Unknown Image Result", cv.WINDOW_NORMAL)
    cv.imshow("Known Image Result", known_vis)
    cv.imshow("Unknown Image Result", unknown_vis)

    print("\nPress any key in an image window to close.")
    cv.waitKey(0)
    cv.destroyAllWindows()

    CALIBRATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(CALIBRATION_RESULTS_PATH),
        K=K,
        dist=dist,
        H=H,
        x_car=round(x_car, 4),
        y_car=round(y_car, 4),
    )
    print(f"Saved calibration results to {CALIBRATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()
    cv = require_cv2()
