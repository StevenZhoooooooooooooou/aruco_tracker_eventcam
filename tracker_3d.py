import cv2
import numpy as np
import json
import os
from metavision_core.event_io import EventsIterator

# ================= CONFIGURATION =================
CONFIG_FILE = "/home/raspi5/settings.json"
MARKER_TEMPLATE = "/home/raspi5/Desktop/event_cam/ArUco_track/marker_9_edges.png"

# Detection Parameters
# MIN_AREA: Minimum contour area in pixels. Smaller values detect smaller/tilted markers.
# MAX_AREA: Maximum contour area in pixels. Filters out very large false detections.
# MATCH_THRESHOLD: Template matching score threshold (0-1). Lower values are more lenient.
MIN_AREA = 400
MAX_AREA = 150000
MATCH_THRESHOLD = 0.25

# Shape Constraints
# MAX_ASPECT_RATIO: Maximum ratio of longest/shortest side. Controls how stretched the quad can be.
# MIN_ANGLE: Minimum interior angle in degrees. Prevents very sharp corners.
# MAX_ANGLE: Maximum interior angle in degrees. Prevents very flat corners.
MAX_ASPECT_RATIO = 2.5
MIN_ANGLE = 30
MAX_ANGLE = 150

# Locking Parameters
# LOCK_SEARCH_RADIUS: Pixel radius to search for marker after locked.
# LOCK_CONFIRM_FRAMES: Number of consecutive frames needed to confirm and lock a target.
# STATIC_THRESHOLD: Event count below this is considered static (no movement).
# MAX_SIZE_CHANGE: Maximum allowed area ratio change between frames when locked.
LOCK_SEARCH_RADIUS = 100
LOCK_CONFIRM_FRAMES = 3
STATIC_THRESHOLD = 30
MAX_SIZE_CHANGE = 2.0

# 3D Pose Estimation Parameters
# MARKER_SIZE: Physical size of the marker in meters (e.g., 0.08 = 8cm).
MARKER_SIZE = 0.08

# Camera Intrinsic Parameters (320x320 resolution)
# CAMERA_MATRIX: 3x3 matrix containing focal length (fx, fy) and principal point (cx, cy).
# DIST_COEFFS: Distortion coefficients. Set to zero for event cameras with minimal distortion.
CAMERA_MATRIX = np.array([
    [320, 0, 160],
    [0, 320, 160],
    [0, 0, 1]
], dtype=np.float32)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)


def load_camera_config(iterator, config_path):
    """
    Load camera bias settings from JSON configuration file.
    Adjusts event camera sensitivity parameters.
    """
    if not os.path.exists(config_path):
        return
    try:
        if hasattr(iterator, 'reader'):
            device = iterator.reader.device
            biases = device.get_i_ll_biases()
            with open(config_path, 'r') as f:
                conf = json.load(f)
            if "ll_biases_state" in conf:
                for item in conf["ll_biases_state"]["bias"]:
                    try:
                        biases.set(item["name"], item["value"])
                    except:
                        pass
    except:
        pass


def calc_angle(p1, p2, p3):
    """
    Calculate angle at point p2 formed by points p1-p2-p3.
    Returns angle in degrees.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.degrees(np.arccos(cos_angle))

    return angle


def is_valid_quad(corners):
    """
    Check if quadrilateral has reasonable shape.
    Validates interior angles and side length ratios.
    Returns True if shape is acceptable.
    """
    pts = corners.reshape(4, 2)

    # Check all 4 interior angles
    angles = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        angle = calc_angle(p1, p2, p3)
        angles.append(angle)

    # All angles must be within range
    for angle in angles:
        if angle < MIN_ANGLE or angle > MAX_ANGLE:
            return False

    # Check side length ratios
    sides = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        length = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        sides.append(length)

    if min(sides) < 1:
        return False

    side_ratio = max(sides) / min(sides)
    if side_ratio > MAX_ASPECT_RATIO:
        return False

    return True


def find_rectangles(contours):
    """
    Filter contours to find quadrilateral candidates that could be markers.
    Applies shape validation to reject distorted quads.
    Returns list of valid candidates.
    """
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)

        if len(approx) != 4:
            continue

        # Must be convex
        if not cv2.isContourConvex(approx):
            continue

        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / min(w, h)
        if aspect > MAX_ASPECT_RATIO:
            continue

        corners = approx.reshape(4, 2).astype(np.float32)

        # Validate quadrilateral shape
        if not is_valid_quad(corners):
            continue

        corners = order_corners(corners)

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            candidates.append({
                'center': (cx, cy),
                'area': area,
                'rect': rect,
                'corners': corners
            })

    return candidates


def order_corners(pts):
    """
    Sort 4 corner points in order: top-left, top-right, bottom-right, bottom-left.
    Required for consistent pose estimation.
    """
    pts = pts.astype(np.float32)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top = sorted_by_y[:2][np.argsort(sorted_by_y[:2, 0])]
    bottom = sorted_by_y[2:][np.argsort(sorted_by_y[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def estimate_pose(corners_2d):
    """
    Estimate 3D position of marker using PnP algorithm.
    Returns translation vector (x, y, z) in meters and success flag.
    """
    half = MARKER_SIZE / 2.0
    corners_3d = np.array([
        [-half, -half, 0], [half, -half, 0],
        [half, half, 0], [-half, half, 0]
    ], dtype=np.float32)

    corners_2d = corners_2d.reshape(-1, 1, 2).astype(np.float32)
    success, rvec, tvec = cv2.solvePnP(
        corners_3d, corners_2d, CAMERA_MATRIX, DIST_COEFFS,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    return (tvec.flatten(), success) if success else (None, False)


def extract_roi(image, rect):
    """
    Extract rotated region of interest from image based on minimum area rectangle.
    Returns cropped and aligned ROI for template matching.
    """
    center, size, angle = rect
    w, h = int(size[0]), int(size[1])
    if w < 15 or h < 15:
        return None

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    cx, cy = int(center[0]), int(center[1])
    x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
    x2, y2 = min(image.shape[1], cx + w // 2), min(image.shape[0], cy + h // 2)

    return rotated[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def match_pattern(roi, template):
    """
    Match ROI against template using normalized cross-correlation.
    Tests 4 rotations and flips to handle different orientations.
    Returns best match score (0-1).
    """
    if roi is None or roi.size == 0:
        return 0

    h, w = roi.shape[:2]
    if h < 15 or w < 15:
        return 0

    try:
        roi_r = cv2.resize(roi, (50, 50))
        tmpl_r = cv2.resize(template, (50, 50))
    except:
        return 0

    _, roi_bin = cv2.threshold(roi_r, 30, 255, cv2.THRESH_BINARY)
    _, tmpl_bin = cv2.threshold(tmpl_r, 30, 255, cv2.THRESH_BINARY)

    if np.sum(roi_bin > 0) < 50:
        return 0

    best = 0
    for k in range(4):
        rot = np.rot90(tmpl_bin, k)
        result = cv2.matchTemplate(roi_bin, rot, cv2.TM_CCOEFF_NORMED)
        best = max(best, result.max())

        flip_h = cv2.flip(rot, 1)
        result = cv2.matchTemplate(roi_bin, flip_h, cv2.TM_CCOEFF_NORMED)
        best = max(best, result.max())

    return best


def dist(p1, p2):
    """
    Calculate Euclidean distance between two 2D points.
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_quad(img, corners, color, thickness=2):
    """
    Draw quadrilateral on image using 4 corner points.
    """
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, color, thickness)


def main():
    """
    Main tracking loop:
    1. Capture events from camera
    2. Find quadrilateral candidates
    3. Match against template to identify marker
    4. Lock onto marker and track position
    5. Estimate 3D pose and output coordinates
    """
    template = cv2.imread(MARKER_TEMPLATE, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("Template not found")
        return
    _, template = cv2.threshold(template, 30, 255, cv2.THRESH_BINARY)

    print("'q' quit, 'r' reset, 's' set origin")

    mv_iterator = EventsIterator(input_path="", delta_t=10000, mode="delta_t")
    load_camera_config(mv_iterator, CONFIG_FILE)
    h, w = mv_iterator.get_size()

    cv2.namedWindow("Marker", cv2.WINDOW_NORMAL)

    locked = False
    locked_center = None
    locked_corners = None
    locked_area = None
    locked_rect = None

    origin = None

    candidate_center = None
    candidate_corners = None
    candidate_area = None
    candidate_rect = None
    confirm_count = 0

    for evs in mv_iterator:
        im = np.zeros((h, w), dtype=np.uint8)
        if evs.size > 0:
            im[evs['y'], evs['x']] = 255

        im_display = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        if evs.size < STATIC_THRESHOLD:
            if locked:
                draw_quad(im_display, locked_corners, (0, 200, 200), 2)
                cv2.circle(im_display, locked_center, 4, (0, 0, 255), -1)
            cv2.imshow("Marker", im_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        kernel = np.ones((2, 2), np.uint8)
        im_clean = cv2.dilate(im, kernel, iterations=1)

        contours, _ = cv2.findContours(im_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = find_rectangles(contours)

        if locked:
            best = None
            best_score = 0
            for c in candidates:
                if dist(c['center'], locked_center) > LOCK_SEARCH_RADIUS:
                    continue
                ratio = c['area'] / locked_area
                if not (1 / MAX_SIZE_CHANGE < ratio < MAX_SIZE_CHANGE):
                    continue
                score = match_pattern(extract_roi(im_clean, c['rect']), template)
                if score > best_score and score > MATCH_THRESHOLD * 0.7:
                    best_score = score
                    best = c

            if best:
                locked_center = best['center']
                locked_corners = best['corners']
                locked_area = best['area']
                locked_rect = best['rect']

            draw_quad(im_display, locked_corners, (0, 255, 0), 2)
            cv2.circle(im_display, locked_center, 4, (0, 0, 255), -1)

            tvec, ok = estimate_pose(locked_corners)
            if ok:
                x, y, z = tvec * 100
                if origin is not None:
                    dx, dy, dz = (tvec - origin) * 1000
                    print(f"\rdX:{dx:+.1f} dY:{dy:+.1f} dZ:{dz:+.1f}mm  ", end="")
                else:
                    print(f"\rX:{x:.1f} Y:{y:.1f} Z:{z:.1f}cm  ", end="")

        else:
            best = None
            best_score = 0
            for c in candidates:
                score = match_pattern(extract_roi(im_clean, c['rect']), template)
                if score > best_score and score > MATCH_THRESHOLD:
                    best_score = score
                    best = c

            if best:
                if candidate_center and dist(best['center'], candidate_center) < 50:
                    confirm_count += 1
                else:
                    confirm_count = 1

                candidate_center = best['center']
                candidate_corners = best['corners']
                candidate_area = best['area']
                candidate_rect = best['rect']

                if confirm_count >= LOCK_CONFIRM_FRAMES:
                    locked = True
                    locked_center = candidate_center
                    locked_corners = candidate_corners
                    locked_area = candidate_area
                    locked_rect = candidate_rect
                    confirm_count = 0
                else:
                    draw_quad(im_display, candidate_corners, (255, 150, 0), 1)
            else:
                candidate_center = None
                confirm_count = 0

        cv2.imshow("Marker", im_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            locked = False
            locked_center = None
            candidate_center = None
            confirm_count = 0
            origin = None
            print("\nReset")
        elif key == ord('s') and locked:
            tvec, ok = estimate_pose(locked_corners)
            if ok:
                origin = tvec
                print(f"\nOrigin set")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()