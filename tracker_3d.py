import cv2
import numpy as np
import time
from metavision_core.event_io import EventsIterator

# ================= SYSTEM CONFIGURATION =================

# Physical parameters for PnP solver
MARKER_SIZE = 0.02      # Physical size of the marker in meters
CAMERA_MATRIX = np.array([
    [320, 0, 160],
    [0, 320, 160],
    [0, 0, 1]
], dtype=np.float32)    # Intrinsic matrix: [fx, 0, cx], [0, fy, cy], [0, 0, 1]
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32) # Distortion coefficients

# Event processing parameters
DELTA_T = 10000         # Integration time window in microseconds (10ms)
DECAY_FACTOR = 0.90     # Leaky integrator decay rate (0.0-1.0); higher = longer trails
MEMORY_TIME = 2.0       # Time in seconds to hold last valid position signal

class PositionLatch:
    """
    Implements a zero-order hold to maintain position state during signal loss.
    """
    def __init__(self, timeout=1.0):
        self.last_pos = None
        self.last_seen_time = 0
        self.timeout = timeout

    def update(self, pos):
        """
        Updates state with new observation or returns latched state if within timeout.
        Returns: (position, is_realtime: bool)
        """
        if pos is not None:
            self.last_pos = pos
            self.last_seen_time = time.time()
            return pos, True
        else:
            if time.time() - self.last_seen_time < self.timeout:
                return self.last_pos, False
            return None, False

def main():
    # --- 1. Initialize ArUco Detector ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    
    # Use SUBPIX refinement for stability on noisy event edges
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # Allow detection of small markers (min 5% of image perimeter)
    params.minMarkerPerimeterRate = 0.05

    # --- 2. Initialize Event Stream & Buffer ---
    mv_iterator = EventsIterator(input_path="", delta_t=DELTA_T, mode="delta_t")
    h, w = mv_iterator.get_size()
    
    # Accumulation buffer (Float32 for precision decay)
    canvas = np.full((h, w), 127.0, dtype=np.float32)
    
    # State latchers for target IDs
    latch_6 = PositionLatch(timeout=MEMORY_TIME)
    latch_7 = PositionLatch(timeout=MEMORY_TIME)

    print("System Ready. Active tracking initialized.")

    for evs in mv_iterator:
        # --- Image Reconstruction (Leaky Integrator) ---
        # Apply exponential decay to simulate visual persistence
        canvas = canvas * DECAY_FACTOR + 127 * (1 - DECAY_FACTOR)

        if evs.size > 0:
            x, y, p = evs['x'], evs['y'], evs['p']
            
            # Update polarity: ON=+50, OFF=-50
            update_val = np.zeros_like(x, dtype=np.float32)
            update_val[p == 1] = 50.0
            update_val[p == 0] = -50.0
            np.add.at(canvas, (y, x), update_val)

        # Clip to uint8 range for OpenCV
        img_display = np.clip(canvas, 0, 255).astype(np.uint8)

        # Optional: Gaussian blur to reduce high-frequency noise
        img_blur = cv2.GaussianBlur(img_display, (3, 3), 0)
        img_color = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR)

        # --- Marker Detection & Pose Estimation ---
        corners, ids, _ = cv2.aruco.detectMarkers(img_blur, aruco_dict, parameters=params)

        raw_pos_6 = None
        raw_pos_7 = None

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS)
            cv2.aruco.drawDetectedMarkers(img_color, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == 6: raw_pos_6 = tvecs[i][0]
                if marker_id == 7: raw_pos_7 = tvecs[i][0]

        # --- Position Latching & Distance Calculation ---
        pos_6, is_realtime_6 = latch_6.update(raw_pos_6)
        pos_7, is_realtime_7 = latch_7.update(raw_pos_7)

        if pos_6 is not None and pos_7 is not None:
            dist_cm = np.linalg.norm(pos_6 - pos_7) * 100
            
            # Status: LIVE (Green) or MEM (Yellow)
            is_live = is_realtime_6 and is_realtime_7
            color = (0, 255, 0) if is_live else (0, 255, 255)
            status_text = "LIVE" if is_live else "MEM"

            cv2.putText(img_color, f"Dist: {dist_cm:.2f}cm [{status_text}]", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            print(f"\rDist: {dist_cm:.2f} cm ({status_text})   ", end="")

        cv2.imshow("Event Tracker", img_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
