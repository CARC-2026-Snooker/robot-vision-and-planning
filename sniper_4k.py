import cv2
import json
import math
import sys
import time
import threading
import numpy as np
from collections import deque
from pupil_apriltags import Detector

# ─────────────────────────────────────────────────────────────────────────────
# TAG / POCKET DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
POCKET_TO_TAG = {
    "Hole_3": 3, "Hole_4": 4, "Hole_5": 5, "Hole_2": 2, "Hole_1": 1
}
POCKETS_XY = {
    "Hole_3": (0, 0), "Hole_4": (600, 0), "Hole_5": (1200, 0),
    "Hole_2": (0, 1050), "Hole_1": (1200, 1050)
}
KNOWN_TAGS = {
    1: {"x": 1.262, "y": 1.05, "facing_angle": 270},
    2: {"x": -0.062, "y": 1.05, "facing_angle": 270},
    3: {"x": 0,   "y": 0,    "facing_angle": 270},
    4: {"x": 0.6, "y": 0,    "facing_angle": 270},
    5: {"x": 1.2, "y": 0,    "facing_angle": 270},
}
TAG_COLORS = {
    1: (255, 105, 180),
    2: (0, 255, 255),
    3: (0, 165, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
}

# ─────────────────────────────────────────────────────────────────────────────
# MAP CONSTANTS  (identical to new_main3.py)
# ─────────────────────────────────────────────────────────────────────────────
ARENA_BOUNDS  = (0.0, 0.0, 1.2, 3.6)
SCALE         = 200
PADDING       = 60
CANVAS_W = int((ARENA_BOUNDS[2] - ARENA_BOUNDS[0]) * SCALE) + 2 * PADDING
CANVAS_H = int((ARENA_BOUNDS[3] - ARENA_BOUNDS[1]) * SCALE) + 2 * PADDING

# ─────────────────────────────────────────────────────────────────────────────
# HD TRACKING CONSTANTS  (same as new_main3.py — tuned for position)
# ─────────────────────────────────────────────────────────────────────────────
MAX_TAG_DISTANCE    = 10.0
OUT_OF_BOUNDS_LEEWAY = 0.15
MAX_POSE_JUMP       = 0.5
TRACKING_TIMEOUT    = 0.3
CROP_TOP_PCT        = 0.20
CROP_BOT_PCT        = 0.20

# Sniper-mode thresholds — slightly tighter than driving because we're stationary
BASE_STILL_THRESHOLD = 0.003   # was 0.005 in new_main3 — sniper is closer/stationary
BASE_MOVE_THRESHOLD  = 0.020   # was 0.030
STILL_ALPHA          = 0.005   # same
MOVE_ALPHA           = 0.30    # slightly faster than 0.25 so it follows when pushed

# Consensus weighting: same as new_main3.py — see comments there
CONSENSUS_SIGMA = 0.045   # 45mm soft-rejection radius for outlier tag estimates

# ─────────────────────────────────────────────────────────────────────────────
# SHOT MODE
# Balls A, B, E have a 0° cut angle — the robot fires in a straight line through
# the ghost ball and into the pocket. For these shots, when the 4K angle
# correction = 0, the robot is ALSO at the correct X position (proven by
# geometry: any X shift changes the angle to the pocket tag). The HD X/Y
# measurement is therefore redundant and noisier than the 4K angle. We skip it.
#
# For all other balls (cut angle > threshold), the HD X/Y IS needed — the robot
# could have the right angle to the tag but be at the wrong X position.
# ─────────────────────────────────────────────────────────────────────────────
STRAIGHT_THRESHOLD = 5.0   # degrees — below this, use straight-line mode

# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Corner pocket offset — STRAIGHT shots at Hole_3 (left) and Hole_5 (right) only.
# Shooting from a shallow angle into a corner risks clipping the side wall.
# This shifts the target alignment point away from the tag center toward the
# open part of the pocket.
#   Hole_3 (left corner)  → offset to the RIGHT  (+)
#   Hole_5 (right corner) → offset to the LEFT   (-)
# Units: mm.  Set to 0 to disable.
CORNER_OFFSET_MM = 25.0   # 0 to ~50mm typical range

# Pixel tolerance for lateral lock (at full camera resolution).
# ~2–7mm accuracy depending on shot geometry.
LATERAL_TOL_PX = 10

# Exposure brightness compensation.
# When manual exposure is set very low to reduce motion blur, the image gets dark.
# This multiplier is applied to every captured HD frame before display and tracking.
# 1.0 = no change. 1.5–2.5 = typical useful range for low-exposure compensation.
# Also applied in new_main3.py via the same constant name (keep them in sync).
# HD_BRIGHTNESS_BOOST = 1.8   # adjust to taste
HD_BRIGHTNESS_BOOST = 1.0   # adjust to taste
BALL_HSV = {
    "Red":  [(np.array([0, 50, 20]),   np.array([15, 255, 255])),
             (np.array([165, 50, 20]), np.array([180, 255, 255]))],
    "Blue": [(np.array([65, 10, 10]),  np.array([170, 255, 255]))],
    # "Blue": [(np.array([80, 20, 10]),  np.array([170, 255, 255]))],
}

# ─────────────────────────────────────────────────────────────────────────────
# 4K ANGLE THREAD CONSTANTS
# Camera matrix from calibration script — verified values
# ─────────────────────────────────────────────────────────────────────────────
# Distortion coefficients from calibration
DIST_COEFFS = np.array([[-1.23467127e+00, 1.37940482e+01, -2.47010796e-02, 6.62621201e-03, -1.51346017e+02]])

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def world_to_pixel(x, y):
    px = int((x - ARENA_BOUNDS[0]) * SCALE) + PADDING
    py = int((y - ARENA_BOUNDS[1]) * SCALE) + PADDING
    return (px, py)

def draw_map_line(canvas, x1, y1, x2, y2, color=(255, 255, 255), thickness=2):
    cv2.line(canvas, world_to_pixel(x1, y1), world_to_pixel(x2, y2), color, thickness)

def get_global_camera_pose(tag_id, t, R):
    """Identical to new_main3.py — do not modify."""
    if tag_id not in KNOWN_TAGS: return None
    cam_pos_in_tag_frame = -np.transpose(R) @ t
    local_x = cam_pos_in_tag_frame[0][0]
    local_z = cam_pos_in_tag_frame[2][0]
    tag_info = KNOWN_TAGS[tag_id]
    theta = math.radians(tag_info["facing_angle"])
    gx = tag_info["x"] + (local_z * math.cos(theta)) - (local_x * math.sin(theta))
    gy = tag_info["y"] + (local_z * math.sin(theta)) + (local_x * math.cos(theta))
    tag_space_dx, tag_space_dz = R[2, 0], R[2, 2]
    gdx = (tag_space_dz * math.cos(theta)) - (tag_space_dx * math.sin(theta))
    gdy = (tag_space_dz * math.sin(theta)) + (tag_space_dx * math.cos(theta))
    return (gx, gy, gdx, gdy)


# ─────────────────────────────────────────────────────────────────────────────
# 4K CAMERA THREAD
# Runs independently so its ~1fps decode never blocks the 30fps HD loop.
# Produces: angle correction, raw display frame (resized), tag found flag.
# ─────────────────────────────────────────────────────────────────────────────
class Sniper4kThread(threading.Thread):
    def __init__(self, target_tag_id, target_relative_angle, ball_color="Red", corner_offset_px=0.0):
        super().__init__(daemon=True)
        self.target_tag_id         = target_tag_id
        self.target_relative_angle = target_relative_angle
        self.ball_color            = ball_color
        self.corner_offset_px      = corner_offset_px   # pixels from tag center to target line
        self.tag_size              = 0.11

        # Shared results (written by thread, read by main loop)
        self._lock         = threading.Lock()
        self._correction   = 999.0
        self._tag_found    = False
        self._frame_720p   = None    # 4K view resized to 720p, with all overlays baked in
        self._mask_720p    = None    # colour mask at 720p (BGR, ready to blend)
        self._lateral_px   = None    # ball_centroid_x - tag_center_x at full res (or None)

        self.running  = True
        self.detector = Detector(families="tag36h11")

        if sys.platform.startswith('win'):
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  3840)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  2560)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Read actual delivered resolution so params scale correctly
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_w = actual_w
        self._actual_h = actual_h
        # base_4k  = [6280.83, 6558.89, 811.37, 499.18]
        base_4k  = [4534.22, 4528.08, 979.27, 775.85]
        scale    = actual_h / 1080.0
        self.camera_params_4k = [p * scale for p in base_4k]

        # Build undistort map once at actual resolution
        fx, fy, cx, cy = self.camera_params_4k
        cam_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            cam_matrix, DIST_COEFFS, None, cam_matrix,
            (actual_w, actual_h), cv2.CV_16SC2
        )
        print(f"[4K Thread] {actual_w}x{actual_h}, scale={scale:.2f}, "
              f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    def _get_ball_mask(self, frame_bgr):
        """Return (mask, centroid_or_None). mask is a binary uint8 image."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        ranges = BALL_HSV.get(self.ball_color, BALL_HSV["Red"])
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        # Light morphology to clean noise without destroying small detections
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Find centroid of largest blob
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroid = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 150:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return mask, centroid

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Undistort
            frame_ud = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

            # ── AprilTag detection ────────────────────────────────────────
            gray    = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2GRAY)
            results = self.detector.detect(gray, estimate_tag_pose=True,
                                           camera_params=self.camera_params_4k,
                                           tag_size=self.tag_size)

            found      = False
            correction = 999.0
            tag_cx     = None   # tag center X in full-res pixels
            tag_cy     = None
            draw_frame = frame_ud.copy()

            for tag in results:
                if tag.tag_id != self.target_tag_id:
                    continue
                found = True
                t_x = tag.pose_t[0][0]
                t_z = tag.pose_t[2][0]
                actual_angle = math.degrees(math.atan2(t_x, t_z))
                correction   = actual_angle - self.target_relative_angle

                # Draw tag bounding box
                corners = tag.corners.astype(int)
                for i in range(4):
                    cv2.line(draw_frame, tuple(corners[i]),
                             tuple(corners[(i+1) % 4]), (0, 0, 255), 8)

                tag_cx = int(tag.center[0])
                tag_cy = int(tag.center[1])
                cv2.circle(draw_frame, (tag_cx, tag_cy), 14, (0, 255, 255), 4)

                # Vertical guide line through tag center (cyan = tag center)
                cv2.line(draw_frame, (tag_cx, 0), (tag_cx, self._actual_h),
                         (0, 200, 255), 3)

                # Offset target line (magenta) — where the ball should align to
                if self.corner_offset_px != 0:
                    target_x = int(tag_cx + self.corner_offset_px)
                    cv2.line(draw_frame, (target_x, 0), (target_x, self._actual_h),
                             (255, 0, 255), 4)
                    cv2.putText(draw_frame, "TARGET LINE",
                                (target_x + 10, self._actual_h // 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)
                break

            # ── Ball colour mask ──────────────────────────────────────────
            ball_mask_binary, ball_centroid = self._get_ball_mask(frame_ud)

            # Coloured mask overlay (semi-transparent, 40% opacity)
            # Masked pixels shown in colour, rest darkened slightly
            mask_colour = (0, 0, 255) if self.ball_color == "Red" else (255, 100, 0)
            mask_overlay = draw_frame.copy()
            mask_overlay[ball_mask_binary > 0] = mask_colour
            draw_frame = cv2.addWeighted(draw_frame, 0.6, mask_overlay, 0.4, 0)

            if ball_centroid is not None:
                bx, by = ball_centroid
                cv2.circle(draw_frame, (bx, by), 20, (0, 255, 0), 4)
                cv2.putText(draw_frame, "BALL", (bx + 25, by),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                # If tag visible, draw horizontal line connecting ball to tag guide
                if tag_cx is not None:
                    cv2.line(draw_frame, (bx, by), (tag_cx, by), (0, 255, 128), 3)

            # Lateral pixel error: ball X minus tag X (full-res pixels)
            # Positive → ball is RIGHT of tag → robot LEFT of line → MOVE RIGHT
            # Negative → ball is LEFT of tag  → robot RIGHT of line → MOVE LEFT
            lateral_px = None
            if ball_centroid is not None and tag_cx is not None:
                lateral_px = ball_centroid[0] - tag_cx

            # ── Downscale to half-res for display ────────────────────────
            # Use actual_w//2 × actual_h//2 so the window fits correctly at
            # any capture resolution (1080p, 1440p, 4K, etc.)
            disp_w = self._actual_w // 2
            disp_h = self._actual_h // 2
            frame_small = cv2.resize(draw_frame, (disp_w, disp_h))
            self._disp_w = disp_w
            self._disp_h = disp_h

            # Also build a standalone colour mask image at display resolution
            disp_w = self._actual_w // 2
            disp_h = self._actual_h // 2
            mask_bgr_small = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
            mask_bgr_small[cv2.resize(ball_mask_binary, (disp_w, disp_h)) > 0] = mask_colour

            with self._lock:
                self._correction = correction
                self._tag_found  = found
                self._frame_720p = frame_small
                self._mask_720p  = mask_bgr_small
                self._lateral_px = lateral_px

    def get_state(self):
        with self._lock:
            return (self._correction, self._tag_found,
                    self._frame_720p, self._mask_720p,
                    self._lateral_px)

    def stop(self):
        self.running = False
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load shot plan ──────────────────────────────────────────────────────
    try:
        with open("selected_target.json", "r") as f:
            target_data = json.load(f)
    except FileNotFoundError:
        print("Error: No selected_target.json found.")
        return

    target_pocket   = target_data["target_pocket"]
    target_tag_id   = POCKET_TO_TAG[target_pocket]
    fire_angle_deg  = target_data["firing_angle_deg"]
    rx, ry          = target_data["robot_alignment_xy"]   # mm, from planner
    px, py          = POCKETS_XY[target_pocket]           # original pocket center, mm
    cut_angle_deg   = target_data.get("cut_angle_deg", 90.0)
    ball_color      = target_data.get("color", "Red")
    ball_xy         = target_data["target_ball_real_xy"]  # [ball_x, ball_y] in mm

    # ── Shot mode ────────────────────────────────────────────────────────────
    SHOT_MODE = "STRAIGHT" if cut_angle_deg < STRAIGHT_THRESHOLD else "CUT"
    print(f"\n[Sniper] Shot mode: {SHOT_MODE}  (cut_angle={cut_angle_deg:.1f}°)")

    # ── target_relative_angle — ORIGINAL working formula ─────────────────────
    # The detector measures actual_relative_angle = atan2(t_x, t_z).
    # This is the angle of the tag from the camera's optical axis.
    # target_relative_angle encodes where the tag SHOULD appear in camera frame
    # when the robot is correctly positioned and facing the firing direction.
    #
    # Original formula (proven working before any offset changes):
    #   dx = px - rx  (real pocket - robot plan position)
    #   dy = py - ry
    #   target_world_angle = atan2(dy, dx)   ← standard math convention
    #   target_relative_angle = target_world_angle - fire_angle_deg
    #
    # fire_angle_deg from the planner now encodes the OFFSET firing direction
    # (toward the effective pocket) for corner shots. Because of this, tra
    # automatically becomes the correct non-zero target that the camera will
    # measure when pointing at the effective pocket. No other changes needed.
    dx = px - rx
    dy = py - ry
    target_world_angle    = math.degrees(math.atan2(dy, dx))
    target_relative_angle = target_world_angle - fire_angle_deg
    target_relative_angle = (target_relative_angle + 180) % 360 - 180
    print(f"[Sniper] target_relative_angle: {target_relative_angle:.4f}°")

    # Lateral sensitivity geometry inputs (depth values for STRAIGHT mode)
    tag_world_y_mm = KNOWN_TAGS[target_tag_id]["y"] * 1000.0
    ball_depth_mm  = ry - ball_xy[1]         # ry - ball_y_mm
    tag_depth_mm   = ry - tag_world_y_mm     # ry - tag_y_mm
    rx_m = rx / 1000.0                       # metres, for HD X-error display

    # ── Start 4K thread ─────────────────────────────────────────────────────
    sniper_thread = Sniper4kThread(target_tag_id, target_relative_angle, ball_color,
                                   corner_offset_px=0.0)
    sniper_thread.start()

    # Compute lateral sensitivity now that thread has opened the camera
    fx_4k = sniper_thread.camera_params_4k[0]
    if ball_depth_mm > 0 and tag_depth_mm > 0:
        lateral_sens   = fx_4k * (1.0/ball_depth_mm - 1.0/tag_depth_mm)
        lateral_acc_mm = LATERAL_TOL_PX / abs(lateral_sens)
    else:
        lateral_sens   = 1.0
        lateral_acc_mm = 99.9
    print(f"[Sniper] Lateral sensitivity: {abs(lateral_sens):.3f} px/mm  "
          f"(tol={LATERAL_TOL_PX}px → ±{lateral_acc_mm:.1f}mm)")

    # Compute corner_offset_px using actual fx, then update thread's guide line
    CORNER_TAGS_SIGN = {3: +1, 5: -1}   # Hole_3 = left corner (+X), Hole_5 = right (−X)
    if SHOT_MODE == "STRAIGHT" and target_tag_id in CORNER_TAGS_SIGN and CORNER_OFFSET_MM != 0:
        corner_sign       = CORNER_TAGS_SIGN[target_tag_id]
        target_lateral_px = corner_sign * (fx_4k * CORNER_OFFSET_MM / tag_depth_mm)
        sniper_thread.corner_offset_px = target_lateral_px
        print(f"[Sniper] Corner offset px: {target_lateral_px:+.1f} px  "
              f"({'RIGHT' if corner_sign>0 else 'LEFT'} of tag center)")
    else:
        target_lateral_px = 0.0

    # ── HD camera (position — 30fps) ─────────────────────────────────────────
    if sys.platform.startswith('win'):
        cap_hd = cv2.VideoCapture(3, cv2.CAP_DSHOW)
    else:
        cap_hd = cv2.VideoCapture(3, cv2.CAP_V4L2)
    cap_hd.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap_hd.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap_hd.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_hd.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_hd.set(cv2.CAP_PROP_FPS, 30)
    cap_hd.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # setting exposure manually so less blur on movement
    # cap_hd.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0) 
    # cap_hd.set(cv2.CAP_PROP_EXPOSURE, -4.0) # Try values between -1 and -9 
    for _ in range(5):
        cap_hd.read()
        time.sleep(0.05)

    actual_h_hd  = int(cap_hd.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # base_hd = [1996.94*0.99444, 1998.60*0.99444, 827.66, 460.60]
    base_hd = [1924.70, 1924.67, 907.46, 483.50]
    scale_hd     = actual_h_hd / 1080.0
    camera_params_hd = [p * scale_hd for p in base_hd]

    detector_hd = Detector(families="tag36h11")
    tag_size_hd = 0.11

    # ── Position tracking state (mirrors new_main3.py exactly) ──────────────
    history_raw_x = deque(maxlen=15)
    history_raw_y = deque(maxlen=15)
    last_x, last_y, last_dx, last_dy = 0.0, 0.0, 1.0, 0.0
    initialized    = False
    last_seen_time = time.time()
    state_text     = "LOST"
    debug_tag_estimates = []

    print(f"\n[Sniper] Target pocket : {target_pocket}  (Tag {target_tag_id})")
    print(f"[Sniper] Required X    : {rx:.1f} mm")
    print(f"[Sniper] Required Y    : {ry:.1f} mm")
    print(f"[Sniper] Target angle  : {target_relative_angle:.2f}°")
    print(f"[Sniper] HD cam  → position (X, Y, heading) at 30fps")
    print(f"[Sniper] 4K cam  → angle correction (threaded, non-blocking)")

    while True:
        # ════════════════════════════════════════════════════════════════════
        # 1. HD FRAME — position tracking (runs at full 30fps)
        # ════════════════════════════════════════════════════════════════════
        ret_hd, frame_hd = cap_hd.read()
        if not ret_hd:
            print("HD camera lost.")
            break

        # Brightness compensation for low-exposure setting
        if HD_BRIGHTNESS_BOOST != 1.0:
            frame_hd = cv2.convertScaleAbs(frame_hd, alpha=HD_BRIGHTNESS_BOOST)

        h_hd = frame_hd.shape[0]
        y_start = int(h_hd * CROP_TOP_PCT)
        y_end   = int(h_hd * (1.0 - CROP_BOT_PCT))
        gray_hd = cv2.cvtColor(frame_hd, cv2.COLOR_BGR2GRAY)[y_start:y_end, :]
        adj_params_hd = [camera_params_hd[0], camera_params_hd[1],
                         camera_params_hd[2], camera_params_hd[3] - y_start]

        tags_hd = detector_hd.detect(gray_hd, estimate_tag_pose=True,
                                     camera_params=adj_params_hd, tag_size=tag_size_hd)
        debug_tag_estimates = []

        for tag in tags_hd:
            tag.corners[:, 1] += y_start
            tag.center[1]     += y_start
            t, R = tag.pose_t, tag.pose_R
            pose = get_global_camera_pose(tag.tag_id, t, R)
            if pose is None: continue
            dist = math.sqrt(t[0][0]**2 + t[1][0]**2 + t[2][0]**2)
            if dist > MAX_TAG_DISTANCE: continue
            gx, gy = pose[0], pose[1]
            if not (ARENA_BOUNDS[0] - OUT_OF_BOUNDS_LEEWAY <= gx <= ARENA_BOUNDS[2] + OUT_OF_BOUNDS_LEEWAY and
                    ARENA_BOUNDS[1] - OUT_OF_BOUNDS_LEEWAY <= gy <= ARENA_BOUNDS[3] + OUT_OF_BOUNDS_LEEWAY):
                continue
            if initialized and math.hypot(gx - last_x, gy - last_y) > MAX_POSE_JUMP:
                continue
            w = 1.0 / (dist**3 + 0.001)
            debug_tag_estimates.append({"id": tag.tag_id, "pose": pose, "dist": dist, "w": w})

        if debug_tag_estimates:
            last_seen_time = time.time()

            # ── Step 1: distance-weighted initial mean ─────────────────────
            init_w = sum(e["w"] for e in debug_tag_estimates)
            sum_dist = sum(e["dist"] for e in debug_tag_estimates)
            init_x = sum(e["pose"][0] * e["w"] for e in debug_tag_estimates) / init_w
            init_y = sum(e["pose"][1] * e["w"] for e in debug_tag_estimates) / init_w

            # ── Step 2: consensus soft-rejection ──────────────────────────
            total_w = raw_x = raw_y = raw_dx = raw_dy = 0.0
            for e in debug_tag_estimates:
                dev = math.hypot(e["pose"][0] - init_x, e["pose"][1] - init_y)
                consensus_w = math.exp(-dev**2 / (2 * CONSENSUS_SIGMA**2))
                w = e["w"] * consensus_w
                raw_x  += e["pose"][0] * w
                raw_y  += e["pose"][1] * w
                raw_dx += e["pose"][2] * w
                raw_dy += e["pose"][3] * w
                total_w += w
            raw_x /= total_w; raw_y /= total_w
            raw_dx /= total_w; raw_dy /= total_w
            mag = math.hypot(raw_dx, raw_dy)
            if mag > 0: raw_dx /= mag; raw_dy /= mag

            history_raw_x.append(raw_x)
            history_raw_y.append(raw_y)

            if not initialized:
                last_x, last_y, last_dx, last_dy = raw_x, raw_y, raw_dx, raw_dy
                initialized = True

            # ── Step 3: still/moving/transitioning with dist^2 scaling ────
            movement_metric = math.hypot(
                np.std(history_raw_x) if len(history_raw_x) >= 2 else 0.0,
                np.std(history_raw_y) if len(history_raw_y) >= 2 else 0.0
            )
            avg_dist = sum_dist / len(debug_tag_estimates)
            dist_mult = max(1.0, min(avg_dist ** 2, 12.0))
            still_thr = BASE_STILL_THRESHOLD * dist_mult
            move_thr  = BASE_MOVE_THRESHOLD  * dist_mult

            if movement_metric <= still_thr:
                alpha, state_text = STILL_ALPHA, "STILL"
            elif movement_metric >= move_thr:
                alpha, state_text = MOVE_ALPHA, "MOVING"
            else:
                ratio = (movement_metric - still_thr) / (move_thr - still_thr)
                alpha = STILL_ALPHA + ratio * (MOVE_ALPHA - STILL_ALPHA)
                state_text = "TRANSITIONING"

            last_x  = alpha * raw_x  + (1.0 - alpha) * last_x
            last_y  = alpha * raw_y  + (1.0 - alpha) * last_y
            last_dx = alpha * raw_dx + (1.0 - alpha) * last_dx
            last_dy = alpha * raw_dy + (1.0 - alpha) * last_dy
            mag_s = math.hypot(last_dx, last_dy)
            if mag_s > 0: last_dx /= mag_s; last_dy /= mag_s

        else:
            if initialized and (time.time() - last_seen_time > TRACKING_TIMEOUT):
                initialized = False
                history_raw_x.clear()
                history_raw_y.clear()
            state_text = "LOST"

        # ════════════════════════════════════════════════════════════════════
        # 2. FETCH LATEST 4K RESULTS  (non-blocking — last result from thread)
        # ════════════════════════════════════════════════════════════════════
        correction, tag_found_4k, frame_4k_720p, mask_720p, lateral_px = sniper_thread.get_state()

        # ════════════════════════════════════════════════════════════════════
        # 3. COMPUTE ERRORS
        # ════════════════════════════════════════════════════════════════════
        # HD position (always computed — used for CUT mode and info display)
        if initialized:
            x_error = last_x * 1000.0 - rx   # mm
            y_error = last_y * 1000.0 - ry   # mm
        else:
            x_error = 999.0
            y_error = 999.0

        ang_locked = tag_found_4k and abs(correction) < 0.3

        if SHOT_MODE == "STRAIGHT":
            # STEP 1 — ANGLE: correction ≈ 0  (TURN LEFT / TURN RIGHT)
            # STEP 2 — LATERAL: lateral_px ≈ target_lateral_px  (MOVE LEFT / MOVE RIGHT)
            #   lateral_px = ball_centroid_x - tag_center_x  (full-res pixels)
            #   target_lateral_px = 0 for center shots, ±offset for corner shots
            #   lateral_error_px = lateral_px - target_lateral_px
            #   > 0: need to MOVE RIGHT  (robot is left of ideal line)
            #   < 0: need to MOVE LEFT   (robot is right of ideal line)
            if lateral_px is not None:
                lateral_error_px = lateral_px - target_lateral_px
                lateral_mm = lateral_error_px / lateral_sens   # mm, + = MOVE RIGHT
                lat_locked = abs(lateral_error_px) < LATERAL_TOL_PX
            else:
                lateral_mm = None
                lat_locked = False
            pos_locked = ang_locked and lat_locked

        else:
            # CUT: HD X/Y + 4K angle each checked independently
            x_ok = initialized and abs(x_error) < 5.0
            y_ok = initialized and abs(y_error) < 10.0
            pos_locked = x_ok and y_ok
            lateral_mm = None
            lat_locked = False

        # ════════════════════════════════════════════════════════════════════
        # 4. DRAW 4K VIEW  (with HUD overlaid)
        # ════════════════════════════════════════════════════════════════════
        if frame_4k_720p is not None:
            hud = frame_4k_720p.copy()
            disp_h, disp_w = hud.shape[:2]
        else:
            disp_w, disp_h = 1280, 720
            hud = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
            cv2.putText(hud, "Waiting for 4K frame...", (50, disp_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 3)

        # ── Semi-transparent colour mask blended into the 4K view ───────────
        # This is already baked into frame_4k_720p by the thread (40% opacity).
        # Nothing extra to do here — the thread draws it on every frame.

        # ── Mode banner (top bar) ────────────────────────────────────────────
        mode_col   = (0, 220, 100) if SHOT_MODE == "STRAIGHT" else (0, 180, 255)
        offset_tag = f"  [corner +{CORNER_OFFSET_MM:.0f}mm]" if (SHOT_MODE=="STRAIGHT" and target_lateral_px!=0) else ""
        banner_txt = f"STRAIGHT SHOT  cut={cut_angle_deg:.0f}°  →  {target_pocket}{offset_tag}" \
                     if SHOT_MODE == "STRAIGHT" \
                else f"CUT SHOT  cut={cut_angle_deg:.0f}°  →  {target_pocket}"
        cv2.rectangle(hud, (0, 0), (1280, 50), (0, 0, 0), -1)
        cv2.putText(hud, banner_txt, (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, mode_col, 2)

        if SHOT_MODE == "STRAIGHT":
            # ────────────────────────────────────────────────────────────────
            # STRAIGHT MODE — two steps, shown top to bottom:
            #
            # STEP 1: ANGLE  (TURN LEFT / TURN RIGHT)
            # STEP 2: LATERAL  (MOVE LEFT / MOVE RIGHT — robot's own frame)
            #
            # Both must be green for the FIRE lock to appear.
            # ────────────────────────────────────────────────────────────────

            # ── STEP 1: Angle ────────────────────────────────────────────────
            step1_locked = ang_locked
            s1_col = (0, 255, 0) if step1_locked else (255, 255, 0)
            cv2.putText(hud, "STEP 1 — ANGLE",
                        (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 180, 180), 2)
            if not tag_found_4k:
                cv2.putText(hud, f"Searching for Tag {target_tag_id}...",
                            (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
            elif step1_locked:
                cv2.putText(hud, f"ANGLE OK  ({correction:+.2f}°)",
                            (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            else:
                d = "RIGHT" if correction > 0 else "LEFT"
                cv2.putText(hud, f"TURN {d}:  {abs(correction):.2f}°",
                            (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 5)

            # Divider
            cv2.line(hud, (20, 148), (660, 148), (60, 60, 60), 1)

            # ── STEP 2: Lateral ──────────────────────────────────────────────
            s2_col = (0, 255, 0) if lat_locked else (255, 255, 0)
            cv2.putText(hud, "STEP 2 — LATERAL  (after angle is set)",
                        (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 180, 180), 2)
            if lateral_mm is None:
                no_ball_msg = "Ball not detected" if tag_found_4k else "Waiting for tag..."
                cv2.putText(hud, no_ball_msg,
                            (30, 218), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 100, 100), 2)
            elif lat_locked:
                cv2.putText(hud, f"LATERAL OK  ({lateral_mm:+.1f} mm)",
                            (30, 218), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            else:
                # lateral_mm > 0 → ball right of tag → robot left of line → MOVE RIGHT
                d = "RIGHT" if lateral_mm > 0 else "LEFT"
                cv2.putText(hud, f"MOVE {d}:  {abs(lateral_mm):.1f} mm",
                            (30, 218), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 5)

            # ── Lock banner ──────────────────────────────────────────────────
            if pos_locked and ang_locked:
                cv2.rectangle(hud, (0, 238), (1280, 310), (0, 70, 0), -1)
                cv2.putText(hud, "LOCKED!  PRESS 'f' TO FIRE",
                            (30, 295), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)

            # HD state — tiny info at bottom
            track_col = (0,200,0) if state_text=="STILL" else (0,140,200) if state_text=="TRANSITIONING" else (0,0,180)
            hd_txt = f"HD: {state_text}  X={x_error:+.0f}mm  Y={y_error:+.0f}mm" if initialized else f"HD: {state_text}"
            cv2.putText(hud, hd_txt, (15, disp_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_col, 1)

        else:
            # ────────────────────────────────────────────────────────────────
            # CUT MODE HUD  (unchanged behaviour)
            # Shows X error, Y error, and angle error independently.
            # ────────────────────────────────────────────────────────────────
            track_col = (0,255,0) if state_text=="STILL" else (0,165,255) if state_text=="TRANSITIONING" else (0,0,255)
            cv2.putText(hud, f"HD: {state_text}",
                        (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, track_col, 2)

            x_col = (0, 255, 0) if (initialized and abs(x_error) < 5.0)  else (255, 255, 0)
            y_col = (0, 255, 0) if (initialized and abs(y_error) < 10.0) else (255, 255, 0)
            a_col = (0, 255, 0) if ang_locked else (255, 255, 0)

            cv2.putText(hud, f"X Error [HD]: {x_error:+.1f} mm" if initialized else "X Error [HD]: ---",
                        (30, 148), cv2.FONT_HERSHEY_SIMPLEX, 1.2, x_col, 3)
            cv2.putText(hud, f"Y Error [HD]: {y_error:+.1f} mm" if initialized else "Y Error [HD]: ---",
                        (30, 194), cv2.FONT_HERSHEY_SIMPLEX, 1.2, y_col, 3)
            cv2.putText(hud,
                        f"Angle  [4K]: {correction:+.2f}°" if tag_found_4k else "Angle [4K]: SEARCHING...",
                        (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, a_col, 3)

            y_off = 308
            if pos_locked and ang_locked:
                cv2.putText(hud, "LOCKED!  PRESS 'f' TO FIRE",
                            (30, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 5)
            else:
                if initialized and abs(x_error) >= 5.0:
                    d = "LEFT" if x_error > 0 else "RIGHT"
                    cv2.putText(hud, f"SHIFT {d}: {abs(x_error):.1f} mm",
                                (30, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 165, 255), 4)
                    y_off += 55
                if initialized and abs(y_error) >= 10.0:
                    d = "FORWARD" if y_error > 0 else "BACK"
                    cv2.putText(hud, f"MOVE {d}: {abs(y_error):.1f} mm",
                                (30, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 165, 255), 4)
                    y_off += 55
                if not ang_locked and tag_found_4k:
                    d = "RIGHT" if correction > 0 else "LEFT"
                    cv2.putText(hud, f"TURN {d}: {abs(correction):.2f}°",
                                (30, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 165, 255), 4)
                elif not tag_found_4k:
                    cv2.putText(hud, f"SEARCHING FOR TAG {target_tag_id}...",
                                (30, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
            if not initialized:
                cv2.putText(hud, "HD TRACKING LOST",
                            (30, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.namedWindow("4K Sniper View", cv2.WINDOW_NORMAL)
        cv2.imshow("4K Sniper View", hud)

        # ════════════════════════════════════════════════════════════════════
        # 5. DRAW 2D MAP  (identical style to new_main3.py)
        # ════════════════════════════════════════════════════════════════════
        map_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # Grid
        pt_tl = world_to_pixel(ARENA_BOUNDS[0], ARENA_BOUNDS[1])
        pt_br = world_to_pixel(ARENA_BOUNDS[2], ARENA_BOUNDS[3])
        cv2.rectangle(map_canvas, pt_tl, pt_br, (0, 150, 150), 2)
        for gs in np.arange(ARENA_BOUNDS[0], ARENA_BOUNDS[2] + 0.1, 0.5):
            cv2.line(map_canvas, world_to_pixel(gs, ARENA_BOUNDS[1]),
                     world_to_pixel(gs, ARENA_BOUNDS[3]), (40, 40, 40), 1)
        for gs in np.arange(ARENA_BOUNDS[1], ARENA_BOUNDS[3] + 0.1, 0.5):
            cv2.line(map_canvas, world_to_pixel(ARENA_BOUNDS[0], gs),
                     world_to_pixel(ARENA_BOUNDS[2], gs), (40, 40, 40), 1)

        # Field lines
        draw_map_line(map_canvas, 0.0, 2.1, 1.2, 2.1, (255, 255, 255), 2)
        draw_map_line(map_canvas, 0.0, 2.4, 1.2, 2.4, (0, 150, 0), 2)
        draw_map_line(map_canvas, 0.0, 2.5, 1.2, 2.5, (150, 150, 150), 2)

        # AprilTags
        for tid, info in KNOWN_TAGS.items():
            pt = world_to_pixel(info["x"], info["y"])
            col = TAG_COLORS.get(tid, (0, 0, 255))
            cv2.rectangle(map_canvas, (pt[0]-6, pt[1]-6), (pt[0]+6, pt[1]+6), col, -1)
            cv2.putText(map_canvas, f"ID {tid}", (pt[0]-14, pt[1]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        # HD debug lines
        for e in debug_tag_estimates:
            tid, pose = e["id"], e["pose"]
            if tid in KNOWN_TAGS:
                pt_tag  = world_to_pixel(KNOWN_TAGS[tid]["x"], KNOWN_TAGS[tid]["y"])
                pt_cam  = world_to_pixel(pose[0], pose[1])
                col = TAG_COLORS.get(tid, (255, 255, 255))
                cv2.line(map_canvas, pt_tag, pt_cam, col, 1)
                cv2.circle(map_canvas, pt_cam, 3, col, -1)
                pt_arr = world_to_pixel(pose[0] + pose[2]*0.15, pose[1] + pose[3]*0.15)
                cv2.arrowedLine(map_canvas, pt_cam, pt_arr, col, 1, tipLength=0.25)

        # Target + shot geometry on map
        tgt_pt    = world_to_pixel(rx / 1000.0, ry / 1000.0)
        ball_pt   = world_to_pixel(target_data["target_ball_real_xy"][0] / 1000.0,
                                   target_data["target_ball_real_xy"][1] / 1000.0)
        ghost_pt  = world_to_pixel(target_data["ghost_ball_xy"][0] / 1000.0,
                                   target_data["ghost_ball_xy"][1] / 1000.0)
        pocket_pt = world_to_pixel(px / 1000.0, py / 1000.0)

        ball_map_col = (0, 0, 255) if ball_color == "Red" else (255, 100, 0)
        cv2.circle(map_canvas, ball_pt,   7, ball_map_col, -1)
        cv2.circle(map_canvas, ghost_pt,  5, (200, 200, 200), 1)
        cv2.circle(map_canvas, pocket_pt, 7, (0, 200, 255), 2)
        cv2.line(map_canvas, ghost_pt, pocket_pt, (255, 200, 0), 1)
        zone_label = target_data.get("target_ball_zone", "?")
        cv2.putText(map_canvas, zone_label, (ball_pt[0]+9, ball_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ball_map_col, 1)

        # Mode-specific target markers
        mode_col_map = (0, 220, 100) if SHOT_MODE == "STRAIGHT" else (255, 0, 255)

        if SHOT_MODE == "STRAIGHT":
            # No target position circle or tolerance bands —
            # the robot just needs to get the angle right, not park at a specific spot.
            # Draw a line from the robot's current position toward the tag (aiming ray).
            tag_info = KNOWN_TAGS.get(target_tag_id)
            if tag_info and initialized:
                tag_map_pt = world_to_pixel(tag_info["x"], tag_info["y"])
                robot_map_pt = world_to_pixel(last_x, last_y)
                cv2.line(map_canvas, robot_map_pt, tag_map_pt, (0, 200, 255), 2)
        else:
            # CUT: show target position dot + X/Y tolerance bands
            cv2.circle(map_canvas, tgt_pt, 5, mode_col_map, -1)
            cv2.putText(map_canvas, "TARGET", (tgt_pt[0]+8, tgt_pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_col_map, 1)
            cv2.line(map_canvas,
                     world_to_pixel(rx/1000.0 - 0.005, ry/1000.0 - 0.02),
                     world_to_pixel(rx/1000.0 - 0.005, ry/1000.0 + 0.02), mode_col_map, 2)
            cv2.line(map_canvas,
                     world_to_pixel(rx/1000.0 + 0.005, ry/1000.0 - 0.02),
                     world_to_pixel(rx/1000.0 + 0.005, ry/1000.0 + 0.02), mode_col_map, 2)
            cv2.line(map_canvas,
                     world_to_pixel(rx/1000.0 - 0.025, ry/1000.0 - 0.010),
                     world_to_pixel(rx/1000.0 + 0.025, ry/1000.0 - 0.010), mode_col_map, 2)
            cv2.line(map_canvas,
                     world_to_pixel(rx/1000.0 - 0.025, ry/1000.0 + 0.010),
                     world_to_pixel(rx/1000.0 + 0.025, ry/1000.0 + 0.010), mode_col_map, 2)

        # Mode label on map
        cv2.putText(map_canvas, SHOT_MODE, (PADDING, CANVAS_H - PADDING + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_col_map, 2)

        # Aiming cone
        cone_len = 0.35
        a_min = math.radians(fire_angle_deg - 5.0)
        a_max = math.radians(fire_angle_deg + 5.0)
        cone_origin = world_to_pixel(last_x, last_y) if initialized else tgt_pt
        pt1 = world_to_pixel(rx/1000.0 + cone_len*math.cos(a_min),
                              ry/1000.0 + cone_len*math.sin(a_min))
        pt2 = world_to_pixel(rx/1000.0 + cone_len*math.cos(a_max),
                              ry/1000.0 + cone_len*math.sin(a_max))
        locked_both = pos_locked and ang_locked
        cone_color = (0, 255, 0) if locked_both else mode_col_map
        overlay = map_canvas.copy()
        cv2.fillPoly(overlay, [np.array([tgt_pt, pt1, pt2], dtype=np.int32)], cone_color)
        cv2.addWeighted(overlay, 0.35, map_canvas, 0.65, 0, map_canvas)
        cv2.line(map_canvas, tgt_pt, pt1, cone_color, 1)
        cv2.line(map_canvas, tgt_pt, pt2, cone_color, 1)

        # Live robot
        if initialized:
            robot_col = (0, 255, 0) if pos_locked else (255, 80, 80)
            cam_pt = world_to_pixel(last_x, last_y)
            cv2.circle(map_canvas, cam_pt, 10, robot_col, -1)
            h_end = world_to_pixel(last_x + last_dx * 0.25, last_y + last_dy * 0.25)
            cv2.arrowedLine(map_canvas, cam_pt, h_end, (0, 255, 0), 2, tipLength=0.2)
            cv2.putText(map_canvas, f"{state_text}",
                        (cam_pt[0]-20, cam_pt[1]+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, robot_col, 1)

        cv2.namedWindow("2D Sniper Map", cv2.WINDOW_NORMAL)
        cv2.imshow("2D Sniper Map", map_canvas)

        # ════════════════════════════════════════════════════════════════════
        # 6. KEY HANDLING
        # ════════════════════════════════════════════════════════════════════
        key = cv2.waitKey(1) & 0xFF

        if key == ord('f') and pos_locked and ang_locked:
            print(f"[{SHOT_MODE}] FIRE!  Angle={correction:.2f}°  "
                  f"X={last_x*1000:.1f}mm  Y={last_y*1000:.1f}mm")
            break
        elif key == 27:
            print("Shot aborted (ESC).")
            break

    sniper_thread.stop()
    cap_hd.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()