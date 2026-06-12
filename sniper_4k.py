import cv2
import json
import math
import sys
import os
import time
import threading
import socket
import numpy as np
from collections import deque
from pupil_apriltags import Detector

# ── Import all settings from central config ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    # Field / tags / pockets
    KNOWN_TAGS, TAG_COLORS, ARENA_BOUNDS,
    POCKETS_MM as POCKETS_XY,
    POCKET_TO_TAG,
    TAG_SIZE_M,
    # Camera hardware — user has renamed these to named device paths
    CAM_FRONT,          # HD tracking camera (rear-facing wide view)
    CAM_4K,                      # 4K sniper camera
    HD_RESOLUTION, HD_FPS, HD_CAM_PARAMS,
    CAM_4K_RESOLUTION, CAM_4K_PARAMS, CAM_4K_DIST_COEFFS,
    HD_CAM_FORWARD_M, HD_CAM_SIDEWAYS_M,
    # HD tracking thresholds
    HD_MAX_TAG_DIST as MAX_TAG_DISTANCE,
    HD_OUT_OF_BOUNDS as OUT_OF_BOUNDS_LEEWAY,
    HD_MAX_POSE_JUMP as MAX_POSE_JUMP,
    HD_TRACKING_TIMEOUT as TRACKING_TIMEOUT,
    HD_CROP_TOP as CROP_TOP_PCT,
    HD_CROP_BOT as CROP_BOT_PCT,
    HD_HISTORY_LEN,
    # Sniper-mode tracking thresholds (tighter than driving — robot is stationary)
    SN_STILL_THR  as BASE_STILL_THRESHOLD,
    SN_MOVE_THR   as BASE_MOVE_THRESHOLD,
    SN_STILL_ALPHA as STILL_ALPHA,
    SN_MOVE_ALPHA  as MOVE_ALPHA,
    HD_CONSENSUS_SIGMA as CONSENSUS_SIGMA,
    # Performance
    DETECT_EVERY_N_FRAMES,
    HD_BRIGHTNESS_BOOST,
    # Shot geometry
    SHOT_STRAIGHT_THR as STRAIGHT_THRESHOLD,
    CORNER_OFFSET_MM,
    ANGLE_LOCK_DEG,
    # Ball detection / 4K overlay
    CAM_4K_BALL_RED, CAM_4K_BALL_BLUE,
    LATERAL_TOL_PX, CUT_MODE_INCR_PX,
    # Map display
    MAP_SCALE as SCALE, MAP_PADDING as PADDING,
    # Network — for telemetry broadcast
    TELEMETRY_PORT, SNIPER_TELEMETRY_PORT, SNIPER_FIRE_PORT,
    LINE_INTERSECT_BLEND, SNIPER_4K_HEADING_WEIGHT
)

from tracking import get_global_camera_pose, apply_cam_offset, compute_line_intersections

HAS_DISPLAY = bool(os.environ.get("DISPLAY"))

# ── Derived constants ─────────────────────────────────────────────────────────
CANVAS_W = int((ARENA_BOUNDS[2] - ARENA_BOUNDS[0]) * SCALE) + 2 * PADDING
CANVAS_H = int((ARENA_BOUNDS[3] - ARENA_BOUNDS[1]) * SCALE) + 2 * PADDING

# Ball HSV lookup — maps "Red"/"Blue" to range lists from config
BALL_HSV = {
    "Red":  CAM_4K_BALL_RED,
    "Blue": CAM_4K_BALL_BLUE,
}
DIST_COEFFS = CAM_4K_DIST_COEFFS

# ── Debug image saving ────────────────────────────────────────────────────────
SNIPER_DEBUG_SAVE_DIR = "sniper_debug_images"

# ── CUT target-line smoothing ─────────────────────────────────────────────────
# The TARGET line position is exponentially smoothed over time so that a tag
# flickering in/out of frame doesn't make the line jump.
#
# Two alpha values — one per detection quality:
#   CUT_TARGET_ALPHA_RATIO : both tags visible  → higher alpha (trust fast, accurate)
#   CUT_TARGET_ALPHA_DEPTH : only one tag        → lower alpha  (trust slow, less accurate)
#
# alpha = 1.0 → no smoothing (instant snap to raw value)
# alpha = 0.05 → very heavy smoothing (~20 frames to reach 63% of a step change)
# At ~1fps camera: 0.25 ≈ 4-frame (4s) time constant for ratio mode
#                  0.10 ≈ 10-frame (10s) time constant for depth fallback mode
#
# CUT_TARGET_STALE_S: if NO tags are visible for this many seconds, the smoothed
# value is discarded and target_col returns None (LOST). Prevents using a very
# old stale estimate indefinitely.
CUT_TARGET_ALPHA_RATIO = 0.15   # both tags detected: follow fairly quickly
CUT_TARGET_ALPHA_DEPTH = 0.03   # single-tag fallback: follow slowly (noisier)
CUT_TARGET_STALE_S     = 60.0    # discard smoothed value if no tags seen for this long

# ── Load position calibration (same file as new_main3.py) ────────────────────
# calibration.json is one level up (Carc2026/ root). If absent, identity used.
_CAL_FILE = os.path.join(os.path.dirname(__file__), "..", "calibration.json")
_CAL = {
    "scale_x": 1.0, "offset_x_m": 0.0,
    "scale_y": 1.0, "offset_y_m": 0.0,
    "position_idw_points": [],
    "position_idw_power": 2,
    "heading_idw_points": [],
    "heading_idw_power": 2,
    "heading_idw_fallback_deg": 0.0,
}
try:
    with open(_CAL_FILE) as _f:
        _loaded = json.load(_f)
        _CAL.update({k: _loaded[k] for k in _CAL if k in _loaded})
    print(f"[Sniper] Calibration loaded: "
          f"X scale={_CAL['scale_x']:.4f} offset={_CAL['offset_x_m']*1000:+.1f}mm  "
          f"Y scale={_CAL['scale_y']:.4f} offset={_CAL['offset_y_m']*1000:+.1f}mm  "
          f"pos IDW {len(_CAL['position_idw_points'])} pts  "
        #   f"heading IDW {len(_CAL['heading_idw_points'])} pts"
    )
except FileNotFoundError:
    print("[Sniper] No calibration.json — using raw tracker values.")
except Exception as _e:
    print(f"[Sniper] calibration.json load error: {_e} — using raw values.")


def _idw_position_correction(raw_x_m: float, raw_y_m: float):
    """IDW residual correction for X and Y — see new_main3.py for full comments."""
    pts = _CAL["position_idw_points"]
    if not pts:
        return 0.0, 0.0
    raw_x_mm = raw_x_m * 1000.0
    raw_y_mm = raw_y_m * 1000.0
    power = _CAL["position_idw_power"]
    eps = 20.0 ** 2
    total_w = sum_cx = sum_cy = 0.0
    for p in pts:
        d2 = (raw_x_mm - p["raw_x_mm"])**2 + (raw_y_mm - p["raw_y_mm"])**2
        w  = 1.0 / (d2 + eps) ** power
        total_w += w
        sum_cx  += w * p["residual_x_m"]
        sum_cy  += w * p["residual_y_m"]
    if total_w < 1e-15:
        return 0.0, 0.0
    return sum_cx / total_w, sum_cy / total_w


def _idw_heading_correction(x_m: float, y_m: float) -> float:
    """Position-dependent heading correction via IDW — identical to new_main3."""
    pts = _CAL["heading_idw_points"]
    if not pts:
        return _CAL["heading_idw_fallback_deg"]
    x_mm = x_m * 1000.0;  y_mm = y_m * 1000.0
    power = _CAL["heading_idw_power"]
    eps   = 10.0 ** 2
    total_w = total_c = 0.0
    for p in pts:
        d2 = (x_mm - p["x_mm"])**2 + (y_mm - p["y_mm"])**2
        w  = 1.0 / (d2 + eps) ** power
        total_w += w
        total_c += w * p["correction_deg"]
    return total_c / total_w

# ─────────────────────────────────────────────────────────────────────────────
# MAP HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def world_to_pixel(x, y):
    px = int((x - ARENA_BOUNDS[0]) * SCALE) + PADDING
    py = int((y - ARENA_BOUNDS[1]) * SCALE) + PADDING
    return (px, py)

def draw_map_line(canvas, x1, y1, x2, y2, color=(255, 255, 255), thickness=2):
    cv2.line(canvas, world_to_pixel(x1, y1), world_to_pixel(x2, y2), color, thickness)

# get_global_camera_pose and apply_cam_offset imported from tracking.py


# ─────────────────────────────────────────────────────────────────────────────
# 4K CAMERA THREAD
# Runs independently so its ~1fps decode never blocks the 30fps HD loop.
# Produces: angle correction, raw display frame (resized), tag found flag.
# ─────────────────────────────────────────────────────────────────────────────
class Sniper4kThread(threading.Thread):
    def __init__(self, target_tag_id, ball_color="Red",
                 corner_offset_px=0.0, corner_offset_mm=0.0,
                 # CUT three-line alignment — firing-line wall-intercept target
                 cut_tag_a_id=None,    # first  Y=0 tag bracketing the intercept
                 cut_tag_b_id=None,    # second Y=0 tag (or None → single-tag fallback)
                 cut_ratio=0.0,        # interpolation weight: 0=tag_a, 1=tag_b
                 cut_off_a_mm=0.0,     # signed mm offset from tag_a (single-tag fallback)
                 cut_off_b_mm=0.0,     # signed mm offset from tag_b (single-tag fallback)
                 # CUT ball aiming offset — shifts the BALL LINE from centroid
                 # to the ghost-ball contact point on the ball surface
                 cut_offset_sign=0.0,  # +1 or -1: which side of ball to aim at
                 cut_sin=0.0):         # sin(cut_angle_deg): how far from centre
        super().__init__(daemon=True)
        self.target_tag_id    = target_tag_id
        self.ball_color       = ball_color
        self.corner_offset_px = corner_offset_px
        self.corner_offset_mm = corner_offset_mm
        self.cut_tag_a_id     = cut_tag_a_id
        self.cut_tag_b_id     = cut_tag_b_id
        self.cut_ratio        = cut_ratio
        self.cut_off_a_mm     = cut_off_a_mm
        self.cut_off_b_mm     = cut_off_b_mm
        self.cut_offset_sign  = cut_offset_sign
        self.cut_sin          = cut_sin
        self.tag_size         = 0.11

        # Shared results (written by thread, read by main loop)
        self._lock          = threading.Lock()
        self._heading_4k_deg   = None   # true robot heading from 4K tag poses (degrees, or None)
        self._heading_4k_n     = 0      # number of tags that contributed to heading estimate
        self._tag_found     = False
        self._frame_720p    = None
        self._mask_720p     = None
        self._lateral_px    = None    # CUT: (aiming_col - cam_cx);  STRAIGHT: not used directly
        self._ball_width_px = None
        # STRAIGHT mode: two separate pixel columns for the new three-line alignment
        # These let main() compute the two independent error signals:
        #   diff_error = ball_col - target_col   → drives vX (strafe)
        #   mid_error  = (ball_col + target_col)/2 - cam_cx → drives vR (rotate)
        self._ball_col      = None    # pixel column of ball centroid (full-res)
        self._target_col    = None    # pixel column of tag + live corner offset (full-res)
        self._cam_cx        = None    # camera centre column (full-res, for main())
        # CUT target-line smoothing state (EMA — persists between frames)
        self._tgt_smooth    = None    # float: current smoothed target column
        self._tgt_last_seen = 0.0     # timestamp of last frame with ≥1 visible tag
        # Debug image save state
        self._saved_startup   = False  # flips True after first non-black frame saved
        self._sniper_save_idx = 0

        self.running  = True
        self.detector = Detector(families="tag36h11")
        self.tag_size = TAG_SIZE_M

        if sys.platform.startswith('win'):
            self.cap = cv2.VideoCapture(CAM_4K, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(CAM_4K, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_4K_RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_4K_RESOLUTION[1])

        # Read actual delivered resolution so params scale correctly
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_w = actual_w
        self._actual_h = actual_h
        scale    = actual_h / 1080.0
        self.camera_params_4k = [p * scale for p in CAM_4K_PARAMS]

        # Build undistort map once at actual resolution
        fx, fy, cx, cy = self.camera_params_4k
        cam_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            cam_matrix, DIST_COEFFS, None, cam_matrix,
            (actual_w, actual_h), cv2.CV_16SC2
        )
        # print(f"[4K Thread] {actual_w}x{actual_h}, scale={scale:.2f}, "
        #       f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    def _get_ball_mask(self, frame_bgr):
        """Return (mask, centroid_or_None, ball_width_px).
        mask is a binary uint8 image.
        ball_width_px is the horizontal span of the largest colour blob, or None.
        """
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        ranges = BALL_HSV.get(self.ball_color, BALL_HSV["Red"])
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
            roi_mask = np.zeros_like(mask)
            # Ignore top 30
            TOP = int(h * 0.40)
            BOT = int(h * 0.95)
            LEFT = int(w * 0.10)
            RIGHT = int(w * 0.90)
            roi_mask[TOP:BOT, LEFT:RIGHT] = 255
            roi_mask[500:600, 500:600] = 255
            mask = cv2.bitwise_and(mask, roi_mask)
        # Light morphology to clean noise without destroying small detections
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Find centroid and width of largest blob
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroid    = None
        ball_width_px = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 150:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # Horizontal span = ball width in pixels (more stable than fitting a circle)
                x_bnd, _, w_bnd, _ = cv2.boundingRect(largest)
                ball_width_px = w_bnd
        return mask, centroid, ball_width_px

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Undistort
            frame_ud = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

            # ── AprilTag detection ────────────────────────────────────────
            # We detect ALL visible tags to build a robust heading estimate.
            # Each tag provides an independent true-heading measurement via
            # get_global_camera_pose() → (gx,gy,gdx,gdy).
            # heading_from_tag = atan2(gdy, gdx) — this is the ACTUAL robot
            # heading in world-frame degrees, not the tag's angle in the image.
            # We distance-weight all tags and take a circular average.
            gray    = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2GRAY)
            results = self.detector.detect(gray, estimate_tag_pose=True,
                                           camera_params=self.camera_params_4k,
                                           tag_size=self.tag_size)

            found      = False
            tag_cx     = None   # target tag center X in full-res pixels (for visual)
            tag_cy     = None
            tag_tz_live = 0.0   # live optical-axis depth of target tag (metres)
            draw_frame = frame_ud.copy()
            h, w = draw_frame.shape[:2]
            # Pixel columns of every detected known tag — keyed by tag_id
            # value: (col_px: int, tz_metres: float)
            all_tag_cols: dict = {}

            # Accumulate heading contributions from all detected tags
            sin_sum = 0.0
            cos_sum = 0.0
            heading_total_w = 0.0
            heading_n = 0

            for tag in results:
                # ── Extract heading from this tag (any tag) ────────────────
                if tag.tag_id not in KNOWN_TAGS:
                    continue
                t_tag = tag.pose_t
                R_tag = tag.pose_R
                pose = get_global_camera_pose(tag.tag_id, t_tag, R_tag)
                if pose is None:
                    continue
                gx, gy, gdx, gdy = pose
                # Distance to tag for weighting (closer = more reliable)
                dist_tag = math.sqrt(t_tag[0][0]**2 + t_tag[1][0]**2 + t_tag[2][0]**2)
                if dist_tag < 1e-6:
                    continue
                w_tag = 1.0 / (dist_tag ** 2)   # 1/d² weighting
                heading_rad = math.atan2(gdy, gdx)
                sin_sum += w_tag * math.sin(heading_rad)
                cos_sum += w_tag * math.cos(heading_rad)
                heading_total_w += w_tag
                heading_n += 1

                # Record pixel column + live depth for ALL known tags
                # (CUT mode needs two Y=0 tags for the interpolation)
                if tag.tag_id in KNOWN_TAGS:
                    all_tag_cols[tag.tag_id] = (
                        int(tag.center[0]),
                        float(tag.pose_t[2][0])   # optical-axis depth in metres
                    )

                # ── Visual overlays for the TARGET tag only ────────────────
                if tag.tag_id != self.target_tag_id:
                    # Non-target tags: small grey box only
                    corners_nt = tag.corners.astype(int)
                    for i in range(4):
                        cv2.line(draw_frame, tuple(corners_nt[i]),
                                 tuple(corners_nt[(i+1) % 4]), (80, 80, 80), 3)
                    continue

                # Target tag found
                found = True
                corners = tag.corners.astype(int)
                for i in range(4):
                    cv2.line(draw_frame, tuple(corners[i]),
                             tuple(corners[(i+1) % 4]), (0, 0, 255), 8)

                tag_cx = int(tag.center[0])
                tag_cy = int(tag.center[1])
                # Live depth along optical axis (metres) — used for corner offset scaling
                tag_tz_live = float(tag.pose_t[2][0])   # t_z in metres
                cv2.circle(draw_frame, (tag_cx, tag_cy), 14, (0, 255, 255), 4)

                # Cyan vertical guide through tag centre (heading reference only)
                cv2.line(draw_frame, (tag_cx, 0), (tag_cx, self._actual_h),
                         (0, 200, 255), 2)

            # ── Compute 4K heading from accumulated tags ───────────────────
            if heading_n > 0 and heading_total_w > 1e-12:
                heading_4k_deg = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
            else:
                heading_4k_deg = None

            # ── Ball colour mask ──────────────────────────────────────────
            ball_mask_binary, ball_centroid, ball_width_px = self._get_ball_mask(frame_ud)

            # Coloured mask overlay (semi-transparent, 40% opacity)
            mask_colour = (0, 0, 255) if self.ball_color == "Red" else (255, 100, 0)
            mask_overlay = draw_frame.copy()
            mask_overlay[ball_mask_binary > 0] = mask_colour
            draw_frame = cv2.addWeighted(draw_frame, 0.6, mask_overlay, 0.4, 0)

            cam_centre_x = self._actual_w // 2

            if ball_centroid is not None:
                bx, by = ball_centroid
                cv2.circle(draw_frame, (bx, by), 20, (0, 255, 0), 4)
                cv2.putText(draw_frame, "BALL", (bx + 25, by),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

            # ── CUT mode: camera-centre line + ball-offset aiming line ────
            # CUT lateral is measured entirely in ball space, no tag needed.
            # The robot strafes until the camera's central axis passes through
            # the ghost-ball contact point on the coloured ball surface.
            #
            #   aiming_col = ball_centroid_x + cut_offset_sign * cut_sin * ball_radius_px
            #
            # lateral_px = aiming_col - cam_centre_x
            #   > 0: aiming point is to the RIGHT of camera centre → MOVE RIGHT
            #   < 0: aiming point is to the LEFT                   → MOVE LEFT
            #
            # Visual:
            #   GREEN  vertical line = camera optical axis (centre of frame) = where we ARE aiming
            #   YELLOW vertical line = desired aiming column (ghost-ball contact point)
            #   When green == yellow → locked.
            # ─────────────────────────────────────────────────────────────

            lateral_px = None
            ball_col   = None
            target_col = None

            if self.cut_tag_a_id is not None:
                # ── CUT MODE — three-line firing-line alignment ───────────
                #
                # Same three-line system as STRAIGHT:
                #   GREEN  (AIM)    = camera centre = robot shooting axis
                #   YELLOW (BALL)   = ball centroid column
                #   MAGENTA(TARGET) = pixel column of the Y=0 wall point where
                #                     the firing line meets it, computed by
                #                     interpolating between two detected tag cols
                #
                # TARGET column resolution order:
                #   1. Both tag_a AND tag_b visible:
                #        target_col = col_a + ratio × (col_b − col_a)
                #        Pure pixel ratio — no depth needed — most accurate.
                #   2. Only tag_a:
                #        target_col = col_a + fx × off_a_mm / (tz_a × 1000)
                #   3. Only tag_b:
                #        target_col = col_b + fx × off_b_mm / (tz_b × 1000)
                #   4. Neither: target_col = None (LOST)
                #
                # Error signals (identical to STRAIGHT):
                #   diff_error = ball_col − target_col  → vX (strafe)
                #   mid_error  = mid(ball,target) − cam_cx  → vR (rotate)

                fx_now  = self.camera_params_4k[0]
                entry_a = all_tag_cols.get(self.cut_tag_a_id)   # (col, tz) or None
                entry_b = all_tag_cols.get(self.cut_tag_b_id)   # (col, tz) or None

                # ── Raw target column estimate ─────────────────────────────
                # Priority: ratio (both tags) > depth from a > depth from b > None
                # Each method gets a different EMA alpha — ratio is more accurate
                # so it's allowed to pull the smoothed value faster.
                both_visible = entry_a is not None and entry_b is not None
                if both_visible:
                    raw_tgt = (entry_a[0]
                               + self.cut_ratio * (entry_b[0] - entry_a[0]))
                    alpha   = CUT_TARGET_ALPHA_RATIO
                    tag_src = (f"T{self.cut_tag_a_id}({entry_a[0]}px)"
                               f"+{self.cut_ratio*100:.0f}%→"
                               f"T{self.cut_tag_b_id}({entry_b[0]}px)")
                elif entry_a is not None and entry_a[1] > 0.01:
                    off_px  = fx_now * self.cut_off_a_mm / (entry_a[1] * 1000.0)
                    raw_tgt = entry_a[0] + off_px
                    alpha   = CUT_TARGET_ALPHA_DEPTH
                    tag_src = f"T{self.cut_tag_a_id}+depth({off_px:+.0f}px)"
                elif entry_b is not None and entry_b[1] > 0.01:
                    off_px  = fx_now * self.cut_off_b_mm / (entry_b[1] * 1000.0)
                    raw_tgt = entry_b[0] + off_px
                    alpha   = CUT_TARGET_ALPHA_DEPTH
                    tag_src = f"T{self.cut_tag_b_id}+depth({off_px:+.0f}px)"
                else:
                    raw_tgt = None
                    alpha   = 0.0
                    tag_src = "NO TAGS"

                # ── EMA smoothing ──────────────────────────────────────────
                # Update last-seen timestamp whenever any raw estimate exists.
                # When tags flicker out: hold the last smoothed value instead
                # of jumping to None.  When they reappear: blend back gently.
                # Ratio-method alpha is higher (faster) than depth-method alpha
                # (lower/slower) so that a less-reliable single-tag reading
                # barely nudges the estimate, while both-tag ratio readings
                # can pull it more decisively.
                now_t = time.time()
                if raw_tgt is not None:
                    self._tgt_last_seen = now_t
                    if self._tgt_smooth is None:
                        self._tgt_smooth = float(raw_tgt)  # cold start: seed directly
                    else:
                        self._tgt_smooth = (alpha * raw_tgt
                                            + (1.0 - alpha) * self._tgt_smooth)
                else:
                    # No tags this frame — keep holding last smoothed value
                    # but discard it if it has been too long (stale)
                    if (self._tgt_smooth is not None
                            and now_t - self._tgt_last_seen > CUT_TARGET_STALE_S):
                        self._tgt_smooth = None

                # Display: show raw vs smoothed so the operator can see the gap
                if self._tgt_smooth is not None and raw_tgt is not None:
                    tag_src += (f"  α={alpha:.2f} "
                                f"raw={raw_tgt:.0f} smth={self._tgt_smooth:.0f}")
                elif self._tgt_smooth is not None:
                    age = now_t - self._tgt_last_seen
                    tag_src = f"HOLDING  smth={self._tgt_smooth:.0f}  ({age:.1f}s ago)"

                tgt_col    = (int(round(self._tgt_smooth))
                              if self._tgt_smooth is not None else None)
                target_col = tgt_col

                # Green: robot shooting axis
                cv2.line(draw_frame, (cam_centre_x, 0), (cam_centre_x, self._actual_h),
                         (0, 255, 0), 3)
                cv2.putText(draw_frame, "AIM",
                            (cam_centre_x + 8, self._actual_h // 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

                # Cyan/light-blue: show both bracket tags with their column
                for tid, clr in [(self.cut_tag_a_id, (0, 200, 255)),
                                 (self.cut_tag_b_id, (100, 200, 255))]:
                    if tid in all_tag_cols:
                        tc = all_tag_cols[tid][0]
                        cv2.line(draw_frame, (tc, 0), (tc, self._actual_h), clr, 2)
                        cv2.putText(draw_frame, f"T{tid}",
                                    (tc + 8, self._actual_h * 2 // 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, clr, 2)

                # Magenta: target column (firing-line wall intercept)
                if tgt_col is not None:
                    # Thick magenta: smoothed target (what the servo uses)
                    cv2.line(draw_frame, (tgt_col, 0), (tgt_col, self._actual_h),
                             (255, 0, 255), 4)
                    cv2.putText(draw_frame, "TARGET",
                                (tgt_col + 10, self._actual_h * 3 // 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 255), 4)

                    # Thin dashed pink: raw (un-smoothed) target — shows drift visually
                    if raw_tgt is not None:
                        raw_col = int(round(raw_tgt))
                        if abs(raw_col - tgt_col) > 1:  # only draw if visibly different
                            for y in range(0, self._actual_h, 24):
                                cv2.line(draw_frame,
                                         (raw_col, y),
                                         (raw_col, min(y + 12, self._actual_h)),
                                         (200, 100, 200), 2)

                    cv2.putText(draw_frame, tag_src,
                                (tgt_col + 10, self._actual_h * 4 // 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 255), 2)

                # Yellow: AIMING COLUMN — the ghost-ball contact point on the ball
                # surface, offset from the ball centroid by sin(cut_angle) × radius.
                # This is NOT the ball centre — it's the side of the ball that the
                # robot's firing axis must pass through to execute the cut correctly.
                #   aiming_col = ball_cx + cut_offset_sign × sin(cut_angle) × ball_radius_px
                if ball_centroid is not None:
                    if ball_width_px is not None and self.cut_sin > 0.0:
                        ball_radius_px = ball_width_px / 2.0
                        aiming_col = int(round(
                            bx + self.cut_offset_sign * self.cut_sin * ball_radius_px
                        ))
                    else:
                        aiming_col = bx   # fallback: treat as straight if no width

                    ball_col = aiming_col   # this is what diff_error/mid_error use

                    # Draw a faint centroid marker so the ball centre is still visible
                    cv2.circle(draw_frame, (bx, by), 8, (0, 180, 80), 2)

                    # Yellow: aiming column (the actual target point on ball surface)
                    cv2.line(draw_frame, (aiming_col, 0), (aiming_col, self._actual_h),
                             (0, 220, 100), 4)
                    cv2.putText(draw_frame, "BALL",
                                (aiming_col + 10, self._actual_h * 5 // 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 100), 3)

                    # Small circle on the ball at the exact contact point
                    cv2.circle(draw_frame, (aiming_col, by), 14, (0, 220, 100), 3)

                    # Width bar showing measured diameter + offset amount
                    half = int(ball_width_px / 2) if ball_width_px else 20
                    bar_y = by + 40
                    cv2.line(draw_frame, (bx - half, bar_y), (bx + half, bar_y),
                             (150, 150, 0), 2)
                    off_px = aiming_col - bx
                    cv2.putText(draw_frame,
                                f"w={ball_width_px}px off={off_px:+d}px ({self.cut_sin*100:.0f}%r)",
                                (bx - half, bar_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 150, 0), 2)

                    if tgt_col is not None:
                        mid_y   = self._actual_h * 3 // 4
                        mid_col = (aiming_col + tgt_col) // 2
                        cv2.line(draw_frame, (aiming_col, mid_y), (tgt_col, mid_y),
                                 (200, 200, 0), 3)
                        cv2.circle(draw_frame, (mid_col, mid_y), 8, (200, 200, 0), -1)
                        cv2.putText(draw_frame, "MID",
                                    (mid_col + 10, mid_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            else:
                # ── STRAIGHT MODE — three-line alignment ─────────────────
                #
                # THREE LINES that must all land on the same pixel column:
                #
                #   GREEN  (AIM)    = camera optical axis = robot shooting line
                #                     always at cam_cx — fixed to robot centre
                #   YELLOW (BALL)   = ball centroid column — tied to the ball
                #   MAGENTA (TARGET)= tag column + live corner offset
                #                     tied to the AprilTag, scales with measured depth
                #
                # TWO INDEPENDENT ERROR SIGNALS:
                #
                #   diff_error = ball_col - target_col  → drives vX (STRAFE)
                #     strafing shifts ball and tag by DIFFERENT amounts (different depths)
                #     → this closes/opens the gap between them
                #
                #   mid_error  = (ball_col + target_col) / 2 - cam_cx  → drives vR (ROTATE)
                #     rotating shifts ALL columns by the SAME amount (same angular shift)
                #     → this centres the average of ball+target onto the camera axis
                #
                # When both = 0:  ball_col = target_col = cam_cx  ← perfect alignment
                # NO HEADING SENSOR NEEDED — purely camera-vision driven.
                #
                # CORNER OFFSET: live from tag pose t_z (metres) so it scales correctly
                # regardless of robot Y distance.  corner_offset_mm is physical mm offset.

                # Green: robot shooting axis (always cam_cx)
                cv2.line(draw_frame, (cam_centre_x, 0), (cam_centre_x, self._actual_h),
                         (0, 255, 0), 3)
                cv2.putText(draw_frame, "AIM",
                            (cam_centre_x + 8, self._actual_h // 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

                if tag_cx is not None:
                    # Live corner offset: fx * mm / (tz_metres * 1000)
                    if abs(self.corner_offset_mm) > 0.01 and tag_tz_live > 0.01:
                        fx_now = self.camera_params_4k[0]
                        live_corner_px = int(round(
                            math.copysign(
                                fx_now * abs(self.corner_offset_mm) / (tag_tz_live * 1000.0),
                                self.corner_offset_mm   # sign from mm value
                            )
                        ))
                    else:
                        live_corner_px = 0

                    tgt_col = tag_cx + live_corner_px
                    target_col = tgt_col

                    # Cyan: tag centre
                    cv2.line(draw_frame, (tag_cx, 0), (tag_cx, self._actual_h),
                             (0, 200, 255), 2)
                    cv2.putText(draw_frame, "TAG",
                                (tag_cx + 8, self._actual_h * 2 // 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 2)

                    # Magenta: target (tag + live corner offset)
                    cv2.line(draw_frame, (tgt_col, 0), (tgt_col, self._actual_h),
                             (255, 0, 255), 4)
                    cv2.putText(draw_frame, "TARGET",
                                (tgt_col + 10, self._actual_h * 3 // 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 255), 4)
                    if abs(self.corner_offset_mm) > 0.01:
                        cv2.putText(draw_frame,
                                    f"tz={tag_tz_live:.2f}m  off={live_corner_px:+d}px",
                                    (tgt_col + 10, self._actual_h * 4 // 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)

                if ball_centroid is not None:
                    ball_col = bx
                    # Yellow: ball column
                    cv2.line(draw_frame, (bx, 0), (bx, self._actual_h),
                             (0, 220, 100), 4)
                    cv2.putText(draw_frame, "BALL",
                                (bx + 10, self._actual_h * 5 // 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 100), 3)

                    if target_col is not None:
                        # Horizontal bar showing diff_error (ball ↔ target gap)
                        mid_y = self._actual_h * 3 // 4
                        cv2.line(draw_frame, (bx, mid_y), (tgt_col, mid_y), (200, 200, 0), 3)
                        mid_col = (bx + tgt_col) // 2
                        cv2.circle(draw_frame, (mid_col, mid_y), 8, (200, 200, 0), -1)
                        cv2.putText(draw_frame, "MID",
                                    (mid_col + 10, mid_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            # ── Downscale to half-res for display ────────────────────────
            # Use actual_w//2 × actual_h//2 so the window fits correctly at
            # any capture resolution (1080p, 1440p, 4K, etc.)
            disp_w = self._actual_w // 4
            disp_h = self._actual_h // 4
            frame_small = cv2.resize(draw_frame, (disp_w, disp_h))
            self._disp_w = disp_w
            self._disp_h = disp_h

            # Also build a standalone colour mask image at display resolution
            disp_w = self._actual_w // 4
            disp_h = self._actual_h // 4
            mask_bgr_small = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
            mask_bgr_small[cv2.resize(ball_mask_binary, (disp_w, disp_h)) > 0] = mask_colour

            with self._lock:
                self._heading_4k_deg   = heading_4k_deg
                self._heading_4k_n     = heading_n
                self._tag_found     = found
                self._frame_720p    = frame_small
                self._mask_720p     = mask_bgr_small
                self._lateral_px    = lateral_px
                self._ball_width_px = ball_width_px if ball_centroid is not None else None
                self._ball_col      = ball_col
                self._target_col    = target_col
                self._cam_cx        = cam_centre_x

            # ── Debug image save ──────────────────────────────────────────
            # Save once when the camera first produces a non-black frame,
            # then again each time a full sniper lock is achieved.
            # "Clean" = undistorted frame before any overlays.
            # "Debug" = quarter-res frame with all overlay lines baked in.
            if not self._saved_startup:
                mean_val = float(frame_ud.mean())
                if mean_val > 5.0:   # not a black frame
                    self._saved_startup = True
                    import os
                    os.makedirs(SNIPER_DEBUG_SAVE_DIR, exist_ok=True)
                    # Find next index
                    existing = []
                    for fn in os.listdir(SNIPER_DEBUG_SAVE_DIR):
                        nm = os.path.splitext(fn)[0]
                        for pfx in ("clean_", "debug_"):
                            if nm.startswith(pfx):
                                try: existing.append(int(nm[len(pfx):]))
                                except ValueError: pass
                    idx = max(existing, default=0) + 1
                    self._sniper_save_idx = idx
                    cv2.imwrite(
                        os.path.join(SNIPER_DEBUG_SAVE_DIR, f"clean_{idx}.jpg"),
                        frame_ud
                    )
                    cv2.imwrite(
                        os.path.join(SNIPER_DEBUG_SAVE_DIR, f"debug_{idx}.jpg"),
                        frame_small
                    )
                    print(f"[Sniper4k] Saved startup frame #{idx} "
                          f"(mean={mean_val:.1f}) → {SNIPER_DEBUG_SAVE_DIR}/")

    def get_state(self):
        with self._lock:
            return (self._heading_4k_deg, self._heading_4k_n, self._tag_found,
                    self._frame_720p, self._mask_720p,
                    self._lateral_px, self._ball_width_px,
                    self._ball_col, self._target_col, self._cam_cx)

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

    fire_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fire_sock.bind(("0.0.0.0", SNIPER_FIRE_PORT))
    fire_sock.setblocking(False)

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

    # ── Target heading = fire direction the robot must achieve ───────────────
    # The robot must face fire_angle_deg (world frame) to execute the shot.
    # heading_error = fire_angle_deg - actual_heading_deg  (normalised ±180°)
    # positive = robot facing too far LEFT  → TURN RIGHT (vR > 0)
    # negative = robot facing too far RIGHT → TURN LEFT  (vR < 0)
    target_heading_deg = fire_angle_deg
    print(f"[Sniper] Target heading: {target_heading_deg:.2f}°  (fire_angle from planner)")

    # Lateral sensitivity geometry inputs
    tag_world_y_mm = KNOWN_TAGS[target_tag_id]["y"] * 1000.0
    ball_depth_mm  = ry - ball_xy[1]         # ry - ball_y_mm
    tag_depth_mm   = ry - tag_world_y_mm     # ry - tag_y_mm
    rx_m = rx / 1000.0                       # metres, for HD X-error display

    # ── CUT mode: compute firing-line Y=0 wall intercept ─────────────────────
    # For CUT mode, the TARGET line is where the robot's firing axis would hit
    # the Y=0 pocket wall.  We express this as an interpolation between two
    # detected Y=0 AprilTag pixel columns so:
    #   • No heading sensor is required
    #   • Small tag-placement errors cancel (both tags shift together with robot movement)
    #   • The offset scales automatically with robot distance (ratio is distance-independent)
    #
    # Firing line: (rx, ry) + t·(cos(fire_angle), sin(fire_angle))
    # At Y=0:  t = −ry / sin(fire_angle)
    #          target_x_mm = rx + t·cos(fire_angle)
    #
    # Y=0 tags sorted by world X (from config):  tag3→X=0, tag4→X=600, tag5→X=1200
    # Find bracket pair [tag_a, tag_b] that straddles target_x_mm.
    # ratio = (target_x − tag_a_x) / (tag_b_x − tag_a_x)
    # At runtime: target_col = col_a + ratio × (col_b − col_a)

    Y0_TAGS = sorted(
        [(int(round(info["x"] * 1000)), tid)
         for tid, info in KNOWN_TAGS.items()
         if abs(info["y"]) < 0.01],
        key=lambda t: t[0]
    )   # [(x_mm, tag_id), ...] sorted ascending

    cut_tag_a_id = None
    cut_tag_b_id = None
    cut_ratio    = 0.0
    cut_off_a_mm = 0.0
    cut_off_b_mm = 0.0

    if SHOT_MODE == "CUT" and abs(math.sin(math.radians(fire_angle_deg))) > 1e-6:
        fire_rad_cut   = math.radians(fire_angle_deg)
        t_wall         = -ry / math.sin(fire_rad_cut)
        target_x_mm    = rx + t_wall * math.cos(fire_rad_cut)
        print(f"[Sniper] CUT firing-line → Y=0 wall at X = {target_x_mm:.1f}mm")

        # Find the bracket pair
        bracketed = False
        for i in range(len(Y0_TAGS) - 1):
            xa, ta = Y0_TAGS[i]
            xb, tb = Y0_TAGS[i + 1]
            if xa <= target_x_mm <= xb:
                cut_tag_a_id = ta
                cut_tag_b_id = tb
                cut_ratio    = (target_x_mm - xa) / (xb - xa)
                cut_off_a_mm = target_x_mm - xa   # + = right of tag_a
                cut_off_b_mm = target_x_mm - xb   # − = left  of tag_b
                bracketed    = True
                break

        if not bracketed:
            # Extrapolate from nearest boundary pair
            if target_x_mm < Y0_TAGS[0][0]:
                xa, ta = Y0_TAGS[0]; xb, tb = Y0_TAGS[1]
            else:
                xa, ta = Y0_TAGS[-2]; xb, tb = Y0_TAGS[-1]
            cut_tag_a_id = ta; cut_tag_b_id = tb
            cut_ratio    = (target_x_mm - xa) / (xb - xa)
            cut_off_a_mm = target_x_mm - xa
            cut_off_b_mm = target_x_mm - xb

        print(f"[Sniper] CUT tag pair: tag{cut_tag_a_id}↔tag{cut_tag_b_id}  "
              f"ratio={cut_ratio:.4f}  "
              f"off_a={cut_off_a_mm:+.1f}mm  off_b={cut_off_b_mm:+.1f}mm")

    # ── CUT ball-offset: which side of the ball to aim at ────────────────────
    # The BALL LINE in CUT mode is NOT the ball centroid — it's the ghost-ball
    # contact point, offset from the centroid by sin(cut_angle) × ball_radius.
    # Direction (which side) is derived from the ghost_ball position projected
    # onto the camera's lateral axis at the planned firing direction.
    cut_offset_sign = 0.0
    cut_sin         = 0.0
    if SHOT_MODE == "CUT":
        fire_rad = math.radians(fire_angle_deg)
        right_x  = -math.sin(fire_rad)   # camera right vector X component
        right_y  =  math.cos(fire_rad)   # camera right vector Y component
        ghost_x  = float(target_data["ghost_ball_xy"][0])
        ghost_y  = float(target_data["ghost_ball_xy"][1])
        ball_x_mm = float(ball_xy[0])
        ball_y_mm = float(ball_xy[1])
        goff_x   = ghost_x - ball_x_mm
        goff_y   = ghost_y - ball_y_mm
        cam_lat  = goff_x * right_x + goff_y * right_y   # mm, + = right in camera
        cut_offset_sign = 1.0 if cam_lat >= 0.0 else -1.0
        cut_sin         = math.sin(math.radians(cut_angle_deg))
        print(f"[Sniper] CUT ball offset: sign={cut_offset_sign:+.0f}  "
              f"sin({cut_angle_deg:.1f}°)={cut_sin:.4f}  "
              f"(ghost cam-lateral={cam_lat:+.1f}mm)")

    # ── Start 4K thread ─────────────────────────────────────────────────────
    sniper_thread = Sniper4kThread(
        target_tag_id, ball_color,
        corner_offset_px=0.0,
        cut_tag_a_id=cut_tag_a_id,
        cut_tag_b_id=cut_tag_b_id,
        cut_ratio=cut_ratio,
        cut_off_a_mm=cut_off_a_mm,
        cut_off_b_mm=cut_off_b_mm,
        cut_offset_sign=cut_offset_sign,
        cut_sin=cut_sin,
    )
    sniper_thread.start()

    # Compute lateral sensitivity now that thread has opened the camera
    fx_4k = sniper_thread.camera_params_4k[0]

    # ── Lateral sensitivity — same formula for both STRAIGHT and CUT ─────────
    # When the robot strafes 1mm, the ball shifts fx/ball_depth pixels in the frame.
    # The target (tag or wall intercept) also shifts, but at its own depth.
    # The net diff_error sensitivity per mm = fx/ball_depth − fx/target_depth.
    # For simplicity we use fx/ball_depth as the mm-per-pixel scale —
    # it's correct to within ~10% and the target depth terms are similar.
    if ball_depth_mm > 0:
        lateral_sens   = fx_4k / ball_depth_mm
        lateral_acc_mm = LATERAL_TOL_PX / lateral_sens if lateral_sens > 1e-9 else 99.9
    else:
        lateral_sens   = 1.0
        lateral_acc_mm = 99.9
    print(f"[Sniper] Lateral sensitivity: {lateral_sens:.3f} px/mm  "
          f"(ball_depth={ball_depth_mm:.0f}mm  tol={LATERAL_TOL_PX}px → ±{lateral_acc_mm:.1f}mm)")

    # ── STRAIGHT: corner offset on target tag ────────────────────────────────
    CORNER_TAGS_SIGN = {3: +1, 5: -1}   # Hole_3 = left corner (+X), Hole_5 = right (−X)
    target_lateral_px = 0.0
    if SHOT_MODE == "STRAIGHT":
        if target_tag_id in CORNER_TAGS_SIGN and CORNER_OFFSET_MM != 0:
            corner_sign        = CORNER_TAGS_SIGN[target_tag_id]
            signed_offset_mm   = corner_sign * CORNER_OFFSET_MM
            target_lateral_px  = fx_4k * signed_offset_mm / tag_depth_mm
            sniper_thread.corner_offset_px = target_lateral_px
            sniper_thread.corner_offset_mm = signed_offset_mm
            print(f"[Sniper] Corner offset: {signed_offset_mm:+.1f}mm  "
                  f"≈ {target_lateral_px:+.1f}px at startup depth  "
                  f"({'RIGHT' if corner_sign>0 else 'LEFT'} of tag center)")

    # ── HD camera (position — 30fps) ─────────────────────────────────────────
    if sys.platform.startswith('win'):
        cap_hd = cv2.VideoCapture(CAM_FRONT, cv2.CAP_DSHOW)
    else:
        cap_hd = cv2.VideoCapture(CAM_FRONT, cv2.CAP_V4L2)
    cap_hd.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap_hd.set(cv2.CAP_PROP_FRAME_WIDTH,  HD_RESOLUTION[0])
    cap_hd.set(cv2.CAP_PROP_FRAME_HEIGHT, HD_RESOLUTION[1])
    cap_hd.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_hd.set(cv2.CAP_PROP_FPS, HD_FPS)
    cap_hd.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    for _ in range(5):
        cap_hd.read()
        time.sleep(0.05)

    actual_h_hd      = int(cap_hd.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_hd         = actual_h_hd / 1080.0
    camera_params_hd = [p * scale_hd for p in HD_CAM_PARAMS]

    detector_hd = Detector(families="tag36h11")
    tag_size_hd = TAG_SIZE_M

    # ── Position tracking state (mirrors new_main3.py exactly) ──────────────
    history_raw_x = deque(maxlen=HD_HISTORY_LEN)
    history_raw_y = deque(maxlen=HD_HISTORY_LEN)
    last_x, last_y, last_dx, last_dy = 0.0, 0.0, 1.0, 0.0
    initialized    = False
    last_seen_time = time.time()
    state_text     = "LOST"
    debug_tag_estimates = []

    print(f"\n[Sniper] Target pocket : {target_pocket}  (Tag {target_tag_id})")
    print(f"[Sniper] Required X    : {rx:.1f} mm")
    print(f"[Sniper] Required Y    : {ry:.1f} mm")
    print(f"[Sniper] Target heading: {target_heading_deg:.2f}°  (fire_angle)")
    print(f"[Sniper] HD+4K weight  : 4K={SNIPER_4K_HEADING_WEIGHT:.2f}  HD={1-SNIPER_4K_HEADING_WEIGHT:.2f}")
    print(f"[Sniper] HD cam  → position (X, Y, heading) at 30fps")
    print(f"[Sniper] 4K cam  → heading from ALL visible tags (threaded)")

    # ── Telemetry broadcast socket (same port as new_main3 — mission_controller ──
    # listens here while sniper is active).  Uses the same packet format as
    # new_main3, extended with sniper-specific fields so SniperState works.
    _telem_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    startup_pkt = {
        "initialized": False,
        "state": "STARTING",
        "app_state": "AIMING",
        "sniper_active": True,
        "shot_mode": SHOT_MODE,
        "target_pocket": target_pocket,
        "target_zone": target_data.get("target_ball_zone", "?"),
    }
    
    _telem_sock.sendto(
        json.dumps(startup_pkt).encode(),
        ("127.0.0.1", SNIPER_TELEMETRY_PORT)
    )

    _hd_frame_count  = 0
    debug_tag_estimates = []   # persists across skipped frames
    raw_circle_estimates = []  # persists across skipped frames

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

        # ── AprilTag detection (skip frames on Pi for performance) ────────
        _hd_frame_count += 1
        if _hd_frame_count % DETECT_EVERY_N_FRAMES == 0:
            gray_hd = cv2.cvtColor(frame_hd, cv2.COLOR_BGR2GRAY)[y_start:y_end, :]
            adj_params_hd = [camera_params_hd[0], camera_params_hd[1],
                             camera_params_hd[2], camera_params_hd[3] - y_start]
            tags_hd = detector_hd.detect(gray_hd, estimate_tag_pose=True,
                                         camera_params=adj_params_hd, tag_size=tag_size_hd)
            debug_tag_estimates  = []   # fresh results this frame
            raw_circle_estimates = []   # fresh results this frame
        else:
            tags_hd = []   # skip; keep previous lists

        for tag in tags_hd:
            tag.corners[:, 1] += y_start
            tag.center[1]     += y_start
            t, R = tag.pose_t, tag.pose_R
            pose = get_global_camera_pose(tag.tag_id, t, R)
            if pose is None: continue
            dist = math.sqrt(t[0][0]**2 + t[1][0]**2 + t[2][0]**2)
            if dist > MAX_TAG_DISTANCE: continue
            cam_x, cam_y, gdx, gdy = pose
            # Apply camera-to-robot-centre offset
            robot_x, robot_y = apply_cam_offset(cam_x, cam_y, gdx, gdy,
                                                 HD_CAM_FORWARD_M, HD_CAM_SIDEWAYS_M)

            # ── Circle-intersection entry — ALL distance-valid tags ────────
            # Kept separate from pose estimates: distance is reliable even when
            # the pose is rejected by bounds/jump filter, so these tags still
            # improve the circle intersection result.
            raw_circle_estimates.append({
                "id": tag.tag_id, "dist": dist,
                "pose": (robot_x, robot_y, gdx, gdy),
            })

            # ── Pose-based entry — apply bounds + jump filter ─────────────
            if not (ARENA_BOUNDS[0] - OUT_OF_BOUNDS_LEEWAY <= robot_x <= ARENA_BOUNDS[2] + OUT_OF_BOUNDS_LEEWAY and
                    ARENA_BOUNDS[1] - OUT_OF_BOUNDS_LEEWAY <= robot_y <= ARENA_BOUNDS[3] + OUT_OF_BOUNDS_LEEWAY):
                continue
            if initialized and math.hypot(robot_x - last_x, robot_y - last_y) > MAX_POSE_JUMP:
                continue
            robot_pose = (robot_x, robot_y, gdx, gdy)
            w = 1.0 / (dist**3 + 0.001)
            debug_tag_estimates.append({"id": tag.tag_id, "pose": robot_pose, "dist": dist, "w": w})
        #     print(f"  Tag {tag.tag_id} dist={dist:.2f}m pose=({robot_x:.3f},{robot_y:.3f})")
        # print(f"[Sniper HD] Tags seen this frame: {len(tags_hd)} | debug_estimates: {len(debug_tag_estimates)} | initialized={initialized}")

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

            # ── Circle-intersection blend ──────────────────────────────────
            # Uses raw_circle_estimates (distance-only filtered) — tags rejected
            # by bounds/jump still contribute their distance to the circles.
            isect = compute_line_intersections(raw_circle_estimates)
            if isect and LINE_INTERSECT_BLEND > 0:
                raw_x = (1 - LINE_INTERSECT_BLEND) * raw_x + LINE_INTERSECT_BLEND * isect[0]
                raw_y = (1 - LINE_INTERSECT_BLEND) * raw_y + LINE_INTERSECT_BLEND * isect[1]

            # ── Apply calibration BEFORE the low-pass filter ───────────────
            # Must be here — applying after smoothing compounds the scale
            # factor every frame and causes position drift (same bug as new_main3).
            # Step 1: global linear correction
            raw_x  = _CAL["scale_x"] * raw_x  + _CAL["offset_x_m"]
            raw_y  = _CAL["scale_y"] * raw_y  + _CAL["offset_y_m"]
            # Step 2: IDW residual correction (local zone errors)
            _res_x, _res_y = _idw_position_correction(raw_x, raw_y)
            raw_x += _res_x
            raw_y += _res_y
            # Heading IDW correction (position-dependent)
            _raw_h  = math.degrees(math.atan2(raw_dy, raw_dx)) % 360
            _h_corr = _idw_heading_correction(raw_x, raw_y)
            _cal_h  = (_raw_h + _h_corr) % 360
            raw_dx  = math.cos(math.radians(_cal_h))
            raw_dy  = math.sin(math.radians(_cal_h))

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
        heading_4k_deg, heading_4k_n, tag_found_4k, frame_4k_720p, mask_720p, lateral_px, ball_width_px, ball_col_4k, target_col_4k, cam_cx_4k = sniper_thread.get_state()

        # ════════════════════════════════════════════════════════════════════
        # 2b. BLEND 4K + HD HEADINGS  (used for CUT mode; STRAIGHT uses visual)
        # ════════════════════════════════════════════════════════════════════
        hd_heading_deg = math.degrees(math.atan2(last_dy, last_dx)) % 360 if initialized else None

        if heading_4k_deg is not None and hd_heading_deg is not None:
            w4k = SNIPER_4K_HEADING_WEIGHT
            whd = 1.0 - SNIPER_4K_HEADING_WEIGHT
            sx = w4k * math.cos(math.radians(heading_4k_deg)) + whd * math.cos(math.radians(hd_heading_deg))
            sy = w4k * math.sin(math.radians(heading_4k_deg)) + whd * math.sin(math.radians(hd_heading_deg))
            blended_heading_deg = math.degrees(math.atan2(sy, sx)) % 360
            source_str = f"4K({heading_4k_n}tags)×{w4k:.2f}+HD×{whd:.2f}"
        elif heading_4k_deg is not None:
            blended_heading_deg = heading_4k_deg
            source_str = f"4K only ({heading_4k_n} tags)"
        elif hd_heading_deg is not None:
            blended_heading_deg = hd_heading_deg
            source_str = "HD only"
        else:
            blended_heading_deg = None
            source_str = "NO HEADING"

        if blended_heading_deg is not None:
            raw_correction = target_heading_deg - blended_heading_deg
            correction = (raw_correction + 180) % 360 - 180
        else:
            correction = 0.0

        # ════════════════════════════════════════════════════════════════════
        # 3. COMPUTE ERRORS
        # ════════════════════════════════════════════════════════════════════
        if initialized:
            x_error = last_x * 1000.0 - rx
            y_error = last_y * 1000.0 - ry
        else:
            x_error = 0.0
            y_error = 0.0

        # ── Three-line error signals (both STRAIGHT and CUT) ─────────────────
        # STRAIGHT: target_col = tag_cx + live_corner_offset  (from run() thread)
        # CUT:      target_col = col_a + ratio*(col_b-col_a)  (from run() thread)
        # In both cases the thread writes _ball_col and _target_col each frame.
        #
        #   diff_error = ball_col − target_col  → vX (strafe)
        #     > 0: ball right of target → MOVE RIGHT
        #     < 0: ball left of target  → MOVE LEFT
        #
        #   mid_error  = (ball_col + target_col)/2 − cam_cx  → vR (rotate)
        #     > 0: midpoint right of cam centre → TURN RIGHT
        #     < 0: midpoint left of cam centre  → TURN LEFT
        #
        # Both zero  ↔  ball_col = target_col = cam_cx  →  perfect alignment.
        # No heading sensor needed for either mode.

        if ball_col_4k is not None and target_col_4k is not None and cam_cx_4k is not None:
            diff_error_px = float(ball_col_4k - target_col_4k)
            mid_error_px  = float((ball_col_4k + target_col_4k) / 2.0 - cam_cx_4k)

            lateral_mm = diff_error_px / lateral_sens if abs(lateral_sens) > 1e-9 else 0.0

            # Convert mid_error (pixels) → degrees for mission_controller vR
            # 1° of rotation shifts both columns by fx*sin(1°)/(depth_m) pixels
            if tag_depth_mm > 0:
                mid_deg = mid_error_px / (fx_4k * math.sin(math.radians(1.0))
                                          / (tag_depth_mm / 1000.0))
            else:
                mid_deg = mid_error_px / 100.0

            correction  = mid_deg   # drives vR in mission_controller
            if SHOT_MODE == "CUT":
                lat_locked  = abs(diff_error_px) < LATERAL_TOL_PX + CUT_MODE_INCR_PX
                ang_locked  = abs(mid_error_px)  < LATERAL_TOL_PX
            else:
                lat_locked  = abs(diff_error_px) < LATERAL_TOL_PX
                ang_locked  = abs(mid_error_px)  < LATERAL_TOL_PX
            pos_locked  = lat_locked and ang_locked

        else:
            diff_error_px = None
            mid_error_px  = None
            lateral_mm    = None
            correction    = 0.0
            lat_locked    = False
            ang_locked    = False
            pos_locked    = False

        # print(f"before broadcast initialized is : {initialized} | x_error={x_error:.1f} | y_error={y_error:.1f} | state={state_text}")

        # ════════════════════════════════════════════════════════════════════
        # 3.5  BROADCAST SNIPER TELEMETRY
        # ════════════════════════════════════════════════════════════════════
        _sniper_pkt = {
            "initialized":   bool(initialized),
            "x_mm":          round(last_x * 1000.0, 1),
            "y_mm":          round(last_y * 1000.0, 1),
            "heading_dx":    round(last_dx, 3),
            "heading_dy":    round(last_dy, 3),
            "state":         state_text,
            "app_state":     "AIMING",
            "sniper_active": True,
            "shot_mode":     SHOT_MODE,
            "target_pocket": target_pocket,
            "target_zone":   target_data.get("target_ball_zone", "?"),
            "ball_color":    ball_color,
            "tag_found_4k":  bool(tag_found_4k),
            "heading_4k_deg":     round(heading_4k_deg, 2) if heading_4k_deg is not None else None,
            "heading_4k_n_tags":  heading_4k_n,
            "heading_hd_deg":     round(hd_heading_deg, 2) if hd_heading_deg is not None else None,
            "blended_heading_deg": round(blended_heading_deg, 2) if blended_heading_deg is not None else None,
            "target_heading_deg": round(target_heading_deg, 2),
            # correction is mid_error_deg for STRAIGHT, heading_error_deg for CUT
            "ang_correction_deg": round(correction, 3) if correction is not None else None,
            "ang_locked":    bool(ang_locked),
            "lateral_mm":    round(lateral_mm, 1) if lateral_mm is not None else None,
            "lat_locked":    bool(lat_locked),
            "ball_width_px": ball_width_px,
            # STRAIGHT-specific: raw pixel errors for debugging
            "straight_diff_px": round(ball_col_4k - target_col_4k, 1) if (ball_col_4k is not None and target_col_4k is not None) else None,
            "straight_mid_px":  round((ball_col_4k + target_col_4k)/2 - cam_cx_4k, 1) if (ball_col_4k is not None and target_col_4k is not None and cam_cx_4k is not None) else None,
            "x_error_mm":    round(x_error, 1),
            "y_error_mm":    round(y_error, 1),
            "pos_locked":    bool(pos_locked),
            "fully_locked":  bool(pos_locked and ang_locked),
        }

        try:
            _telem_sock.sendto(
                json.dumps(_sniper_pkt).encode('utf-8'),
                ("127.0.0.1", SNIPER_TELEMETRY_PORT)
            )
        except Exception as e:
            print(f"[Sniper TELEMETRY ERROR] {e}")

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
            # STRAIGHT MODE HUD — three-line alignment, no heading sensor.
            #
            # STEP 1: ROTATE  — centre the midpoint(ball, target) on camera axis
            #   GREEN=AIM  MAGENTA=TARGET  YELLOW=BALL  (see 4K view)
            #   mid_error = (ball_col + target_col)/2 - cam_cx  → drives vR
            #
            # STEP 2: STRAFE  — collapse ball line onto target line
            #   diff_error = ball_col - target_col  → drives vX
            #
            # When both zero: all three lines coincide = perfect alignment.
            # ────────────────────────────────────────────────────────────────

            # ── STEP 1: Rotate (mid_error → vR) ──────────────────────────────
            if ball_col_4k is None or target_col_4k is None:
                cv2.putText(hud, "STEP 1 — ROTATE  (need ball + tag in view)",
                            (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)
                cv2.putText(hud, "Waiting for ball and tag...",
                            (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 100, 100), 2)
            else:
                mid_px = (ball_col_4k + target_col_4k) / 2.0 - cam_cx_4k
                mid_info = f"  (mid={mid_px:+.0f}px from centre)"
                cv2.putText(hud, f"STEP 1 — ROTATE  (aim midpoint of ball+target){mid_info}",
                            (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
                if ang_locked:
                    cv2.putText(hud, f"ROTATE OK  ({correction:+.2f}°  mid={mid_px:+.0f}px)",
                                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                else:
                    d = "RIGHT" if correction > 0 else "LEFT"
                    cv2.putText(hud, f"TURN {d}:  {abs(correction):.2f}°",
                                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 5)

            # Divider
            cv2.line(hud, (20, 148), (660, 148), (60, 60, 60), 1)

            # ── STEP 2: Strafe (diff_error → vX) ─────────────────────────────
            cv2.putText(hud, "STEP 2 — STRAFE  (close gap: ball line onto target line)",
                        (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)
            if lateral_mm is None:
                cv2.putText(hud, "Waiting for ball and tag...",
                            (30, 218), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 100, 100), 2)
            elif lat_locked:
                cv2.putText(hud, f"STRAFE OK  ({lateral_mm:+.1f} mm)",
                            (30, 218), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            else:
                d = "RIGHT" if lateral_mm > 0 else "LEFT"
                cv2.putText(hud, f"MOVE {d}:  {abs(lateral_mm):.1f} mm",
                            (30, 218), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 5)

            # ── Lock banner ──────────────────────────────────────────────────
            if pos_locked:
                cv2.rectangle(hud, (0, 238), (1280, 310), (0, 70, 0), -1)
                cv2.putText(hud, "LOCKED!  PRESS 'f' TO FIRE",
                            (30, 295), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)

            # HD state — tiny info at bottom
            track_col = (0,200,0) if state_text=="STILL" else (0,140,200) if state_text=="TRANSITIONING" else (0,0,180)
            hd_txt = f"HD: {state_text}  X={x_error:+.0f}mm  Y={y_error:+.0f}mm" if initialized else f"HD: {state_text}"
            cv2.putText(hud, hd_txt, (15, disp_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_col, 1)

        else:
            # ────────────────────────────────────────────────────────────────
            # CUT MODE HUD — identical three-line layout to STRAIGHT.
            # TARGET line = firing-line Y=0 wall intercept (interpolated tags).
            # ────────────────────────────────────────────────────────────────

            cut_label = (f"cut={cut_angle_deg:.0f}°  "
                         f"T{cut_tag_a_id}↔T{cut_tag_b_id}  r={cut_ratio:.2f}"
                         if cut_tag_a_id is not None else f"cut={cut_angle_deg:.0f}°")

            # ── STEP 1: Rotate ────────────────────────────────────────────────
            if ball_col_4k is None or target_col_4k is None:
                cv2.putText(hud, f"STEP 1 — ROTATE  ({cut_label})  need ball+tags in view",
                            (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
                cv2.putText(hud, "Waiting for ball and tag...",
                            (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 100, 100), 2)
            else:
                mid_px = (ball_col_4k + target_col_4k) / 2.0 - cam_cx_4k
                cv2.putText(hud, f"STEP 1 — ROTATE  ({cut_label})  mid={mid_px:+.0f}px",
                            (30, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
                if ang_locked:
                    cv2.putText(hud, f"ROTATE OK  ({correction:+.2f}°  mid={mid_px:+.0f}px)",
                                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                else:
                    d = "RIGHT" if correction > 0 else "LEFT"
                    cv2.putText(hud, f"TURN {d}:  {abs(correction):.2f}°",
                                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 5)

            cv2.line(hud, (20, 148), (660, 148), (60, 60, 60), 1)

            # ── STEP 2: Strafe ────────────────────────────────────────────────
            cv2.putText(hud, "STEP 2 — STRAFE  (close gap: ball line onto firing-line target)",
                        (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)
            if lateral_mm is None:
                cv2.putText(hud, "Waiting for ball and tag...",
                            (30, 218), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 100, 100), 2)
            elif lat_locked:
                cv2.putText(hud, f"STRAFE OK  ({lateral_mm:+.1f} mm)",
                            (30, 218), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            else:
                d = "RIGHT" if lateral_mm > 0 else "LEFT"
                cv2.putText(hud, f"MOVE {d}:  {abs(lateral_mm):.1f} mm",
                            (30, 218), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 5)

            if pos_locked:
                cv2.rectangle(hud, (0, 238), (1280, 310), (0, 70, 0), -1)
                cv2.putText(hud, "LOCKED!  PRESS 'f' TO FIRE",
                            (30, 295), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)

            track_col = (0,200,0) if state_text=="STILL" else (0,140,200) if state_text=="TRANSITIONING" else (0,0,180)
            hd_txt = f"HD: {state_text}  X={x_error:+.0f}mm  Y={y_error:+.0f}mm" if initialized else f"HD: {state_text}"
            cv2.putText(hud, hd_txt, (15, disp_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_col, 1)

        if HAS_DISPLAY:
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
            # CUT: draw a line from target robot position through ghost_ball to pocket.
            # This shows the intended shot geometry on the map.
            cv2.circle(map_canvas, ghost_pt, 6, (255, 255, 255), 2)
            cv2.line(map_canvas, tgt_pt, ghost_pt, (255, 255, 100), 2)
            cv2.line(map_canvas, ghost_pt, pocket_pt, (255, 200, 0), 2)
            cv2.putText(map_canvas, "GHOST", (ghost_pt[0]+6, ghost_pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            # Draw target position dot
            cv2.circle(map_canvas, tgt_pt, 5, mode_col_map, -1)
            cv2.putText(map_canvas, "TARGET", (tgt_pt[0]+8, tgt_pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_col_map, 1)

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
        locked_both = pos_locked
        cone_color = (0, 255, 0) if locked_both else mode_col_map
        overlay = map_canvas.copy()
        cv2.fillPoly(overlay, [np.array([tgt_pt, pt1, pt2], dtype=np.int32)], cone_color)
        cv2.addWeighted(overlay, 0.35, map_canvas, 0.65, 0, map_canvas)
        cv2.line(map_canvas, tgt_pt, pt1, cone_color, 1)
        cv2.line(map_canvas, tgt_pt, pt2, cone_color, 1)

        # ── Circle-intersection rays (dashed grey — distance-based, wider filter) ──
        # Uses raw_circle_estimates so tags excluded from pose tracking due to
        # bounds/jump checks still show their distance circles here.
        # Distance is stable and doesn't need the same strict filtering as pose.
        isect_viz = compute_line_intersections(raw_circle_estimates)

        if isect_viz:
            isect_pt = world_to_pixel(isect_viz[0], isect_viz[1])

            # White ring showing intersection location
            cv2.circle(map_canvas, isect_pt, 6, (255, 255, 255), 2)

            # Dashed rays from tag locations to the intersection
            for entry in raw_circle_estimates:
                tid = entry["id"]

                if tid in KNOWN_TAGS:
                    tag_info = KNOWN_TAGS[tid]
                    pt_tag = world_to_pixel(tag_info["x"], tag_info["y"])

                    x0, y0 = pt_tag
                    x1, y1 = isect_pt

                    steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / 8))

                    for s in range(0, steps, 2):
                        fx0 = int(x0 + (x1 - x0) * s / steps)
                        fy0 = int(y0 + (y1 - y0) * s / steps)

                        fx1 = int(x0 + (x1 - x0) * min(s + 1, steps) / steps)
                        fy1 = int(y0 + (y1 - y0) * min(s + 1, steps) / steps)

                        cv2.line(
                            map_canvas,
                            (fx0, fy0),
                            (fx1, fy1),
                            (200, 200, 200),
                            1
                        )

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

        if HAS_DISPLAY:
            cv2.namedWindow("2D Sniper Map", cv2.WINDOW_NORMAL)
            cv2.imshow("2D Sniper Map", map_canvas)

        # ════════════════════════════════════════════════════════════════════
        # 6. KEY HANDLING
        # ════════════════════════════════════════════════════════════════════
        if HAS_DISPLAY:
            key = cv2.waitKey(1) & 0xFF
        else:
            key = -1

        if key == ord('f') and pos_locked:
            print(f"[{SHOT_MODE}] FIRE!  Angle={correction:.2f}°  "
                  f"X={last_x*1000:.1f}mm  Y={last_y*1000:.1f}mm")
            break
        elif key == 27:
            print("Shot aborted (ESC).")
            break

        try:
            raw_cmd, cmd_addr = fire_sock.recvfrom(1024)
            msg = json.loads(raw_cmd.decode())
            if msg.get("cmd") == "fire":
                print(f"[{SHOT_MODE}] FIRE!  Angle={correction:.2f}°  "
                  f"X={last_x*1000:.1f}mm  Y={last_y*1000:.1f}mm")
                break
        except BlockingIOError:
            pass   # no command waiting — normal path every frame
        except Exception as _cmd_e:
            print(f"[CMD] command error: {_cmd_e}")

    sniper_thread.stop()
    cap_hd.release()
    _telem_sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()