import cv2
import numpy as np
import math
from pupil_apriltags import Detector
from collections import deque
import os
import sys
import time
import socket
import json
import subprocess

# ── Import all settings from central config ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    KNOWN_TAGS, TAG_COLORS, ARENA_BOUNDS,
    HD_DETECTION_ZONES as DETECTION_ZONES,
    CAM_FRONT, TAG_SIZE_M,
    HD_RESOLUTION, HD_FPS, HD_CAM_PARAMS, HD_CAM_DIST_COEFFS,
    HD_CROP_TOP as CROP_TOP_PERCENT,
    HD_CROP_BOT as CROP_BOTTOM_PERCENT,
    HD_MAX_TAG_DIST as MAX_TAG_DISTANCE,
    HD_OUT_OF_BOUNDS as OUT_OF_BOUNDS_LEEWAY,
    HD_MAX_POSE_JUMP as MAX_POSE_JUMP,
    HD_TRACKING_TIMEOUT as TRACKING_TIMEOUT,
    HD_STILL_ALPHA as STILL_ALPHA, HD_MOVE_ALPHA as MOVE_ALPHA,
    HD_STILL_THR as BASE_STILL_THRESHOLD, HD_MOVE_THR as BASE_MOVE_THRESHOLD,
    HD_CONSENSUS_SIGMA as CONSENSUS_SIGMA, HD_HISTORY_LEN,
    HD_CAM_FORWARD_M, HD_CAM_SIDEWAYS_M,
    LINE_INTERSECT_BLEND,
    DETECT_EVERY_N_FRAMES, DISPLAY_SCALE, HD_BRIGHTNESS_BOOST,
    MAP_SCALE as SCALE, MAP_PADDING as PADDING,
    TELEMETRY_PORT, MOVEMENT_HOST, MOVEMENT_CMD_PORT,
    MOVE_DEAD_MM, ROT_DEAD_DEG,
)
# VISION_CMD_PORT and IMU_TELEMETRY_PORT may not exist in older config files
try:
    from config import VISION_CMD_PORT
except ImportError:
    VISION_CMD_PORT = 5007

try:
    from config import IMU_TELEMETRY_PORT
except ImportError:
    IMU_TELEMETRY_PORT = 5009
from tracking import (
    get_global_camera_pose,
    apply_cam_offset,
    fuse_tag_estimates,
    compute_line_intersections,
)
from movement_client import MovementClient, compute_move_cmd

HAS_DISPLAY = bool(os.environ.get("DISPLAY"))

# ── Load position calibration (written by calibrate_position.py) ─────────────
# calibration.json lives in the project root (one level up from this file).
# If absent or invalid, identity transform is used (no correction).
_CAL_FILE = os.path.join(os.path.dirname(__file__), "..", "calibration.json")
_CAL = {
    "scale_x": 1.0, "offset_x_m": 0.0,
    "scale_y": 1.0, "offset_y_m": 0.0,
    # NEW: per-point residual IDW for X and Y
    "position_idw_points": [],
    "position_idw_power": 2,
    # Heading IDW (unchanged)
    "heading_idw_points": [],
    "heading_idw_power": 2,
    "heading_idw_fallback_deg": 0.0,
}
try:
    with open(_CAL_FILE) as _f:
        _loaded = json.load(_f)
        _CAL.update({k: _loaded[k] for k in _CAL if k in _loaded})
    print(f"[new_main3] Calibration loaded from {_CAL_FILE}")
    print(f"  X: scale={_CAL['scale_x']:.4f}  offset={_CAL['offset_x_m']*1000:+.1f}mm")
    print(f"  Y: scale={_CAL['scale_y']:.4f}  offset={_CAL['offset_y_m']*1000:+.1f}mm")
    print(f"  Position IDW: {len(_CAL['position_idw_points'])} residual points")
    print(f"  Heading IDW: {len(_CAL['heading_idw_points'])} pts  "
          f"fallback={_CAL['heading_idw_fallback_deg']:+.3f}°")
except FileNotFoundError:
    print("[new_main3] No calibration.json found — using raw tracker values.")
except Exception as _e:
    print(f"[new_main3] calibration.json load error: {_e} — using raw values.")


def _idw_position_correction(raw_x_m: float, raw_y_m: float):
    """
    Return (corr_x_m, corr_y_m) — the IDW-interpolated residual corrections
    to add on top of the global linear correction.

    Each calibration point stores the residual = true - linear_corrected at
    that point.  IDW interpolates those residuals so that between calibration
    points the correction transitions smoothly.  Far from all calibration
    points (fallback) the correction approaches zero (the global linear handles
    the bulk drift; the IDW only adds the local fine corrections).

    Lookup is done in raw-sensor space (raw_x_m, raw_y_m) using the stored
    raw position of each calibration point — this avoids a chicken-and-egg
    where we'd need the correction to find the correction.
    """
    pts = _CAL["position_idw_points"]
    if not pts:
        return 0.0, 0.0

    raw_x_mm = raw_x_m * 1000.0
    raw_y_mm = raw_y_m * 1000.0
    power = _CAL["position_idw_power"]
    eps = 20.0 ** 2   # 20mm epsilon to avoid division by zero at exact cal points

    total_w = 0.0
    sum_cx  = 0.0
    sum_cy  = 0.0

    for p in pts:
        # Distance in raw-sensor space to this calibration point
        d2 = (raw_x_mm - p["raw_x_mm"])**2 + (raw_y_mm - p["raw_y_mm"])**2
        w  = 1.0 / (d2 + eps) ** power
        total_w += w
        sum_cx  += w * p["residual_x_m"]
        sum_cy  += w * p["residual_y_m"]

    if total_w < 1e-15:
        return 0.0, 0.0

    return sum_cx / total_w, sum_cy / total_w


def _idw_heading_correction(x_m: float, y_m: float) -> float:
    """Position-dependent heading correction via IDW — identical to sniper_4k."""
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

arena_w_meters = ARENA_BOUNDS[2] - ARENA_BOUNDS[0]
arena_h_meters = ARENA_BOUNDS[3] - ARENA_BOUNDS[1]
CANVAS_WIDTH  = int(arena_w_meters * SCALE) + (2 * PADDING)
CANVAS_HEIGHT = int(arena_h_meters * SCALE) + (2 * PADDING)

def world_to_pixel(x, y):
    px = int((x - ARENA_BOUNDS[0]) * SCALE) + PADDING
    py = int((y - ARENA_BOUNDS[1]) * SCALE) + PADDING
    return (px, py)

def draw_map_line(canvas, x1, y1, x2, y2, color=(255, 255, 255), thickness=2):
    cv2.line(canvas, world_to_pixel(x1, y1), world_to_pixel(x2, y2), color, thickness)


def open_hd_camera():
    """Open and configure the HD webcam. Returns cap handle."""
    if sys.platform.startswith('win'):
        cap = cv2.VideoCapture(CAM_FRONT, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAM_FRONT, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  HD_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HD_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, HD_FPS)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    return cap


def main():
    detector = Detector(families="tag36h11")
    tag_size_meters = TAG_SIZE_M

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    TARGET_IP, TARGET_PORT = "127.0.0.1", TELEMETRY_PORT

    # Command socket — mission_controller sends {"cmd":"scan"} here,
    # new_main3 handles it using its own open camera handle.
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_sock.bind(("0.0.0.0", VISION_CMD_PORT))
    cmd_sock.setblocking(False)

    # IMU subscriber — receives snapshots from movement_server at ~30Hz.
    # Non-blocking: we just drain whatever arrived since the last frame.
    # Falls back gracefully if movement_server has no IMU (is_ready stays False).
    _imu_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _imu_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _imu_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    _imu_sock.bind(("0.0.0.0", IMU_TELEMETRY_PORT))
    _imu_sock.setblocking(False)
    _imu_snap = {"is_ready": False, "is_moving": False}  # latest IMU state   # non-blocking: checked every frame, never waits

    # Movement client — sends drive commands to movement_server.py
    mc = MovementClient()

    # -----------------------------------------------------------------------
    # OUTER RESTART LOOP
    # After sniper_4k.py fires and exits, subprocess.run() returns here.
    # We loop back to re-open the HD camera and reset to DRIVING state,
    # creating an infinite test cycle until the table is cleared or 'q' pressed.
    # -----------------------------------------------------------------------
    while True:
        cap = open_hd_camera()

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale_factor = actual_h / 1080.0
        camera_params = [p * scale_factor for p in HD_CAM_PARAMS]

        # ── HD undistortion map (built once, applied every frame) ────────────
        # If HD_CAM_DIST_COEFFS is None, undistortion is skipped (fastest path).
        import numpy as _np
        if HD_CAM_DIST_COEFFS is not None:
            _fx, _fy, _cx, _cy = camera_params
            _cam_mtx = _np.array([[_fx, 0, _cx], [0, _fy, _cy], [0, 0, 1]], dtype=_np.float64)
            _hd_map1, _hd_map2 = cv2.initUndistortRectifyMap(
                _cam_mtx, HD_CAM_DIST_COEFFS, None, _cam_mtx,
                (actual_w, actual_h), cv2.CV_16SC2
            )
        else:
            _hd_map1, _hd_map2 = None, None

        scale_x = actual_w / 1920.0
        scale_y = actual_h / 1080.0

        history_raw_x, history_raw_y = deque(maxlen=HD_HISTORY_LEN), deque(maxlen=HD_HISTORY_LEN)
        last_x, last_y, last_dx, last_dy = 0.0, 0.0, 1.0, 0.0
        initialized = False
        last_seen_time = time.time()

        # --- STATE MACHINE VARIABLES ---
        APP_STATE = "DRIVING"  # "DRIVING" or "AIMING"
        shooting_plans = {}
        matched_target_zone = None

        print("\n[HD Navigator] Online. Press 's' to scan targets, 'q' to quit.")

        quit_requested = False  # propagate 'q' out of the inner loop
        _frame_count = 0        # for DETECT_EVERY_N_FRAMES skipping
        debug_tag_estimates = []  # holds last detection result; reused on skip frames
        raw_circle_estimates = [] # distance-only filtered; used for circle intersection viz

        # ---- INNER FRAME LOOP ----
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Brightness compensation for manual low-exposure setting
            if HD_BRIGHTNESS_BOOST != 1.0:
                frame = cv2.convertScaleAbs(frame, alpha=HD_BRIGHTNESS_BOOST)

            # Apply HD lens undistortion if coefficients are configured
            if _hd_map1 is not None:
                frame = cv2.remap(frame, _hd_map1, _hd_map2, cv2.INTER_LINEAR)

            # Keep a pristine, un-drawn-on frame for the scanner
            clean_frame = frame.copy()

            h, w = frame.shape[:2]
            
            # Draw the Ball Detection Zones overlay on the HD Camera
            for z_name, (zx, zy, zw, zh) in DETECTION_ZONES.items():
                sx = int(zx * scale_x)
                sy = int(zy * scale_y)
                sw = int(zw * scale_x)
                sh = int(zh * scale_y)

                cv2.rectangle(
                    frame,
                    (sx, sy),
                    (sx + sw, sy + sh),
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    z_name,
                    (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            y_start, y_end = int(h * CROP_TOP_PERCENT), int(h * (1.0 - CROP_BOTTOM_PERCENT))

            # ── AprilTag detection (skip frames on Pi for performance) ────
            _frame_count += 1
            run_detection = (_frame_count % DETECT_EVERY_N_FRAMES == 0)
            if run_detection:
                gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
                gray_cropped = gray[y_start:y_end, :]
                adj_camera_params = [camera_params[0], camera_params[1], camera_params[2], camera_params[3] - y_start]
                tags = detector.detect(gray_cropped, estimate_tag_pose=True, camera_params=adj_camera_params, tag_size=tag_size_meters)
                debug_tag_estimates = []   # fresh results this frame
                raw_circle_estimates = []  # fresh raw list (distance-only filtered)
            else:
                tags = []   # skip detection; keep previous lists for display

            for tag in tags:
                tag.corners[:, 1] += y_start
                tag.center[1] += y_start
                
                # Draw Tag Bounding Boxes directly on the video feed
                tag_color = TAG_COLORS.get(tag.tag_id, (0, 255, 0))
                corners = tag.corners.astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), tag_color, 2)
                center = tuple(tag.center.astype(int))
                cv2.circle(frame, center, 4, (0, 0, 255), -1)

                t, R = tag.pose_t, tag.pose_R
                pose_data = get_global_camera_pose(tag.tag_id, t, R)
                
                if pose_data:
                    dist = math.sqrt(t[0][0]**2 + t[1][0]**2 + t[2][0]**2)
                    if dist > MAX_TAG_DISTANCE:
                        continue
                    cam_x, cam_y, gdx, gdy = pose_data
                    # Apply camera-to-robot offset
                    robot_x, robot_y = apply_cam_offset(cam_x, cam_y, gdx, gdy,
                                                         HD_CAM_FORWARD_M, HD_CAM_SIDEWAYS_M)

                    # ── Raw entry for circle intersection ─────────────────
                    # Distance is reliable regardless of pose quality, so we keep
                    # ALL distance-valid tags here even if the pose is rejected below.
                    # The circle intersection only uses tag world position + distance,
                    # not the estimated camera pose.
                    raw_circle_estimates.append({
                        "id": tag.tag_id, "dist": dist,
                        # pose field needed by compute_distance_circles for the
                        # tag-to-camera direction used in weight calculation only
                        "pose": (robot_x, robot_y, gdx, gdy),
                    })

                    # ── Pose-based entry — apply bounds + jump filter ──────
                    if not (ARENA_BOUNDS[0] - OUT_OF_BOUNDS_LEEWAY <= robot_x <= ARENA_BOUNDS[2] + OUT_OF_BOUNDS_LEEWAY and
                            ARENA_BOUNDS[1] - OUT_OF_BOUNDS_LEEWAY <= robot_y <= ARENA_BOUNDS[3] + OUT_OF_BOUNDS_LEEWAY):
                        continue
                    if initialized and math.hypot(robot_x - last_x, robot_y - last_y) > MAX_POSE_JUMP:
                        continue
                    robot_pose = (robot_x, robot_y, gdx, gdy)
                    w = 1.0 / (dist**3 + 0.001)
                    debug_tag_estimates.append({
                        "id": tag.tag_id, "pose": robot_pose,
                        "dist": dist, "w": w
                    })
                    text = f"ID:{tag.tag_id} -> Dist:{dist:.2f}m"
                    cv2.putText(frame, text, (center[0] - 50, center[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color, 2)

            if len(debug_tag_estimates) > 0:
                last_seen_time = time.time()

                # ── Fuse tag estimates (consensus + dist^2 weighting) ──────
                fused = fuse_tag_estimates(debug_tag_estimates, last_x, last_y, initialized)
                if fused:
                    raw_avg_x, raw_avg_y, raw_avg_dx, raw_avg_dy, sum_distances, n = fused

                    # ── Circle-intersection extra estimate ─────────────────
                    # Uses raw_circle_estimates (distance-only filtered) so that
                    # tags excluded by the pose bounds/jump filter still contribute
                    # to the distance-based circle intersection — distance is stable
                    # even when the pose estimate is unreliable.
                    isect = compute_line_intersections(raw_circle_estimates)
                    if isect and LINE_INTERSECT_BLEND > 0:
                        raw_avg_x = (1-LINE_INTERSECT_BLEND)*raw_avg_x + LINE_INTERSECT_BLEND*isect[0]
                        raw_avg_y = (1-LINE_INTERSECT_BLEND)*raw_avg_y + LINE_INTERSECT_BLEND*isect[1]

                    # ── Apply calibration to raw value BEFORE smoothing ────
                    # MUST be here. If applied to last_x/last_y after the filter,
                    # the scale factor multiplies itself every frame → drift.
                    # Step 1: global linear correction (bulk drift + scale)
                    raw_avg_x = _CAL["scale_x"] * raw_avg_x + _CAL["offset_x_m"]
                    raw_avg_y = _CAL["scale_y"] * raw_avg_y + _CAL["offset_y_m"]
                    # Step 2: IDW residual correction (local zone errors)
                    # Each calibration point stored its residual = true - linear_corrected.
                    # IDW interpolates those residuals spatially so each zone of the table
                    # gets its own fine correction on top of the global linear.
                    _res_x, _res_y = _idw_position_correction(raw_avg_x, raw_avg_y)
                    raw_avg_x += _res_x
                    raw_avg_y += _res_y
                    # Heading IDW correction — uses the fully corrected position
                    _raw_h  = math.degrees(math.atan2(raw_avg_dy, raw_avg_dx)) % 360
                    _h_corr = _idw_heading_correction(raw_avg_x, raw_avg_y)
                    _cal_h  = (_raw_h + _h_corr) % 360
                    raw_avg_dx = math.cos(math.radians(_cal_h))
                    raw_avg_dy = math.sin(math.radians(_cal_h))

                    history_raw_x.append(raw_avg_x); history_raw_y.append(raw_avg_y)
                    if not initialized:
                        last_x, last_y, last_dx, last_dy = raw_avg_x, raw_avg_y, raw_avg_dx, raw_avg_dy
                        initialized = True

                    # ── Drain IMU socket (keep only the most recent packet) ──
                    # Non-blocking: if movement_server hasn't sent yet this frame,
                    # we just keep the previous _imu_snap — no delay.
                    while True:
                        try:
                            _raw, _ = _imu_sock.recvfrom(256)
                            _imu_snap = json.loads(_raw.decode())
                        except BlockingIOError:
                            break
                        except Exception:
                            break

                    # ── STILL / MOVING ─────────────────────────────────────
                    # Use IMU accel if the IMU is calibrated and ready.
                    # IMU is much faster (~0.1s response) than the camera std-window (~0.5s).
                    # Fall back to the camera std-window if IMU is not available.
                    if _imu_snap.get("is_ready"):
                        if _imu_snap.get("is_moving"):
                            alpha, state_text = MOVE_ALPHA, "MOVING"
                        else:
                            alpha, state_text = STILL_ALPHA, "STILL"
                    else:
                        # Camera std-window fallback (original logic)
                        movement_metric = math.sqrt(np.std(history_raw_x)**2 + np.std(history_raw_y)**2) if len(history_raw_x) >= 2 else 0.0
                        avg_dist = sum_distances / n
                        dist_multiplier = max(1.0, min(avg_dist ** 2, 12.0))
                        still_threshold = BASE_STILL_THRESHOLD * dist_multiplier
                        move_threshold  = BASE_MOVE_THRESHOLD  * dist_multiplier
                        if movement_metric <= still_threshold:
                            alpha, state_text = STILL_ALPHA, "STILL"
                        elif movement_metric >= move_threshold:
                            alpha, state_text = MOVE_ALPHA, "MOVING"
                        else:
                            ratio = (movement_metric - still_threshold) / (move_threshold - still_threshold)
                            alpha, state_text = STILL_ALPHA + ratio * (MOVE_ALPHA - STILL_ALPHA), "TRANSITIONING"
                        
                    last_x = alpha * raw_avg_x + (1.0 - alpha) * last_x
                    last_y = alpha * raw_avg_y + (1.0 - alpha) * last_y
                    last_dx = alpha * raw_avg_dx + (1.0 - alpha) * last_dx
                    last_dy = alpha * raw_avg_dy + (1.0 - alpha) * last_dy
                    
                    mag_smooth = math.sqrt(last_dx**2 + last_dy**2)
                    if mag_smooth > 0: last_dx /= mag_smooth; last_dy /= mag_smooth

                    text_color = (0, 255, 0)
            else:
                if initialized and (time.time() - last_seen_time > TRACKING_TIMEOUT):
                    initialized = False; history_raw_x.clear(); history_raw_y.clear()
                state_text = "LOST"
                text_color = (0, 0, 255)

            # On-screen telemetry readout
            curr_heading_deg = (math.degrees(math.atan2(last_dy, last_dx)) + 360) % 360
            cv2.putText(frame, f"State: {state_text} | Pos: ({last_x*100:.1f}cm, {last_y*100:.1f}cm) | Head: {curr_heading_deg:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2)

            # ── Compute movement command for AIMING state ────────────────
            move_cmd_dict = {"cmd": "stop"}
            if APP_STATE == "AIMING" and initialized and matched_target_zone is not None:
                plan = shooting_plans[matched_target_zone]
                plan_x, plan_y = plan["robot_alignment_xy"]
                plan_head = plan["firing_angle_deg"]
                curr_x_mm = last_x * 1000; curr_y_mm = last_y * 1000
                curr_head = (math.degrees(math.atan2(last_dy, last_dx)) + 360) % 360
                ang_err = curr_head - plan_head
                if ang_err >  180: ang_err -= 360
                if ang_err < -180: ang_err += 360
                # Transform world errors into robot frame
                x_err_w = curr_x_mm - plan_x; y_err_w = curr_y_mm - plan_y
                side_err = -x_err_w * last_dy + y_err_w * last_dx
                fwd_err  =  x_err_w * last_dx + y_err_w * last_dy
                cmd = compute_move_cmd(x_error_mm=side_err, y_error_mm=-fwd_err,
                                       ang_error_deg=ang_err)
                move_cmd_dict = cmd.to_dict()
                mc.send(cmd)

            # UDP Telemetry Broadcast (extended packet)
            telemetry_packet = {
                "initialized": initialized,
                "x_mm":  round(last_x * 1000, 1),
                "y_mm":  round(last_y * 1000, 1),
                "x_cm":  round(last_x * 100, 2),   # backwards compat
                "y_cm":  round(last_y * 100, 2),
                "heading_dx": round(last_dx, 3),
                "heading_dy": round(last_dy, 3),
                "state":      state_text,
                "app_state":  APP_STATE,
                "movement_cmd": move_cmd_dict,
            }
            try:
                sock.sendto(json.dumps(telemetry_packet).encode('utf-8'), (TARGET_IP, TARGET_PORT))
            except:
                pass

            # --- STATE MACHINE UI & KEYBINDS ---
            if HAS_DISPLAY:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            # ── Check for remote commands from mission_controller ──────────
            # Non-blocking: if no command waiting, continues immediately.
            try:
                raw_cmd, cmd_addr = cmd_sock.recvfrom(1024)
                msg = json.loads(raw_cmd.decode())
                if msg.get("cmd") == "scan":
                    # Treat exactly like pressing 's' — use already-captured frame
                    print("[CMD] Remote scan triggered by mission_controller")
                    cv2.imwrite("current_scan.jpg", clean_frame)
                    # Pass current robot position so the planner uses projected zones
                    _hdg = (math.degrees(math.atan2(last_dy, last_dx)) + 360) % 360

                    # ── Find tag 4 pixel in the scan frame ────────────────────
                    # Re-detect tags on the pristine scan frame so the pixel
                    # position is unaffected by any overlaid graphics.
                    _scan_gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
                    _scan_tags = detector.detect(_scan_gray, estimate_tag_pose=False,
                                                 camera_params=None, tag_size=None)
                    _tag4_px = None
                    for _t in _scan_tags:
                        if _t.tag_id == 4:
                            _tag4_px = float(_t.center[0])
                            print(f"[CMD] Tag 4 found at pixel x={_tag4_px:.1f}")
                            break
                    if _tag4_px is None:
                        print("[CMD] Tag 4 not found in scan frame — using raw projection")

                    _planner_args = [
                        sys.executable, "snooker_planner.py", "current_scan.jpg",
                        "--robot_x", str(round(last_x * 1000, 1)),
                        "--robot_y", str(round(last_y * 1000, 1)),
                        "--heading", str(round(_hdg, 2)),
                    ]
                    if _tag4_px is not None:
                        _planner_args += ["--tag4_px", str(round(_tag4_px, 1))]
                    subprocess.run(_planner_args)
                    if os.path.exists("robot_shooting_plan.json"):
                        with open("robot_shooting_plan.json", "r") as f:
                            shooting_plans = json.load(f)
                        APP_STATE = "AIMING"
                        print(f"[CMD] Scan done. Plans: {list(shooting_plans.keys())}")
                    else:
                        print("[CMD] Scan done. No plan generated.")
                    # Reply so do_scan() knows it's finished
                    cmd_sock.sendto(json.dumps({"status": "done"}).encode(), cmd_addr)

                elif msg.get("cmd") == "confirm_shot":
                    # Treat exactly like pressing 'c' — release camera, run sniper,
                    # reply when sniper exits so mission_controller knows it's done.
                    target_data = msg.get("target_data", {})
                    print(f"[CMD] confirm_shot received for zone {target_data.get('target_ball_zone','?')}")
                    with open("selected_target.json", "w") as _sf:
                        json.dump(target_data, _sf)
                    print("Shot Confirmed! Releasing HD camera → handing to 4K Sniper...")
                    cap.release()
                    cv2.destroyAllWindows()
                    subprocess.run([sys.executable, "sniper_4k.py"])
                    # Sniper exited — reply to mission_controller
                    cmd_sock.sendto(json.dumps({"status": "sniper_done"}).encode(), cmd_addr)
                    print("[HD Navigator] Sniper exited. Restarting in DRIVING mode...")
                    break   # break inner loop → outer loop re-opens HD camera

            except BlockingIOError:
                pass   # no command waiting — normal path every frame
            except Exception as _cmd_e:
                print(f"[CMD] command error: {_cmd_e}")
            
            if APP_STATE == "DRIVING":
                cv2.putText(frame, "STATE: DRIVING. Press 's' to Scan Targets", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                if key == ord('s'):
                    print("Scanning table...")
                    cv2.imwrite("current_scan.jpg", clean_frame)
                    _hdg = (math.degrees(math.atan2(last_dy, last_dx)) + 360) % 360

                    # ── Find tag 4 pixel in the scan frame ────────────────────
                    _scan_gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
                    _scan_tags = detector.detect(_scan_gray, estimate_tag_pose=False,
                                                 camera_params=None, tag_size=None)
                    _tag4_px = None
                    for _t in _scan_tags:
                        if _t.tag_id == 4:
                            _tag4_px = float(_t.center[0])
                            print(f"[S-key] Tag 4 found at pixel x={_tag4_px:.1f}")
                            break
                    if _tag4_px is None:
                        print("[S-key] Tag 4 not found in scan frame — using raw projection")

                    _planner_args = [
                        sys.executable, "snooker_planner.py", "current_scan.jpg",
                        "--robot_x", str(round(last_x * 1000, 1)),
                        "--robot_y", str(round(last_y * 1000, 1)),
                        "--heading", str(round(_hdg, 2)),
                    ]
                    if _tag4_px is not None:
                        _planner_args += ["--tag4_px", str(round(_tag4_px, 1))]
                    subprocess.run(_planner_args)
                    
                    if os.path.exists("robot_shooting_plan.json"):
                        with open("robot_shooting_plan.json", "r") as f:
                            shooting_plans = json.load(f)
                        APP_STATE = "AIMING"
                        if list(shooting_plans.keys()):
                            print(f"Loaded plans for zones: {list(shooting_plans.keys())}")
                        else:
                            print("no zones detected")
                            sys.exit()

            elif APP_STATE == "AIMING":
                matched_target_zone = None
                curr_x_mm, curr_y_mm = last_x * 1000, last_y * 1000
                curr_heading_deg = (math.degrees(math.atan2(last_dy, last_dx)) + 360) % 360
                
                # Check if we are parked correctly
                for zone, plan in shooting_plans.items():
                    plan_x, plan_y = plan["robot_alignment_xy"]
                    plan_head = plan["firing_angle_deg"]
                    
                    dist_err = math.hypot(curr_x_mm - plan_x, curr_y_mm - plan_y)
                    head_err = abs(curr_heading_deg - plan_head)
                    if head_err > 180: head_err = 360 - head_err
                    
                    if dist_err < 60.0 and head_err < 5.0:
                        matched_target_zone = zone
                        break
                
                if matched_target_zone:
                    target_p = shooting_plans[matched_target_zone]["target_pocket"]
                    cv2.putText(frame, f"ALIGNED: Zone {matched_target_zone} -> {target_p}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(frame, "PRESS 'c' TO CONFIRM SHOT & SWITCH TO 4K", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if key == ord('c'):
                        with open("selected_target.json", "w") as f:
                            json.dump(shooting_plans[matched_target_zone], f)
                        
                        print("Shot Confirmed! Releasing HD camera → handing to 4K Sniper...")
                        cap.release()
                        cv2.destroyAllWindows()
                        subprocess.run([sys.executable, "sniper_4k.py"])
                        
                        # sniper_4k.py has exited (fired or ESC'd).
                        # Break the inner loop → outer loop will re-open HD camera in DRIVING state.
                        print("[HD Navigator] Sniper exited. Restarting in DRIVING mode...")
                        break
                else:
                    cv2.putText(frame, "STATE: AIMING. Check 2D Map to find Target.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            # =========================================================================
            # --- DRAW THE 2D REAL-TIME MAP ---
            # =========================================================================
            map_canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
            
            # Draw boundaries and grid
            pt_top_left = world_to_pixel(ARENA_BOUNDS[0], ARENA_BOUNDS[1])
            pt_bottom_right = world_to_pixel(ARENA_BOUNDS[2], ARENA_BOUNDS[3])
            cv2.rectangle(map_canvas, pt_top_left, pt_bottom_right, (0, 150, 150), 2)
            for grid_step in np.arange(ARENA_BOUNDS[0], ARENA_BOUNDS[2] + 0.1, 0.5):
                cv2.line(map_canvas, world_to_pixel(grid_step, ARENA_BOUNDS[1]), world_to_pixel(grid_step, ARENA_BOUNDS[3]), (40, 40, 40), 1)
            for grid_step in np.arange(ARENA_BOUNDS[1], ARENA_BOUNDS[3] + 0.1, 0.5):
                cv2.line(map_canvas, world_to_pixel(ARENA_BOUNDS[0], grid_step), world_to_pixel(ARENA_BOUNDS[2], grid_step), (40, 40, 40), 1)

            draw_map_line(map_canvas, x1=0.0, y1=2.1, x2=1.2, y2=2.1, color=(255, 255, 255), thickness=2)
            draw_map_line(map_canvas, x1=0.0, y1=2.4, x2=1.2, y2=2.4, color=(0, 150, 0), thickness=2)
            draw_map_line(map_canvas, x1=0.0, y1=2.5, x2=1.2, y2=2.5, color=(150, 150, 150), thickness=2)
                
            # Draw AprilTags
            for tag_id, info in KNOWN_TAGS.items():
                pt = world_to_pixel(info["x"], info["y"])
                color = TAG_COLORS.get(tag_id, (0, 0, 255))
                cv2.rectangle(map_canvas, (pt[0]-6, pt[1]-6), (pt[0]+6, pt[1]+6), color, -1)
                cv2.putText(map_canvas, f"ID {tag_id}", (pt[0]-14, pt[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # ── Per-tag tracking lines (solid, tag colour) ──────────────
            for entry in debug_tag_estimates:
                tag_id = entry["id"]
                pose = entry["pose"]
                
                if tag_id in KNOWN_TAGS:
                    tag_info = KNOWN_TAGS[tag_id]
                    pt_tag_real = world_to_pixel(tag_info["x"], tag_info["y"])
                    pt_cam_pred = world_to_pixel(pose[0], pose[1])
                    color = TAG_COLORS.get(tag_id, (255, 255, 255))
                    
                    cv2.line(map_canvas, pt_tag_real, pt_cam_pred, color, 1)
                    cv2.circle(map_canvas, pt_cam_pred, 3, color, -1)
                    
                    arrow_length = 0.15 
                    pt_arrow_end = world_to_pixel(pose[0] + pose[2] * arrow_length, pose[1] + pose[3] * arrow_length)
                    cv2.arrowedLine(map_canvas, pt_cam_pred, pt_arrow_end, color, 1, tipLength=0.25)

            # ── Circle-intersection rays (dashed grey — distance-based, wider filter) ──
            # Uses raw_circle_estimates so tags excluded from pose tracking due to
            # bounds/jump checks still show their distance circles here.
            # Distance is stable and doesn't need the same strict filtering as pose.
            isect_viz = compute_line_intersections(raw_circle_estimates)
            if isect_viz:
                isect_pt = world_to_pixel(isect_viz[0], isect_viz[1])
                cv2.circle(map_canvas, isect_pt, 6, (255, 255, 255), 2)   # white ring
                # Dashed lines from each tag (raw list) toward intersection point
                for entry in raw_circle_estimates:
                    tid = entry["id"]
                    if tid in KNOWN_TAGS:
                        ti = KNOWN_TAGS[tid]
                        pt_tag = world_to_pixel(ti["x"], ti["y"])
                        x0, y0 = pt_tag; x1, y1 = isect_pt
                        steps = max(1, int(math.hypot(x1-x0, y1-y0) / 8))
                        for s in range(0, steps, 2):
                            fx0 = int(x0 + (x1-x0)*s/steps)
                            fy0 = int(y0 + (y1-y0)*s/steps)
                            fx1 = int(x0 + (x1-x0)*min(s+1,steps)/steps)
                            fy1 = int(y0 + (y1-y0)*min(s+1,steps)/steps)
                            cv2.line(map_canvas, (fx0, fy0), (fx1, fy1), (200, 200, 200), 1)
         
            # --- DYNAMIC PLAN VISUALIZER (AIMING state only) ---
            if APP_STATE == "AIMING" and shooting_plans:
                curr_x_mm, curr_y_mm = last_x * 1000, last_y * 1000
                curr_heading_deg = (math.degrees(math.atan2(last_dy, last_dx)) + 360) % 360
                
                for zone, plan in shooting_plans.items():
                    plan_x, plan_y = plan["robot_alignment_xy"]
                    plan_head = plan["firing_angle_deg"]
                    
                    dist_err = math.hypot(curr_x_mm - plan_x, curr_y_mm - plan_y)
                    head_err = abs(curr_heading_deg - plan_head)
                    if head_err > 180: head_err = 360 - head_err
                    
                    if dist_err < 60.0 and head_err < 5.0:
                        p_color = (0, 255, 0)      # Green (Fully Aligned)
                    elif dist_err < 60.0:
                        p_color = (0, 255, 255)    # Yellow (Good position, Wrong rotation)
                    elif dist_err < 200.0:
                        p_color = (0, 165, 255)    # Orange (Approaching)
                    else:
                        p_color = (0, 0, 255)      # Red (Far Away)
                    
                    center_pt = world_to_pixel(plan_x / 1000.0, plan_y / 1000.0)
                    
                    # Draw +/- 60mm tolerance circle around target
                    tol_px = int(0.06 * SCALE)
                    cv2.circle(map_canvas, center_pt, tol_px, p_color, 2)
                    cv2.putText(map_canvas, f"Zone {zone}", (center_pt[0]+tol_px+5, center_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, p_color, 1)
                    
                    # Draw Aiming Cone (+/- 5 degrees sweep)
                    cone_len = 0.35  # 35cm visual length
                    a_min = math.radians(plan_head - 5.0)
                    a_max = math.radians(plan_head + 5.0)
                    
                    pt1 = world_to_pixel((plan_x/1000.0) + cone_len * math.cos(a_min), (plan_y/1000.0) + cone_len * math.sin(a_min))
                    pt2 = world_to_pixel((plan_x/1000.0) + cone_len * math.cos(a_max), (plan_y/1000.0) + cone_len * math.sin(a_max))
                    
                    # Draw semi-transparent cone fill
                    overlay = map_canvas.copy()
                    pts = np.array([center_pt, pt1, pt2], dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], p_color)
                    cv2.addWeighted(overlay, 0.4, map_canvas, 0.6, 0, map_canvas)
                    cv2.line(map_canvas, center_pt, pt1, p_color, 1)
                    cv2.line(map_canvas, center_pt, pt2, p_color, 1)
            
            # Draw live robot on top of everything
            if initialized:
                cam_pt = world_to_pixel(last_x, last_y)
                cv2.circle(map_canvas, cam_pt, 10, (255, 80, 80), -1)
                heading_end = world_to_pixel(last_x + last_dx * 0.25, last_y + last_dy * 0.25)
                cv2.arrowedLine(map_canvas, cam_pt, heading_end, (0, 255, 0), 2, tipLength=0.2)
                cv2.putText(map_canvas, "Robot", (cam_pt[0]-16, cam_pt[1]+24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Show Both Windows (scale down for Pi GPU performance)
            if HAS_DISPLAY:
                if DISPLAY_SCALE != 1.0:
                    dw = int(frame.shape[1] * DISPLAY_SCALE)
                    dh = int(frame.shape[0] * DISPLAY_SCALE)
                    cv2.imshow("HD Navigator", cv2.resize(frame, (dw, dh)))

                    mw = int(map_canvas.shape[1] * DISPLAY_SCALE)
                    mh = int(map_canvas.shape[0] * DISPLAY_SCALE)
                    cv2.imshow(
                        "2D Real-Time Field Map",
                        cv2.resize(map_canvas, (mw, mh))
                    )
                else:
                    cv2.imshow("HD Navigator", frame)
                    cv2.imshow("2D Real-Time Field Map", map_canvas)
            
            if key == ord('q'):
                quit_requested = True
                break

        # ---- End of inner frame loop ----
        cap.release()
        cv2.destroyAllWindows()

        if quit_requested:
            break
        # else: outer loop continues → re-opens HD camera in DRIVING state


if __name__ == '__main__':
    main()