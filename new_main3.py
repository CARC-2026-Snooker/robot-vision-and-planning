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

# --- 1. DEFINE YOUR FIELD / KNOWN TAGS ---
KNOWN_TAGS = { 
    1: {"x": 1.262, "y": 1.05, "facing_angle": 270},
    2: {"x": -0.062, "y": 1.05, "facing_angle": 270},
    3: {"x": 0, "y": 0, "facing_angle": 270},
    4: {"x": 0.6, "y": 0, "facing_angle": 270},
    5: {"x": 1.2, "y": 0, "facing_angle": 270},
}

# Unique BGR colors assigned to each tag ID
TAG_COLORS = {
    1: (255, 105, 180), # Hot Pink
    2: (0, 255, 255),   # Yellow
    3: (0, 165, 255),   # Orange
    4: (255, 255, 0),   # Cyan
    5: (255, 0, 255),   # Magenta
}

# Format: (x_min, y_min, x_max, y_max) in meters.
ARENA_BOUNDS = (0.0, 0.0, 1.2, 3.6)

# --- 2. DETECTION ZONES (For Visual Overlay) ---
DETECTION_ZONES = {
    "B": (390, 590, 100, 100),
    "C": (620, 550, 70, 70),
    "D": (810, 550, 70, 70),
    "E": (1000, 560, 70, 70),
    "F": (1180, 550, 70, 70),
    "G": (1380, 550, 70, 70),
    "A": (1590, 590, 100, 100),
}

MAX_TAG_DISTANCE = 10.0      
OUT_OF_BOUNDS_LEEWAY = 0.15  
MAX_POSE_JUMP = 0.5       
TRACKING_TIMEOUT = 0.3       

BASE_STILL_THRESHOLD = 0.005   
BASE_MOVE_THRESHOLD = 0.030    
STILL_ALPHA = 0.005            
MOVE_ALPHA = 0.25

# Consensus weighting sigma (metres): tags whose estimate deviates more than this
# from the distance-weighted mean are soft-rejected. 45mm is generous for real noise.
CONSENSUS_SIGMA = 0.045

# Exposure brightness compensation — see sniper_4k.py for explanation.
HD_BRIGHTNESS_BOOST = 1.8   # 1.0 = off, 1.5–2.5 = typical low-exposure range

CROP_TOP_PERCENT = 0.20       
CROP_BOTTOM_PERCENT = 0.20    
MASK_ALPHA = 0.3              

SCALE = 200    
PADDING = 60   

arena_w_meters = ARENA_BOUNDS[2] - ARENA_BOUNDS[0]
arena_h_meters = ARENA_BOUNDS[3] - ARENA_BOUNDS[1]

CANVAS_WIDTH = int(arena_w_meters * SCALE) + (2 * PADDING)
CANVAS_HEIGHT = int(arena_h_meters * SCALE) + (2 * PADDING)

def get_global_camera_pose(tag_id, t, R):
    if tag_id not in KNOWN_TAGS: return None
    cam_pos_in_tag_frame = -np.transpose(R) @ t
    local_x, local_z = cam_pos_in_tag_frame[0][0], cam_pos_in_tag_frame[2][0]
    tag_info = KNOWN_TAGS[tag_id]
    theta = math.radians(tag_info["facing_angle"])
    global_cam_x = tag_info["x"] + (local_z * math.cos(theta)) - (local_x * math.sin(theta))
    global_cam_y = tag_info["y"] + (local_z * math.sin(theta)) + (local_x * math.cos(theta))
    tag_space_dx, tag_space_dz = R[2, 0], R[2, 2]
    global_dx = (tag_space_dz * math.cos(theta)) - (tag_space_dx * math.sin(theta))
    global_dy = (tag_space_dz * math.sin(theta)) + (tag_space_dx * math.cos(theta))
    return (global_cam_x, global_cam_y, global_dx, global_dy)

def world_to_pixel(x, y):
    px = int((x - ARENA_BOUNDS[0]) * SCALE) + PADDING
    py = int((y - ARENA_BOUNDS[1]) * SCALE) + PADDING
    return (px, py)

def draw_map_line(canvas, x1, y1, x2, y2, color=(255, 255, 255), thickness=2):
    cv2.line(canvas, world_to_pixel(x1, y1), world_to_pixel(x2, y2), color, thickness)


def open_hd_camera():
    """Open and configure the HD webcam. Returns cap handle."""
    if sys.platform.startswith('win'):
        cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

    # ORIGINAL OPTIMIZED VIDEO CAP SETTINGS — do not change
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # setting exposure manually so less blur on movement
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0) 
    cap.set(cv2.CAP_PROP_EXPOSURE, -4.0) # Try values between -1 and -9 

    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    return cap


def main():
    detector = Detector(families="tag36h11")
    tag_size_meters = 0.11

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    TARGET_IP, TARGET_PORT = "127.0.0.1", 5005

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
        base_params = [1996.94*0.99444, 1998.60*0.99444, 827.66, 460.60]
        scale_factor = actual_h / 1080.0
        camera_params = [p * scale_factor for p in base_params]

        scale_x = actual_w / 1920.0
        scale_y = actual_h / 1080.0

        history_raw_x, history_raw_y = deque(maxlen=15), deque(maxlen=15)
        last_x, last_y, last_dx, last_dy = 0.0, 0.0, 1.0, 0.0
        initialized = False
        last_seen_time = time.time()

        # --- STATE MACHINE VARIABLES ---
        APP_STATE = "DRIVING"  # "DRIVING" or "AIMING"
        shooting_plans = {}
        matched_target_zone = None

        print("\n[HD Navigator] Online. Press 's' to scan targets, 'q' to quit.")

        quit_requested = False  # propagate 'q' out of the inner loop

        # ---- INNER FRAME LOOP ----
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Brightness compensation for manual low-exposure setting
            if HD_BRIGHTNESS_BOOST != 1.0:
                frame = cv2.convertScaleAbs(frame, alpha=HD_BRIGHTNESS_BOOST)

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
            gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
            gray_cropped = gray[y_start:y_end, :]

            adj_camera_params = [camera_params[0], camera_params[1], camera_params[2], camera_params[3] - y_start]
            tags = detector.detect(gray_cropped, estimate_tag_pose=True, camera_params=adj_camera_params, tag_size=tag_size_meters)
            debug_tag_estimates = []

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
                    
                    global_cam_x, global_cam_y, global_dx, global_dy = pose_data
                    
                    if not (ARENA_BOUNDS[0] - OUT_OF_BOUNDS_LEEWAY <= global_cam_x <= ARENA_BOUNDS[2] + OUT_OF_BOUNDS_LEEWAY and
                            ARENA_BOUNDS[1] - OUT_OF_BOUNDS_LEEWAY <= global_cam_y <= ARENA_BOUNDS[3] + OUT_OF_BOUNDS_LEEWAY):
                        continue
                    
                    if initialized:
                        jump_dist = math.sqrt((global_cam_x - last_x)**2 + (global_cam_y - last_y)**2)
                        if jump_dist > MAX_POSE_JUMP:
                            continue 
                    
                    debug_tag_estimates.append({
                        "id": tag.tag_id,
                        "pose": pose_data,
                        "distance": dist
                    })

                    text = f"ID:{tag.tag_id} -> Dist:{dist:.2f}m"
                    cv2.putText(frame, text, (center[0] - 50, center[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color, 2)

            if len(debug_tag_estimates) > 0:
                last_seen_time = time.time()

                # ── Step 1: distance-weighted mean (initial estimate) ──────
                total_weight_init = 0.0
                weighted_x_init = weighted_y_init = 0.0
                weighted_dx_init = weighted_dy_init = 0.0
                sum_distances = 0.0
                for entry in debug_tag_estimates:
                    pose, dist = entry["pose"], entry["distance"]
                    sum_distances += dist
                    w = 1.0 / (dist**3 + 0.001)
                    weighted_x_init  += pose[0] * w
                    weighted_y_init  += pose[1] * w
                    weighted_dx_init += pose[2] * w
                    weighted_dy_init += pose[3] * w
                    total_weight_init += w
                init_x = weighted_x_init / total_weight_init
                init_y = weighted_y_init / total_weight_init

                # ── Step 2: consensus soft-rejection ──────────────────────
                # Tags whose estimate is far from the initial mean get
                # exponentially down-weighted. This rejects oblique-angle
                # outliers without hard-cutting any tag entirely.
                total_weight = 0.0
                weighted_x = weighted_y = 0.0
                weighted_dx = weighted_dy = 0.0
                for entry in debug_tag_estimates:
                    pose, dist = entry["pose"], entry["distance"]
                    dev = math.hypot(pose[0] - init_x, pose[1] - init_y)
                    consensus_w = math.exp(-dev**2 / (2 * CONSENSUS_SIGMA**2))
                    dist_w = 1.0 / (dist**3 + 0.001)
                    w = dist_w * consensus_w
                    weighted_x  += pose[0] * w
                    weighted_y  += pose[1] * w
                    weighted_dx += pose[2] * w
                    weighted_dy += pose[3] * w
                    total_weight += w

                raw_avg_x  = weighted_x  / total_weight
                raw_avg_y  = weighted_y  / total_weight
                raw_avg_dx = weighted_dx / total_weight
                raw_avg_dy = weighted_dy / total_weight

                mag = math.sqrt(raw_avg_dx**2 + raw_avg_dy**2)
                if mag > 0: raw_avg_dx /= mag; raw_avg_dy /= mag

                history_raw_x.append(raw_avg_x); history_raw_y.append(raw_avg_y)
                if not initialized:
                    last_x, last_y, last_dx, last_dy = raw_avg_x, raw_avg_y, raw_avg_dx, raw_avg_dy
                    initialized = True
                
                movement_metric = math.sqrt(np.std(history_raw_x)**2 + np.std(history_raw_y)**2) if len(history_raw_x) >= 2 else 0.0

                # ── Step 3: dist^2 threshold scaling ──────────────────────
                # Pose noise grows roughly as dist^2 at oblique angles, so
                # scale thresholds by dist^2 (capped) instead of linear dist.
                avg_dist = sum_distances / len(debug_tag_estimates)
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
            cv2.putText(frame, f"State: {state_text} | Pos: ({last_x*100:.1f}cm, {last_y*100:.1f}cm)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            # UDP Telemetry Broadcast
            telemetry_packet = {
                "initialized": initialized, "x_cm": round(last_x * 100, 2), "y_cm": round(last_y * 100, 2),
                "heading_dx": round(last_dx, 3), "heading_dy": round(last_dy, 3), "state": state_text
            }
            try:
                sock.sendto(json.dumps(telemetry_packet).encode('utf-8'), (TARGET_IP, TARGET_PORT))
            except:
                pass

            # --- STATE MACHINE UI & KEYBINDS ---
            key = cv2.waitKey(1) & 0xFF
            
            if APP_STATE == "DRIVING":
                cv2.putText(frame, "STATE: DRIVING. Press 's' to Scan Targets", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                if key == ord('s'):
                    print("Scanning table...")
                    # Save the pristine image so detection boxes don't mess up color masking
                    cv2.imwrite("current_scan.jpg", clean_frame)
                    subprocess.run([sys.executable, "snooker_planner.py", "current_scan.jpg"])
                    
                    if os.path.exists("robot_shooting_plan.json"):
                        with open("robot_shooting_plan.json", "r") as f:
                            shooting_plans = json.load(f)
                        APP_STATE = "AIMING"
                        print(f"Loaded plans for zones: {list(shooting_plans.keys())}")

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

            # 4. Plot Debug Tracking Lines & Individual Camera Heading Vectors
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

            # Show Both Windows
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