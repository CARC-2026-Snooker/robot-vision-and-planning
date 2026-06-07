import cv2
import numpy as np
import os
import sys
import time
import math
from pathlib import Path

# ── Import all settings from central config ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CAM_FRONT,
    HD_SCAN_RESOLUTION,
    HD_CAM_PARAMS,
    HD_CAM_FORWARD_M, HD_CAM_SIDEWAYS_M,
    HD_DETECTION_ZONES,
    REAL_ZONES,
    HD_BALL_RED_LO1, HD_BALL_RED_HI1,
    HD_BALL_RED_LO2, HD_BALL_RED_HI2,
    HD_BALL_BLUE_LO, HD_BALL_BLUE_HI,
    HD_BALL_WHITE_LO, HD_BALL_WHITE_HI,
    HD_BALL_PERCENT_THRESHOLD,
    HD_SCAN_PROJ_OFFSET_X, HD_SCAN_PROJ_OFFSET_Y,
    HD_SCAN_V_CROP_TOP, HD_SCAN_V_CROP_BOT, HD_BALL_MIN_PIXELS
)

def project_zones(robot_x_mm: float, robot_y_mm: float,
                  heading_deg: float,
                  img_w: int, img_h: int) -> dict:
    """
    Compute dynamic Voronoi-style detection zones from robot position.

    Projects each known ball zone (REAL_ZONES) into pixel space using the
    pinhole camera model, then builds max-padding columns: each zone's column
    spans from the midpoint to the previous ball's projected pixel on the left
    to the midpoint to the next ball on the right, and full usable height top
    to bottom. This means every bit of the frame is assigned to exactly one zone.

    Returns dict of zone_name → [x, y, w, h] in pixels, same format as
    HD_DETECTION_ZONES but dynamically computed from robot position.

    Falls back to None if fewer than 2 zones can be projected into frame.
    """
    # ── Camera intrinsics scaled to actual capture resolution ─────────────────
    # HD_CAM_PARAMS = [fx, fy, cx, cy] at 1080p. Scale by actual_h/1080.
    fx_1080, fy_1080, cx_1080, cy_1080 = HD_CAM_PARAMS
    scale = img_h / 1080.0
    fx = fx_1080 * scale
    cx = cx_1080 * scale   # cx/cy scale with the image

    # ── Camera world position ─────────────────────────────────────────────────
    # Camera is HD_CAM_FORWARD_M ahead of robot centre, in the direction the
    # robot is facing (270° = toward tags = toward -Y world).
    h_rad = math.radians(heading_deg)
    cam_x = robot_x_mm + HD_CAM_FORWARD_M * 1000 * math.cos(h_rad) \
                       + HD_CAM_SIDEWAYS_M * 1000 * math.sin(h_rad)
    cam_y = robot_y_mm + HD_CAM_FORWARD_M * 1000 * math.sin(h_rad) \
                       - HD_CAM_SIDEWAYS_M * 1000 * math.cos(h_rad)

    # ── Project each ball zone into pixel space ───────────────────────────────
    # Camera faces heading_deg in world. The ball's lateral offset in the camera
    # frame = component of (ball - camera) perpendicular to heading.
    # Camera right = perpendicular CCW of heading = (-sin θ, cos θ).
    # Camera Z (depth) = component along heading = (cos θ, sin θ).
    cos_h = math.cos(h_rad)
    sin_h = math.sin(h_rad)

    projected = {}
    for name, (bx_mm, by_mm) in REAL_ZONES.items():
        dx = bx_mm - cam_x
        dy = by_mm - cam_y
        # Depth (along camera Z axis = forward)
        depth =  dx * cos_h + dy * sin_h
        if depth <= 50:          # ball behind or too close — skip
            continue
        # Lateral (along camera X axis = robot right = CCW perp)
        lateral = -dx * sin_h + dy * cos_h
        # Pinhole projection to pixel column
        u = cx + fx * lateral / depth + HD_SCAN_PROJ_OFFSET_X
        projected[name] = u

    if len(projected) < 2:
        return None   # can't build meaningful zones

    # ── Build Voronoi columns (midpoints between neighbours) ──────────────────
    sorted_zones = sorted(projected.items(), key=lambda kv: kv[1])

    # Vertical extent: full usable frame height
    y_top = int(img_h * HD_SCAN_V_CROP_TOP) + HD_SCAN_PROJ_OFFSET_Y
    y_bot = int(img_h * (1.0 - HD_SCAN_V_CROP_BOT)) + HD_SCAN_PROJ_OFFSET_Y
    y_top = max(0, min(y_top, img_h - 1))
    y_bot = max(y_top + 1, min(y_bot, img_h))
    zone_h = y_bot - y_top

    zones = {}
    for i, (name, u) in enumerate(sorted_zones):
        # Left boundary: midpoint to previous, or frame left edge
        if i == 0:
            x_left = 0
        else:
            x_left = int((u + sorted_zones[i - 1][1]) / 2)

        # Right boundary: midpoint to next, or frame right edge
        if i == len(sorted_zones) - 1:
            x_right = img_w
        else:
            x_right = int((u + sorted_zones[i + 1][1]) / 2)

        x_left  = max(0, min(x_left,  img_w))
        x_right = max(0, min(x_right, img_w))
        if x_right <= x_left:
            continue

        zones[name] = [x_left, y_top, x_right - x_left, zone_h]

    return zones if zones else None


def detect_balls_in_zones(image_path=None,
                          robot_x_mm: float = None,
                          robot_y_mm: float = None,
                          heading_deg: float = 270.0,
                          tag4_pixel_x: float = None,
                          waitkey: int = 1):
    """
    Detect red/blue balls in the image, one detection zone per REAL_ZONES entry.

    Zone mode (in priority order):
      1. PROJECTED + TAG CORRECTED: robot position known AND tag 4 visible in frame
         AND heading is close to 270°. Projects REAL_ZONES into pixel columns,
         then shifts all zone centres so tag 4's projected pixel matches its
         actual detected pixel. Eliminates tracker noise.
      2. PROJECTED: robot position known but tag 4 not available / heading too oblique.
         Uses raw projection — zones still shift with robot position.
      3. STATIC fallback: no robot position. Uses HD_DETECTION_ZONES from config.py.

    Detection threshold: absolute minimum pixel count (HD_BALL_MIN_PIXELS) rather
    than percentage of zone area. This works better with the large projected zones.
    """
    if image_path: # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from '{image_path}'. Check the file path.")
            return {}
    else: # capture with camera
        # 1. Platform-Specific Camera Initialization
        if sys.platform.startswith('win'):
            cap = cv2.VideoCapture(CAM_FRONT, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(CAM_FRONT, cv2.CAP_V4L2)

        if not cap.isOpened():
            raise IOError("Webcam could not be opened.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  HD_SCAN_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HD_SCAN_RESOLUTION[1])

        # Camera Warm-up Loop
        for _ in range(5):
            cap.read()
            time.sleep(0.1)

        # Capture a single frame
        ret, img = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to capture image from camera.")

        # Save the image sequentially
        folder_path = "camera_images"
        os.makedirs(folder_path, exist_ok=True)
        number = 1
        while os.path.exists(os.path.join(folder_path, f"img_{number}.jpg")):
            number += 1
        save_path = os.path.join(folder_path, f"img_{number}.jpg")
        cv2.imwrite(save_path, img)
        print(f"Photo captured and saved as: {save_path}")
        
    # 2. Build detection zones ─────────────────────────────────────────────────
    img_h, img_w = img.shape[:2]

    # Try projected zones first (when robot position is known)
    zones = None
    mode_desc = "static"
    if robot_x_mm is not None and robot_y_mm is not None:
        # Check if heading is close enough to 270° for the tag correction to be valid.
        # If the robot is at a sharp angle the tag won't be where we expect.
        ang_err = abs(((heading_deg - 270.0) + 180) % 360 - 180)
        heading_ok = ang_err < 15.0   # within 15° of straight-on

        # Compute raw projection
        zones_raw = project_zones(robot_x_mm, robot_y_mm, heading_deg, img_w, img_h)

        if zones_raw:
            if tag4_pixel_x is not None and heading_ok:
                # ── TAG-CORRECTED projection ───────────────────────────────────
                # Project tag 4 (world x=600, y=0) to find where we EXPECT it
                fx_1080, _, cx_1080, _ = HD_CAM_PARAMS
                scale = img_h / 1080.0
                fx = fx_1080 * scale
                cx = cx_1080 * scale
                h_rad = math.radians(heading_deg)
                cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)
                cam_x = robot_x_mm + HD_CAM_FORWARD_M * 1000 * cos_h
                cam_y = robot_y_mm + HD_CAM_FORWARD_M * 1000 * sin_h
                tag4_wx, tag4_wy = 600.0, 0.0  # world pos of tag 4 in mm
                dx = tag4_wx - cam_x;  dy = tag4_wy - cam_y
                depth   =  dx * cos_h + dy * sin_h
                lateral = -dx * sin_h + dy * cos_h
                if depth > 50:
                    expected_tag4_u = cx + fx * lateral / depth
                    pixel_correction = tag4_pixel_x - expected_tag4_u
                    # Shift all zones by the correction
                    zones = {}
                    for name, (zx, zy, zw, zh) in zones_raw.items():
                        new_x = max(0, int(zx + pixel_correction))
                        zones[name] = [new_x, zy, zw, zh]
                    mode_desc = f"projected+tag4 (correction={pixel_correction:+.1f}px)"
                else:
                    zones = zones_raw
                    mode_desc = "projected (tag4 behind camera)"
            else:
                zones = zones_raw
                reason = f"heading {ang_err:.1f}° off" if not heading_ok else "tag4 not found"
                mode_desc = f"projected ({reason} — no tag correction)"

        if zones:
            print(f"[detect] Zones: {mode_desc}")
        else:
            print("[detect] Projection failed — using static zones")

    # Fall back to static hardcoded zones
    if zones is None:
        scale_x = img_w / 1920.0
        scale_y = img_h / 1080.0
        zones = {
            name: [int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)]
            for name, (x, y, w, h) in HD_DETECTION_ZONES.items()
        }
        mode_desc = "static"

    # 3. Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 4. Define Color Thresholds — from config.py
    lower_red1  = np.array(HD_BALL_RED_LO1);   upper_red1  = np.array(HD_BALL_RED_HI1)
    lower_red2  = np.array(HD_BALL_RED_LO2);   upper_red2  = np.array(HD_BALL_RED_HI2)
    lower_blue  = np.array(HD_BALL_BLUE_LO);   upper_blue  = np.array(HD_BALL_BLUE_HI)
    lower_white = np.array(HD_BALL_WHITE_LO);  upper_white = np.array(HD_BALL_WHITE_HI)

    # Absolute pixel count threshold — works better than % with large projected zones.
    # A snooker ball at 2m distance covers roughly 1000-4000px² in 1080p.
    # Scale to actual image size.
    img_scale = (img_w * img_h) / (1920.0 * 1080.0)
    MIN_BALL_PX = int(HD_BALL_MIN_PIXELS * img_scale)

    # Generate global colour masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    global_mask_red   = cv2.bitwise_or(mask_red1, mask_red2)
    global_mask_blue  = cv2.inRange(hsv, lower_blue,  upper_blue)
    global_mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Prepare Debug Layers
    debug_masks_img = np.zeros_like(img)
    debug_masks_img[global_mask_red > 0] = (0, 0, 255)
    debug_masks_img[global_mask_blue > 0] = (255, 0, 0)
    debug_masks_img[global_mask_white > 0] = (255, 255, 255)

    mask_color_layer = np.zeros_like(img)
    mask_color_layer[global_mask_red > 0] = (0, 0, 255)
    mask_color_layer[global_mask_blue > 0] = (255, 0, 0)
    mask_color_layer[global_mask_white > 0] = (255, 255, 255)
    debug_overlay_img = cv2.addWeighted(img, 0.7, mask_color_layer, 0.3, 0)

    output_img = img.copy()

    # --- NEW: Dictionary to store results for the snooker_planner ---
    detected_results = {}

    # 5. Process Each Zone — largest blob threshold (not total pixel count)
    # For each colour, find connected components in the zone ROI and take
    # the largest blob's pixel count. Scattered noise pixels don't form a
    # single large blob, so they can't cross the threshold even if their
    # total count would. Only a real ball — a solid coloured circle — produces
    # a blob big enough to matter.
    def largest_blob(mask_roi):
        """Return (pixel_count, centroid_x, centroid_y) of the largest connected component,
        or (0, None, None) if the mask is empty."""
        if cv2.countNonZero(mask_roi) == 0:
            return 0, None, None
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_roi, connectivity=8)
        # Label 0 is background — skip it
        if n_labels <= 1:
            return 0, None, None
        # stats columns: CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA
        best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        best_px    = int(stats[best_label, cv2.CC_STAT_AREA])
        cx         = int(centroids[best_label, 0])
        cy         = int(centroids[best_label, 1])
        return best_px, cx, cy

    for zone_name, (x, y, w, h) in zones.items():
        # Clamp zone to image bounds
        x2, y2 = min(x + w, img_w), min(y + h, img_h)
        x, y   = max(0, x), max(0, y)
        if x2 <= x or y2 <= y:
            detected_results[zone_name] = "Empty"
            continue

        roi_red   = global_mask_red  [y:y2, x:x2]
        roi_blue  = global_mask_blue [y:y2, x:x2]
        roi_white = global_mask_white[y:y2, x:x2]

        red_px,   red_cx,   red_cy   = largest_blob(roi_red)
        blue_px,  blue_cx,  blue_cy  = largest_blob(roi_blue)
        white_px, white_cx, white_cy = largest_blob(roi_white)

        status        = "Empty"
        color_to_draw = (255, 255, 255)
        ball_center   = None

        # Winner-takes-all on largest blob; must also clear the minimum size
        if red_px >= MIN_BALL_PX and red_px >= blue_px and red_px >= white_px:
            status = f"Red Ball ({red_px}px)"
            color_to_draw = (0, 0, 255)
            if red_cx is not None:
                ball_center = (x + red_cx, y + red_cy)

        elif blue_px >= MIN_BALL_PX and blue_px >= white_px:
            status = f"Blue Ball ({blue_px}px)"
            color_to_draw = (255, 0, 0)
            if blue_cx is not None:
                ball_center = (x + blue_cx, y + blue_cy)

        elif white_px >= MIN_BALL_PX:
            status = f"White Ball ({white_px}px)"
            color_to_draw = (255, 255, 255)
            if white_cx is not None:
                ball_center = (x + white_cx, y + white_cy)

        # Store status in our dictionary map
        detected_results[zone_name] = status

        # Draw debug elements (use clamped bounds)
        for target_img in [output_img, debug_masks_img, debug_overlay_img]:
            cv2.rectangle(target_img, (x, y), (x2, y2), color_to_draw, 2)
            cv2.putText(target_img, f"{zone_name}: {status}", (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_to_draw, 1, cv2.LINE_AA)
            if ball_center is not None:
                cv2.circle(target_img, ball_center, 5, (0, 255, 255), -1)

        print(f"Zone {zone_name} -> Decided: {status}")

    # Save output images
    cv2.imwrite("debug_masks_combined.png", debug_masks_img)
    
    output_dir = Path("debug_images")
    output_dir.mkdir(exist_ok=True)

    i = 1
    while (output_dir / f"debug_overlay_{i}.png").exists():
        i += 1

    cv2.imwrite(
        str(output_dir / f"debug_overlay_{i}.png"),
        debug_overlay_img
    )

    # Note: When running autonomously on the actual robot, you can comment these out
    # so the robot code doesn't freeze waiting for you to press a key.
    # camera_name = "1. Snooker Bot Vision - Results"
    # combined_name = "2. Debug - Combined Pure Masks"
    # debug_name = "3. Debug - Translucent Overlay"
    # cv2.namedWindow(camera_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(combined_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(debug_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(camera_name, 800, 450)
    # cv2.resizeWindow(combined_name, 800, 450)
    # cv2.resizeWindow(debug_name, 800, 450)
    # cv2.imshow(camera_name, output_img)
    # cv2.imshow(combined_name, debug_masks_img)
    # cv2.imshow(debug_name, debug_overlay_img)
    # cv2.waitKey(waitkey)

    return detected_results  # <--- THIS MAKES INTEGRATION WORK!

if __name__ == "__main__":
    # Test execution manually using an image file
    detect_balls_in_zones(r"camera_images\img_21.jpg", 0)