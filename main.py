import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path

# ── Import all settings from central config ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CAM_FRONT,       # HD camera used for scanning (same device as tracking)
    HD_SCAN_RESOLUTION,
    HD_DETECTION_ZONES,
    HD_BALL_RED_LO1, HD_BALL_RED_HI1,
    HD_BALL_RED_LO2, HD_BALL_RED_HI2,
    HD_BALL_BLUE_LO, HD_BALL_BLUE_HI,
    HD_BALL_WHITE_LO, HD_BALL_WHITE_HI,
    HD_BALL_PERCENT_THRESHOLD,
)

def detect_balls_in_zones(image_path = None, waitkey = 1):
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
        
    # 2. Define Regions of Interest (ROIs) — from config.py
    # Zones defined at 1080p in config; scale to actual image resolution
    img_h, img_w = img.shape[:2]
    scale_x = img_w / 1920.0
    scale_y = img_h / 1080.0
    zones = {
        name: [int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)]
        for name, (x, y, w, h) in HD_DETECTION_ZONES.items()
    }

    # 3. Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 4. Define Color Thresholds — from config.py
    lower_red1 = np.array(HD_BALL_RED_LO1)
    upper_red1 = np.array(HD_BALL_RED_HI1)
    lower_red2 = np.array(HD_BALL_RED_LO2)
    upper_red2 = np.array(HD_BALL_RED_HI2)
    lower_blue  = np.array(HD_BALL_BLUE_LO)
    upper_blue  = np.array(HD_BALL_BLUE_HI)
    lower_white = np.array(HD_BALL_WHITE_LO)
    upper_white = np.array(HD_BALL_WHITE_HI)
    PERCENT_THRESHOLD = HD_BALL_PERCENT_THRESHOLD

    # Generate global masks for debugging
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    global_mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    global_mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
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

    # 5. Process Each Zone
    for zone_name, (x, y, w, h) in zones.items():
        roi_red = global_mask_red[y:y+h, x:x+w]
        roi_blue = global_mask_blue[y:y+h, x:x+w]
        roi_white = global_mask_white[y:y+h, x:x+w]

        red_pixel_count = cv2.countNonZero(roi_red)
        blue_pixel_count = cv2.countNonZero(roi_blue)
        white_pixel_count = cv2.countNonZero(roi_white)

        zone_area = w * h
        red_percent = red_pixel_count / zone_area
        blue_percent = blue_pixel_count / zone_area
        white_percent = white_pixel_count / zone_area

        status = "Empty"
        color_to_draw = (255, 255, 255) 
        ball_center = None 

        if red_percent > PERCENT_THRESHOLD and red_percent > blue_percent and red_percent > white_percent:
            status = f"Red Ball ({red_percent:.1%})"
            color_to_draw = (0, 0, 255) 
            y_indices, x_indices = np.where(roi_red > 0)
            if len(x_indices) > 0:
                ball_center = (x + int(np.mean(x_indices)), y + int(np.mean(y_indices)))
            
        elif blue_percent > PERCENT_THRESHOLD and blue_percent > white_percent:
            status = f"Blue Ball ({blue_percent:.1%})"
            color_to_draw = (255, 0, 0) 
            y_indices, x_indices = np.where(roi_blue > 0)
            if len(x_indices) > 0:
                ball_center = (x + int(np.mean(x_indices)), y + int(np.mean(y_indices)))

        elif white_percent > PERCENT_THRESHOLD:
            status = f"White Ball ({white_percent:.1%})"
            color_to_draw = (255, 255, 255) 
            y_indices, x_indices = np.where(roi_white > 0)
            if len(x_indices) > 0:
                ball_center = (x + int(np.mean(x_indices)), y + int(np.mean(y_indices)))

        # Store status in our dictionary map
        detected_results[zone_name] = status

        # 7. Draw debug elements
        for target_img in [output_img, debug_masks_img, debug_overlay_img]:
            cv2.rectangle(target_img, (x, y), (x + w, y + h), color_to_draw, 2)
            cv2.putText(target_img, f"{zone_name}: {status}", (x, y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_to_draw, 1, cv2.LINE_AA)
            if ball_center is not None:
                cv2.circle(target_img, ball_center, 2, (0, 255, 255), -1) 
        
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