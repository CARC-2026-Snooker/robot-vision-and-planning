import cv2
import numpy as np
import os
import sys
import time

def detect_balls_in_zones(image_path = None, waitkey = 1):
    if image_path: # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from '{image_path}'. Check the file path.")
            return {}
    else: # capture with camera
        # 1. Platform-Specific Camera Initialization
        if sys.platform.startswith('win'):
            cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(2, cv2.CAP_V4L2) # Best for Raspberry Pi / Linux

        if not cap.isOpened():
            raise IOError("Webcam could not be opened.")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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
        
    # 2. Define Regions of Interest (ROIs)
    zones = {
        "B": (360, 560, 170, 170),
        "C": (610, 500, 130, 130),
        "D": (790, 500, 130, 130),
        "E": (960, 520, 130, 130),
        "F": (1130, 500, 130, 130),
        "G": (1320, 500, 130, 130),
        "A": (1550, 560, 170, 170),
    }

    for zone, region in zones.items(): # change as set / dont change
        zones[zone] = [int(x * (1080.0/1080.0)) for x in region]

    # 3. Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 4. Define Color Thresholds
    lower_red1 = np.array([0, 50, 20])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 20])
    upper_red2 = np.array([180, 255, 255])
    
    lower_blue = np.array([95, 50, 30])
    upper_blue = np.array([150, 255, 255])

    lower_white = np.array([180, 75, 255]) 
    upper_white = np.array([180, 75, 255])

    PERCENT_THRESHOLD = 0.05

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
    cv2.imwrite("debug_overlay.png", debug_overlay_img)

    # Note: When running autonomously on the actual robot, you can comment these out
    # so the robot code doesn't freeze waiting for you to press a key.
    camera_name = "1. Snooker Bot Vision - Results"
    combined_name = "2. Debug - Combined Pure Masks"
    debug_name = "3. Debug - Translucent Overlay"
    cv2.namedWindow(camera_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(combined_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(debug_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(camera_name, 800, 450)
    cv2.resizeWindow(combined_name, 800, 450)
    cv2.resizeWindow(debug_name, 800, 450)
    cv2.imshow(camera_name, output_img)
    cv2.imshow(combined_name, debug_masks_img)
    cv2.imshow(debug_name, debug_overlay_img)
    cv2.waitKey(waitkey)

    return detected_results  # <--- THIS MAKES INTEGRATION WORK!

if __name__ == "__main__":
    # Test execution manually using an image file
    detect_balls_in_zones(r"camera_images\img_21.jpg", 0)