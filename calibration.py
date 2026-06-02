import cv2
import numpy as np
import os
import sys
import time

def main():
    # --- 1. CONFIGURATION ---
    # Number of INNER corners on your checkerboard (width, height)
    # A standard OpenCV printable board is often (9, 6)
    CHECKERBOARD_SIZE = (9, 6)
    
    # The physical size of a single square on your printed board in meters
    # e.g., 25mm = 0.025 meters
    SQUARE_SIZE_METERS = 0.020

    # Criteria for refining the corner detection (sub-pixel accuracy)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare the 3D points of the real-world checkerboard
    # This creates an array of coordinates: (0,0,0), (1,0,0), (2,0,0) ...
    # scaled by the actual square size.
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_METERS

    # Arrays to store 3D points (real world) and 2D points (image pixels)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    print("Starting camera feed...")
    print("Press 'c' to capture a frame when the checkerboard is detected.")
    print("Press 'q' to finish capturing and calculate the calibration.")
    
    if sys.platform.startswith('win'):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(1, cv2.CAP_V4L2) # Best for Raspberry Pi / Linux

    if not cap.isOpened():
        raise IOError("Webcam could not be opened.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    captured_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.flip(frame, -1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

        # If found, draw them on the frame
        if ret_corners:
            # Refine the corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Draw the corners
            cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners2, ret_corners)
            
            # Show how many frames we've captured
            cv2.putText(frame, f"Captured: {captured_frames} (Need ~15-20)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Checkerboard NOT detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.namedWindow('Camera Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Calibration', 800, 450)
        cv2.imshow('Camera Calibration', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and ret_corners:
            # Save the points for calibration
            objpoints.append(objp)
            imgpoints.append(corners2)
            captured_frames += 1
            print(f"Frame captured! Total: {captured_frames}")
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- 2. CALCULATE CALIBRATION ---
    if captured_frames < 5:
        print("\nNot enough frames captured for reliable calibration. Try to get at least 15.")
        return

    print("\nCalculating calibration parameters... this might take a second.")
    
    # Get the image size from the last frame
    img_size = (gray.shape[1], gray.shape[0])

    # Run the calibration math
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    if ret:
        print("\n--- CALIBRATION SUCCESSFUL ---")
        
        # Extract focal lengths (fx, fy) and optical centers (cx, cy)
        fx = mtx[0, 0]
        fy = mtx[1, 1]
        cx = mtx[0, 2]
        cy = mtx[1, 2]

        print(f"\nYour Camera Matrix:\n{mtx}")
        print(f"\nYour Distortion Coefficients:\n{dist}")
        
        print("\n=======================================================")
        print("COPY AND PASTE THIS INTO YOUR APRILTAG SCRIPT:")
        print(f"camera_params = [{fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f}]")
        print("=======================================================")
        
        # Calculate Reprojection Error (Lower is better, ideally under 1.0)
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
            
        print(f"\nTotal re-projection error: {mean_error/len(objpoints):.4f}")
        print("(If the error is higher than 1.0, you might want to recalibrate with a flatter board or better lighting).")
        
    else:
        print("\nCalibration failed. Please try again with clearer lighting and a flat board.")

if __name__ == '__main__':
    main()