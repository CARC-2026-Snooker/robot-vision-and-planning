import cv2
import numpy as np
import sys
import time


# ==========================================================
# POSE EXTRACTION
# ==========================================================

def board_pose(corners):
    pts = corners.reshape(-1, 2)

    center = np.mean(pts, axis=0)

    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)

    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]

    size = np.sqrt(width * height)

    # Estimate rotation using top-left -> top-right
    # For a 9x6 inner-corner board, top row has 9 points
    top_left = pts[0]
    top_right = pts[8]

    dx = top_right[0] - top_left[0]
    dy = top_right[1] - top_left[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Estimate perspective tilt using left and right side heights
    left_height = np.linalg.norm(pts[0] - pts[-9])
    right_height = np.linalg.norm(pts[8] - pts[-1])

    if max(left_height, right_height) > 0:
        tilt_ratio = min(left_height, right_height) / max(left_height, right_height)
    else:
        tilt_ratio = 1.0

    return (
        center[0],
        center[1],
        size,
        angle,
        tilt_ratio
    )


# ==========================================================
# SIMILARITY
# ==========================================================

def pose_similarity(new_pose, old_pose, image_width, image_height):
    cx1, cy1, size1, angle1, tilt1 = new_pose
    cx2, cy2, size2, angle2, tilt2 = old_pose

    image_diag = np.sqrt(image_width ** 2 + image_height ** 2)

    # Position similarity
    center_dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    center_similarity = max(0.0, 1.0 - center_dist / image_diag)

    # Size similarity
    size_similarity = 1.0 - min(abs(size1 - size2) / max(size1, size2), 1.0)

    # Rotation similarity
    angle_diff = abs(angle1 - angle2)
    angle_diff = min(angle_diff, 360 - angle_diff)
    rotation_similarity = max(0.0, 1.0 - angle_diff / 90.0)

    # Tilt similarity
    tilt_similarity = 1.0 - min(abs(tilt1 - tilt2), 1.0)

    # Combined score
    similarity = (
        0.35 * center_similarity +
        0.25 * size_similarity +
        0.20 * rotation_similarity +
        0.20 * tilt_similarity
    )

    return similarity


# ==========================================================
# MAIN
# ==========================================================

def main():
    CHECKERBOARD_SIZE = (9, 6)
    SQUARE_SIZE_METERS = 0.028

    TARGET_FRAMES = 100
    SIMILARITY_THRESHOLD = 0.90
    MIN_CAPTURE_GAP = 0.5

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    objp = np.zeros(
        (CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3),
        np.float32
    )
    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD_SIZE[0],
        0:CHECKERBOARD_SIZE[1]
    ].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_METERS

    objpoints = []
    imgpoints = []
    accepted_poses = []

    print("\n================================")
    print("AUTO CAMERA CALIBRATION")
    print("================================")
    print(f"Target Frames: {TARGET_FRAMES}")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"Min Capture Gap: {MIN_CAPTURE_GAP}s")
    print("")
    print("Move board around:")
    print("- center")
    print("- corners")
    print("- near/far")
    print("- rotate")
    print("- tilt")
    print("")
    print("Press Q to quit.")
    print("================================\n")

    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture("/dev/camera_front", cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture("/dev/camera_front", cv2.CAP_V4L2)

    if not cap.isOpened():
        raise IOError("Camera could not be opened.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    captured_frames = 0
    last_capture_time = 0

    similarity_text = "---"
    status_text = "Waiting"
    gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            None
        )

        current_time = time.time()

        if found:
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            cv2.drawChessboardCorners(
                frame,
                CHECKERBOARD_SIZE,
                corners2,
                found
            )

            pose = board_pose(corners2)

            best_similarity = 0.0
            for old_pose in accepted_poses:
                sim = pose_similarity(
                    pose,
                    old_pose,
                    frame.shape[1],
                    frame.shape[0]
                )
                best_similarity = max(best_similarity, sim)

            similarity_text = f"{best_similarity:.2f}"

            enough_time = (
                current_time - last_capture_time >= MIN_CAPTURE_GAP
            )

            if enough_time and best_similarity < SIMILARITY_THRESHOLD:
                objpoints.append(objp.copy())
                imgpoints.append(corners2)
                accepted_poses.append(pose)

                captured_frames += 1
                last_capture_time = current_time
                status_text = "CAPTURED"

                print(
                    f"[CAPTURED] {captured_frames}/{TARGET_FRAMES} "
                    f"(similarity={best_similarity:.2f})"
                )
            else:
                status_text = "Too Similar"

        else:
            similarity_text = "---"
            status_text = "No Board"

        cv2.putText(
            frame,
            f"Captured: {captured_frames}/{TARGET_FRAMES}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Similarity: {similarity_text}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Status: {status_text}",
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if captured_frames >= TARGET_FRAMES:
            print("\nTarget reached.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frames < 5:
        print("Not enough captures for calibration.")
        return

    print("\nCalculating calibration...")

    img_size = (gray.shape[1], gray.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_size,
        None,
        None
    )

    if not ret:
        print("\nCalibration failed.")
        return

    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    print("\n================================")
    print("CALIBRATION SUCCESSFUL")
    print("================================")

    print("\nCamera Matrix:")
    print(mtx)

    print("\nDistortion Coefficients:")
    print(dist)

    print("\n================================")
    print("COPY INTO YOUR CONFIG FILE")
    print("================================")
    print(f"camera_params = [{fx:.8f}, {fy:.8f}, {cx:.8f}, {cy:.8f}]")

    dist_flat = dist.flatten()
    dist_str = ", ".join(f"{x:.8e}" for x in dist_flat)
    print(f"CAM_4K_DIST_COEFFS = np.array([[{dist_str}]])")
    print("================================")

    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            mtx,
            dist
        )

        error = cv2.norm(
            imgpoints[i],
            imgpoints2,
            cv2.NORM_L2
        ) / len(imgpoints2)

        mean_error += error

    mean_error /= len(objpoints)

    print(f"\nReprojection Error: {mean_error:.4f}")

    if mean_error < 0.5:
        print("Excellent calibration.")
    elif mean_error < 1.0:
        print("Good calibration.")
    else:
        print("Calibration is usable, but more board positions may help.")


if __name__ == "__main__":
    main()