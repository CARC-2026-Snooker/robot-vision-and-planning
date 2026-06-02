import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)

    if not cap.isOpened():
        print(f"{i}: FAILED")
        continue

    ret, frame = cap.read()

    if ret:
        print(f"{i}: OK {frame.shape}")
    else:
        print(f"{i}: OPENED BUT NO FRAME")

    cap.release()