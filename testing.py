import cv2

for cam in ["/dev/camera_4k", "/dev/camera_front", "/dev/camera_back"]:
    cap = cv2.VideoCapture(cam)
    print(cam, cap.isOpened())