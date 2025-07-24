import cv2
import torch
import numpy as np
import pickle
import os
from collections import deque
from ultralytics import YOLO
from utilis import YOLO_Detection, label_detection
from datetime import datetime

# Load YOLO model
model_path = r"D:\effinet\Smart-Parking-Space-Detector-using-YOLO-and-OpenCV-main\yolov8n.pt"
model = YOLO(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load parking slot ROIs
with open(r'D:\effinet\Smart-Parking-Space-Detector-using-YOLO-and-OpenCV-main\Space_ROIs', 'rb') as f:
    posList = pickle.load(f)

# Load video file
video_path = r"D:\effinet\Smart-Parking-Space-Detector-using-YOLO-and-OpenCV-main\input_video\input_video\parking_space.mp4"
if not os.path.exists(video_path):
    print(f"❌ Video not found at {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Memory buffer for detection smoothing
slot_buffers = [deque(maxlen=5) for _ in posList]  # 5-frame memory buffer

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Detection
    boxes, classes, names = YOLO_Detection(model, frame)

    # Get center points of detections
    detection_points = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        detection_points.append((cx, cy))
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

    occupied_count = 0

    # Check each slot for occupancy
    for idx, slot in enumerate(posList):
        occupied = any(cv2.pointPolygonTest(np.array(slot, np.int32), pt, False) >= 0 for pt in detection_points)
        slot_buffers[idx].append(1 if occupied else 0)

        # Apply smoothing
        final_status = sum(slot_buffers[idx]) > (len(slot_buffers[idx]) // 2)

        if final_status:
            color = (0, 0, 255)  # Red = Occupied
            occupied_count += 1
        else:
            color = (0, 255, 0)  # Green = Free

        # Draw slot polygon
        pts = np.array(slot, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, color, 2)

    # Display count info
    free_count = len(posList) - occupied_count
    cv2.rectangle(frame, (20, 10), (300, 60), (255, 255, 255), -1)
    cv2.putText(frame, f"Free: {free_count} | Occupied: {occupied_count}",
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Show frame
    cv2.imshow("Smart Parking - Advanced", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
