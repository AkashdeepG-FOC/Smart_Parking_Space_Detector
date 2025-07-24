🔥 Smart Parking Space Detector using YOLOv8 & OpenCV
🚗 An advanced real-time smart parking system that uses YOLOv8 (Ultralytics) and OpenCV to detect occupied and vacant parking slots in video feeds using polygon-based ROI mapping. Ideal for smart cities, mall parking, and surveillance applications.

📌 Features
✅ Real-time parking slot detection from video

🎯 Uses YOLOv8 object detection to identify vehicles

📐 Slot occupancy checked via custom polygon ROIs

🔁 Frame buffer smoothing to reduce false detections

📊 Live count of free and occupied slots

🧠 Built for CCTV/IoT camera feed analysis

⚡ Lightweight and efficient for edge deployment

📁 Project Structure
bash
Copy
Edit
Smart-Parking-Detector-YOLOv8/
│
├── input_video/               # Test video input
│   └── parking_space.mp4
│
├── yolov8n.pt                 # YOLOv8 pretrained model
│
├── Space_ROIs                 # Pickle file of slot polygons
│
├── utilis.py                  # Helper functions (detection logic)
│
├── main.py                    # Main detection code
│
└── README.md                  # Project documentation
📹 How It Works
Parking ROIs are pre-defined as polygon regions (loaded from Space_ROIs).

YOLOv8 detects vehicles in each frame.

Detected vehicle center points are checked against all ROI polygons.

A deque buffer smooths detections over 5 frames to avoid flicker.

Final slot status is shown with red (occupied) or green (free) outlines.

Counts are displayed overlayed on the video.

🚀 Requirements
Python 3.8+

OpenCV (cv2)

PyTorch

Ultralytics YOLOv8

Install dependencies:

bash
Copy
Edit
pip install ultralytics opencv-python
🛠️ Run the Code
bash
Copy
Edit
python main.py
📷 Demo Screenshot
(You can add a screenshot or gif here in the GitHub repo to show output)

🤖 Future Improvements
Real-time video streaming from IP camera

Web dashboard integration using Flask / FastAPI

MQTT / IoT alerts for parking systems

ROI polygon editor for easy setup

