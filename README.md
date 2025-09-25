# Object-detection-using-web-camera

# 📷 Real-Time Object Detection Using YOLOv4 and Webcam

## 🧠 Overview

This project demonstrates *real-time object detection* using the *YOLOv4* model on a webcam feed. It uses *OpenCV* to access your webcam and applies a deep learning model to detect and label objects live in each frame.

---

## 🚀 Features

- 🎯 Real-time object detection with YOLOv4
- 📦 Draws bounding boxes with labels and confidence scores
- 🔍 Detects multiple objects per frame
- 🖥 Uses your webcam for live video input
- 🛠 Easily customizable to use custom-trained models

---

## 📁 Project Structure

object-detection-webcam/
│

├── detect.py # Main detection script

├── yolov4.weights # Pre-trained YOLOv4 weights

├── yolov4.cfg # YOLOv4 configuration file

├── coco.names # COCO class labels (80 classes)

├── requirements.txt # Python dependencies

└── README.md # This README file


---

## 🧰 Requirements

- Python 3.x
- OpenCV
- NumPy

### 📦 Install Dependencies

Using pip:

bash
pip install -r requirements.txt

# Download YOLOv4 Files

Download the following files and place them in the root folder:

yolov4.weights

yolov4.cfg

coco.names

How to Run

Make sure you have the necessary files, then run:

python detect.py


Press q to quit the program.

Detected objects will be labeled in the webcam feed.

🧪 How It Works

Capture frames from webcam with OpenCV.

Preprocess each frame to a blob suitable for YOLOv4.

Run inference using OpenCV's DNN module.

Post-process outputs:

Filter by confidence threshold (0.5).

Apply Non-Max Suppression to remove duplicates.

Display bounding boxes and class labels in real-time.

# code:
python

import cv2
import numpy as np

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up video capture for webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Prepare the image for YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get YOLO output
    outputs = net.forward(output_layers)
    
    # Initialize lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the image with detected objects
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
 


📸 Example Output

Here's what the output looks like :

<img width="791" height="592" alt="Screenshot 2025-09-20 145356 - Copy" src="https://github.com/user-attachments/assets/cfb4e33a-89b4-49a2-809a-f74bc3dd9e71" />




🧾 Sample requirements.txt
opencv-python
numpy

# 🛠 Troubleshooting

❌ Webcam not detected: Make sure it's not in use by another app.

⚠ No objects detected: Ensure yolov4.weights, cfg, and coco.names are in the correct location.

🐢 Too slow?: Lower the video resolution or switch to a smaller YOLO model (e.g., YOLOv4-tiny).

# 📄 License

This project is licensed under the MIT License. See the LICENSE
 file for details.

# 🙌 Acknowledgments

YOLOv4 by Alexey Bochkovskiy

COCO Dataset

OpenCV

### ✅ What to Do Next:
1. Save this content as README.md in your project folder.
