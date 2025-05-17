from ultralytics import YOLO
import cv2
import os
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize model and tracker
model = YOLO("scripts/yolov8s.pt")
tracker = DeepSort(max_age=30)

# Video input
video_path = "data/test_yolo.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Cannot open video file:", video_path)
    exit()

# Output directories
os.makedirs("demo", exist_ok=True)
os.makedirs("features", exist_ok=True)

# Video output
out_path = "demo/test_yolo_deepsort_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
if not out.isOpened():
    print("Cannot open output file:", out_path)
    exit()

# Tracking results
tracking_data = []
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    results = model(frame)
    boxes = results[0].boxes
    detections = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Object tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cls_name = model.names[track.det_class]
        conf = track.det_conf

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save tracking info
        tracking_data.append({
            "frame": frame_id,
            "track_id": track_id,
            "class": cls_name,
            "conf": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    out.write(frame)
    frame_id += 1

cap.release()
out.release()

# Save CSV
df = pd.DataFrame(tracking_data)
csv_path = "features/tracking_output.csv"
df.to_csv(csv_path, index=False)

print("Tracking completed.")
print("Video saved to:", out_path)
print("CSV saved to:", csv_path)
