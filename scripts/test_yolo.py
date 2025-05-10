from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt") 

video_path = "../data/test_yolo.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Cannot open video file:", video_path)
    exit()

out_path = "../demo/yolo_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
if not out.isOpened():
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

    # Check detection results
    if len(results[0].boxes) == 0:
        print("No objects detected")
    else:
        print(f"Detected {len(results[0].boxes)} objects")

    # Generate annotated frame
    annotated_frame = results[0].plot()
    if annotated_frame is None:
        print("Annotated frame is None, please check model output")
    else:
        print("Successfully generated annotated frame")

    # Write to output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
print("Detection completed, results saved to:", out_path)
