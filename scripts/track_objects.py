import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import csv
import json
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def parse_args():
    parser = argparse.ArgumentParser(description="Track objects using YOLOv8 and DeepSORT")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the YOLOv8 model weights (.pt file)')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input video file or directory of videos')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for object detection')
    parser.add_argument('--output_dir', type=str, default='../data/tracking_results',
                        help='Directory to save tracking results')
    parser.add_argument('--save_video', action='store_true',
                        help='Save output video with tracking visualizations')
    parser.add_argument('--max_cosine_distance', type=float, default=0.4,
                        help='Maximum cosine distance for DeepSORT')
    return parser.parse_args()

def process_video(model, video_path, tracker, output_dir, conf_threshold, save_video):
    """Process a single video file with YOLOv8 and DeepSORT."""
    # Create a directory for this video's results
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video writer if requested
    out = None
    if save_video:
        output_video_path = os.path.join(video_output_dir, f"{video_name}_tracked.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Saving output video to {output_video_path}")
    
    # Setup CSV writer for tracking results
    csv_path = os.path.join(video_output_dir, "tracking_results.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'track_id', 'class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'])
    
    # Dictionary to store tracking data for later analysis
    tracking_data = defaultdict(list)
    
    # Process each frame
    frame_id = 0
    print(f"Processing video: {video_path} with {total_frames} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 detection
        results = model(frame, conf=conf_threshold)[0]
        
        # Convert YOLO detections to a format DeepSORT can use
        detections = []
        for detection in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = detection
            class_id = int(class_id)
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # DeepSORT expects detections in format: (ltwh, confidence, class_id, original data)
            ltwh = [x1, y1, x2 - x1, y2 - y1]
            detections.append((ltwh, conf, class_id, detection))
        
        # Update tracker with new detections
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltwh = track.to_ltwh()
            class_id = track.get_det_class()
            conf = track.get_det_conf()
            
            # Convert ltwh to xyxy format for visualization and saving
            x1, y1, w, h = map(int, ltwh)
            x2, y2 = x1 + w, y1 + h
            
            # Draw bounding box and ID for visualization
            if save_video:
                color = (0, 255, 0)  # Green for tracks
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Get class name (replace with your actual class mapping)
                class_names = ["vehicle", "pedestrian", "cyclist"]  # Update as needed
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                # Draw text with track ID and class
                text = f"ID:{track_id} {class_name} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save tracking results to CSV
            csv_writer.writerow([frame_id, track_id, class_id, conf, x1, y1, x2, y2])
            
            # Save to tracking data dictionary for later analysis
            tracking_data[track_id].append({
                'frame': frame_id,
                'class': class_id,
                'confidence': float(conf),
                'bbox': [x1, y1, x2, y2]
            })
        
        # Write the frame to output video if requested
        if save_video and out is not None:
            out.write(frame)
        
        # Update progress
        if frame_id % 100 == 0:
            print(f"Processed {frame_id}/{total_frames} frames")
        
        frame_id += 1
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    csv_file.close()
    
    # Save tracking data as JSON for later analysis
    json_path = os.path.join(video_output_dir, "tracking_data.json")
    with open(json_path, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"Tracking completed for {video_path}")
    print(f"Tracking results saved to {video_output_dir}")
    
    return tracking_data

def process_directory(model, directory_path, tracker, output_dir, conf_threshold, save_video):
    """Process all video files in a directory."""
    video_extensions = ['.mp4', '.avi', '.mov']
    videos = []
    
    # Find all video files in the directory
    for ext in video_extensions:
        videos.extend(list(Path(directory_path).glob(f"**/*{ext}")))
    
    print(f"Found {len(videos)} videos in {directory_path}")
    
    # Process each video
    all_tracking_data = {}
    for video_path in videos:
        tracking_data = process_video(
            model, str(video_path), tracker, output_dir, conf_threshold, save_video
        )
        if tracking_data:
            all_tracking_data[str(video_path)] = tracking_data
    
    # Save combined tracking data
    combined_json_path = os.path.join(output_dir, "all_tracking_data.json")
    with open(combined_json_path, 'w') as f:
        json.dump(all_tracking_data, f, indent=2)
    
    print(f"All tracking results saved to {output_dir}")
    return all_tracking_data

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model from {args.model_path}")
    model = YOLO(args.model_path)
    
    # Initialize DeepSORT tracker
    tracker = DeepSort(
        max_age=30,  # Maximum number of frames a track remains active after being lost
        n_init=3,    # Number of frames needed to confirm a track
        max_cosine_distance=args.max_cosine_distance,
        nn_budget=100,  # Maximum size of the appearance descriptors gallery
        override_track_class=None,
        embedder="mobilenet",  # Feature extractor
        half=True,  # Use half precision for faster processing
        bgr=True,   # BGR format for input images
    )
    
    # Process video(s)
    if os.path.isfile(args.video_path):
        # Process a single video file
        process_video(
            model, args.video_path, tracker, args.output_dir, 
            args.conf_threshold, args.save_video
        )
    elif os.path.isdir(args.video_path):
        # Process all videos in a directory
        process_directory(
            model, args.video_path, tracker, args.output_dir, 
            args.conf_threshold, args.save_video
        )
    else:
        print(f"Error: {args.video_path} is not a valid file or directory")

if __name__ == "__main__":
    main()