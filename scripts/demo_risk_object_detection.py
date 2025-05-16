import os
import cv2
import torch
import argparse
import numpy as np
import json
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Import the ROM classifier and utility functions
from train_rom_classifier import ROMClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Demo of Risk Object Mining System")
    parser.add_argument('--yolo_model', type=str, required=True,
                        help='Path to trained YOLOv8 model weights')
    parser.add_argument('--rom_model', type=str, required=True,
                        help='Path to trained ROM model weights')
    parser.add_argument('--rom_config', type=str, required=True,
                        help='Path to ROM model configuration')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input video from RiskBench')
    parser.add_argument('--output_video', type=str, default='../demo/risk_detection_output.mp4',
                        help='Path to save output video')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for object detection')
    parser.add_argument('--risk_threshold', type=float, default=0.6,
                        help='Threshold for classifying an object as risky')
    return parser.parse_args()

def extract_sequence_features(track_history, track_class, track_conf, sequence_length):
    """Extract features for ROM classifier from tracking history."""
    if len(track_history) < sequence_length:
        return None
    
    # Use the most recent frames
    recent_history = list(track_history)[-sequence_length:]
    
    # Extract features for each frame in the sequence
    sequence_features = []
    for i in range(len(recent_history) - 1):  # -1 because we need next frame for velocity
        frame_data = recent_history[i]
        next_frame_data = recent_history[i+1]
        
        # Get bounding box
        x1, y1, x2, y2 = frame_data
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate velocity (change in position)
        next_x1, next_y1, next_x2, next_y2 = next_frame_data
        next_center_x = (next_x1 + next_x2) / 2
        next_center_y = (next_y1 + next_y2) / 2
        
        velocity_x = next_center_x - center_x
        velocity_y = next_center_y - center_y
        
        # Calculate area change
        area = width * height
        next_area = (next_x2 - next_x1) * (next_y2 - next_y1)
        area_change = next_area - area
        
        # Compile features for this frame
        frame_features = [
            center_x, center_y,  # Position
            width, height,       # Size
            velocity_x, velocity_y,  # Velocity
            area_change,         # Area change
            track_class,         # Object class
            track_conf           # Detection confidence
        ]
        
        sequence_features.append(frame_features)
    
    return np.array([sequence_features])

def main():
    args = parse_args()
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model from {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    
    # Load ROM classifier model
    print(f"Loading ROM model from {args.rom_model}")
    # Load model configuration
    with open(args.rom_config, 'r') as f:
        model_config = json.load(f)
    
    # Initialize ROM classifier
    rom_model = ROMClassifier(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 2)
    )
    rom_model.load_state_dict(torch.load(args.rom_model))
    rom_model.eval()  # Set to evaluation mode
    
    # Initialize DeepSORT tracker
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_cosine_distance=0.4,
        nn_budget=100,
        embedder="mobilenet",
        half=True,
        bgr=True,
    )
    
    # Open video file
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup output video writer
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    
    # Track history for each object (to build sequence features)
    # Use dict with track_id as key and deque of bbox history as value
    sequence_length = model_config.get('sequence_length', 30)
    track_histories = {}
    track_class_ids = {}
    track_confidences = {}
    track_risk_scores = {}
    
    frame_id = 0
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 detection
        results = yolo_model(frame, conf=args.conf_threshold)[0]
        
        # Convert YOLO detections to DeepSORT format
        detections = []
        for detection in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = detection
            class_id = int(class_id)
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # DeepSORT expects detections in format: (ltwh, confidence, class_id)
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
            
            # Convert ltwh to xyxy format
            x1, y1, w, h = map(int, ltwh)
            x2, y2 = x1 + w, y1 + h
            
            # Update track history
            if track_id not in track_histories:
                track_histories[track_id] = deque(maxlen=sequence_length*2)  # Double length for buffer
                track_class_ids[track_id] = class_id
                track_confidences[track_id] = conf
            
            # Add current bbox to history
            track_histories[track_id].append((x1, y1, x2, y2))
            
            # If we have enough history, evaluate ROM risk
            if len(track_histories[track_id]) >= sequence_length:
                # Extract features for ROM classifier
                sequence_features = extract_sequence_features(
                    track_histories[track_id], 
                    track_class_ids[track_id],
                    track_confidences[track_id],
                    sequence_length
                )
                
                if sequence_features is not None:
                    # Convert to tensor and get risk prediction
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(sequence_features)
                        risk_score = rom_model(features_tensor).item()
                        
                        # Store risk score
                        track_risk_scores[track_id] = risk_score
            
            # Determine if object is risky
            is_risk = track_id in track_risk_scores and track_risk_scores[track_id] > args.risk_threshold
            
            # Determine color based on risk (red for risky, green for safe)
            color = (0, 0, 255) if is_risk else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Get class name (update with your actual class mapping)
            class_names = ["vehicle", "pedestrian", "cyclist"]  # Update as needed
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            # Draw text with track ID, class, and risk score if available
            risk_text = f" Risk:{track_risk_scores[track_id]:.2f}" if track_id in track_risk_scores else ""
            text = f"ID:{track_id} {class_name}{risk_text}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # For high-risk objects, add extra emphasis
            if is_risk:
                # Draw thicker border
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 3)
                
                # Add "HIGH RISK" label
                cv2.putText(frame, "HIGH RISK", (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Update progress
        if frame_id % 100 == 0:
            print(f"Processed frame {frame_id}")
        
        frame_id += 1
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Risk detection completed. Output saved to {args.output_video}")

if __name__ == "__main__":
    main()
