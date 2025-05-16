
import os
import json
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network for Risk Object Mining (ROM)")
    parser.add_argument('--tracking_dir', type=str, required=True,
                        help='Directory containing tracking results')
    parser.add_argument('--riskbench_annotations', type=str, required=True,
                        help='Path to RiskBench risk annotations')
    parser.add_argument('--output_dir', type=str, default='../models/rom_model',
                        help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Length of sequence for temporal features')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    return parser.parse_args()

# Custom dataset for ROM training
class ROMDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Neural network for ROM classification
class ROMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(ROMClassifier, self).__init__()
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Binary classification
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        x = self.relu(self.fc1(lstm_out))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x

# Function to extract features from tracking data
def extract_features_from_tracking(tracking_data, sequence_length):
    """
    Extract movement and interaction features from tracking data
    """
    features = []
    
    # Process each tracked object
    for track_id, track_data in tracking_data.items():
        # Sort track data by frame number
        track_data = sorted(track_data, key=lambda x: x['frame'])
        
        # Skip if track is too short
        if len(track_data) < sequence_length:
            continue
        
        # Extract sequences of features
        for i in range(len(track_data) - sequence_length + 1):
            sequence = track_data[i:i+sequence_length]
            
            # Extract features for each frame in the sequence
            sequence_features = []
            for j in range(len(sequence) - 1):  # -1 because we need next frame for velocity
                frame_data = sequence[j]
                next_frame_data = sequence[j+1]
                
                # Get bounding box
                x1, y1, x2, y2 = frame_data['bbox']
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Calculate velocity (change in position)
                next_x1, next_y1, next_x2, next_y2 = next_frame_data['bbox']
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
                    frame_data['class'],  # Object class
                    frame_data['confidence']  # Detection confidence
                ]
                
                sequence_features.append(frame_features)
            
            # Convert to numpy array
            sequence_features = np.array(sequence_features)
            features.append(sequence_features)
    
    return np.array(features)

# Function to load and preprocess RiskBench annotations
def load_risk_annotations(annotation_path):
    """
    Load risk annotations from RiskBench
    """
    with open(annotation_path, 'r') as f:
        risk_annotations = json.load(f)
    
    # Process annotations based on RiskBench format
    # This will need to be adapted to the actual RiskBench annotation format
    # For now, we'll assume a format where each object has a risk label
    
    risk_objects = {}
    for scenario, annotations in risk_annotations.items():
        for obj_id, risk_info in annotations.get('risk_objects', {}).items():
            # Store with scenario and object ID
            risk_objects[(scenario, obj_id)] = 1  # 1 for risk object
    
    return risk_objects

# Function to match tracking data with risk annotations
def match_tracking_with_annotations(tracking_data, risk_annotations, sequence_length):
    """
    Match tracking data with risk annotations to create labeled dataset
    """
    # Extract features
    all_features = []
    all_scenario_obj_ids = []
    
    # Process each video's tracking data
    for video_path, video_tracking_data in tracking_data.items():
        scenario_name = Path(video_path).stem
        
        # Extract features from this video
        video_features = extract_features_from_tracking(video_tracking_data, sequence_length)
        
        # Keep track of which scenario and object ID each feature sequence belongs to
        for track_id in video_tracking_data.keys():
            if len(video_tracking_data[track_id]) >= sequence_length:
                # For each possible sequence in this track
                for i in range(len(video_tracking_data[track_id]) - sequence_length + 1):
                    all_scenario_obj_ids.append((scenario_name, track_id))
        
        all_features.extend(video_features)
    
    # Create labels
    labels = []
    for scenario, obj_id in all_scenario_obj_ids:
        # Check if this object is a risk object
        is_risk = 1 if (scenario, obj_id) in risk_annotations else 0
        labels.append(is_risk)
    
    return np.array(all_features), np.array(labels)

# Train the ROM model
def train_rom_model(features, labels, args):
    """
    Train the ROM classifier model
    """
    # Convert to PyTorch tensors
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.FloatTensor(labels).unsqueeze(1)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features_tensor, labels_tensor, test_size=args.test_split, random_state=42
    )
    
    # Create data loaders
    train_dataset = ROMDataset(X_train, y_train)
    test_dataset = ROMDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    input_dim = features.shape[2]  # Number of features per time step
    model = ROMClassifier(input_dim=input_dim)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    train_losses = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Log training progress
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    loss_plot_path = os.path.join(args.output_dir, 'training_loss.png')
    plt.savefig(loss_plot_path)
    
    return model, X_test, y_test

# Evaluate the model
def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluate the trained model and report metrics
    """
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_binary = (y_pred > 0.5).float()
    
    # Convert to numpy for sklearn metrics
    y_true = y_test.numpy().flatten()
    y_pred_np = y_pred_binary.numpy().flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_np)
    precision = precision_score(y_true, y_pred_np, zero_division=0)
    recall = recall_score(y_true, y_pred_np, zero_division=0)
    f1 = f1_score(y_true, y_pred_np, zero_division=0)
    
    # Print metrics
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save metrics to file
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_np)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Non-Risk', 'Risk']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    
    return metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tracking data
    print("Loading tracking data...")
    tracking_data = {}
    for root, dirs, files in os.walk(args.tracking_dir):
        for file in files:
            if file == "tracking_data.json":
                video_name = os.path.basename(root)
                with open(os.path.join(root, file), 'r') as f:
                    video_tracking_data = json.load(f)
                tracking_data[video_name] = video_tracking_data
    
    print(f"Loaded tracking data for {len(tracking_data)} videos")
    
    # Load risk annotations
    print("Loading RiskBench annotations...")
    risk_annotations = load_risk_annotations(args.riskbench_annotations)
    print(f"Loaded {len(risk_annotations)} risk object annotations")
    
    # Match tracking data with risk annotations
    print("Preparing dataset...")
    features, labels = match_tracking_with_annotations(
        tracking_data, risk_annotations, args.sequence_length
    )
    
    print(f"Dataset prepared: {len(features)} samples, {sum(labels)} positive (risk) samples")
    
    # Train the model
    print("\nTraining ROM model...")
    model, X_test, y_test = train_rom_model(features, labels, args)
    
    # Evaluate the model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test, args.output_dir)
    
    # Save the model
    model_path = os.path.join(args.output_dir, 'rom_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save model configuration
    model_config = {
        'input_dim': features.shape[2],
        'sequence_length': args.sequence_length,
        'hidden_dim': 128,
        'num_layers': 2
    }
    
    config_path = os.path.join(args.output_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print("ROM training completed successfully")

if __name__ == "__main__":
    main()