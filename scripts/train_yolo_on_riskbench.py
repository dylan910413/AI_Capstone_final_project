from ultralytics import YOLO
import argparse
import os
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on RiskBench dataset")
    parser.add_argument('--data_yaml', type=str, required=True,
                        help='Path to the dataset.yaml file created by prepare_riskbench.py')
    parser.add_argument('--model_size', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (empty for auto, 0 for GPU 0, cpu for CPU)')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Directory to save the trained model')
    return parser.parse_args()

def train_yolo(args):
    """Train YOLOv8 on the RiskBench dataset."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset configuration
    with open(args.data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Print dataset information
    print(f"Training on dataset with {data_config['nc']} classes:")
    for i, name in enumerate(data_config['names']):
        print(f"  Class {i}: {name}")
    
    # Determine model to use
    model_name = f"yolov8{args.model_size}"
    if args.pretrained:
        print(f"Using pretrained model: {model_name}")
        model = YOLO(f"{model_name}.pt")
    else:
        print(f"Creating new model: {model_name}")
        model = YOLO(f"{model_name}.yaml")
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}...")
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.output_dir,
        name=f"riskbench_yolov8{args.model_size}",
        exist_ok=True
    )
    
    # Print training results
    print(f"Training completed. Model saved to {args.output_dir}/riskbench_yolov8{args.model_size}")
    print(f"Final metrics: {results}")
    
    # Return the path to the best model
    best_model_path = os.path.join(args.output_dir, f"riskbench_yolov8{args.model_size}", "weights", "best.pt")
    return best_model_path

if __name__ == "__main__":
    args = parse_args()
    trained_model_path = train_yolo(args)
    print(f"Best model saved to: {trained_model_path}")
    print("You can use this model for inference or continue with object tracking using DeepSORT.")