# Risk Object Mining in Traffic Scenes

**AI Capstone Project - Traffic Scene Understanding**  
*109611092 李耀凱 109700046 侯均頲*

## Overview

This project automatically mines and identifies critical objects in traffic scenarios that significantly influence driver behavior and increase accident risk. It combines object detection, tracking, and risk analysis to identify dangerous objects in traffic scenes.

## System Architecture

The system consists of three main components:

1. **Object Detection**: Uses YOLOv8 to detect dynamic objects such as vehicles, pedestrians, and cyclists in each frame.
2. **Object Tracking**: Utilizes DeepSORT to associate detected objects across frames and assign consistent object IDs.
3. **Risk Object Mining (ROM)**: Uses a neural network to analyze object movement patterns and identify risky objects.

## Dataset

This project uses the [RiskBench dataset](https://github.com/HCIS-Lab/RiskBench), a comprehensive benchmark for risk assessment in autonomous driving.

## Setup and Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd AI_Capstone_final_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Downloading RiskBench Dataset

You can download the RiskBench dataset using one of the following methods:

### Option 1: Direct Download (Recommended)

Visit the [RiskBench GitHub page](https://github.com/HCIS-Lab/RiskBench) and follow their instructions to download the dataset. RiskBench provides multiple options for downloading:
- Split into large zip files
- Split into 2GB tar files (recommended for easier handling)

Download the files to a `data/riskbench` directory within this project.

### Option 2: Using Git LFS

If the repository uses Git LFS for large files:
```bash
git lfs install
git clone https://github.com/HCIS-Lab/RiskBench.git data/riskbench
```

## Usage Instructions

### 1. Prepare RiskBench Dataset for YOLOv8

Convert the RiskBench dataset to a format suitable for YOLOv8:

```bash
python scripts/prepare_riskbench.py --riskbench_dir path/to/riskbench --output_dir data/riskbench_yolo
```

### 2. Train YOLOv8 on RiskBench

Train a YOLOv8 model on the prepared dataset:

```bash
python scripts/train_yolo_on_riskbench.py --data_yaml data/riskbench_yolo/dataset.yaml --model_size s --epochs 50
```

Options:
- `--model_size`: YOLOv8 model size (n, s, m, l, x)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--pretrained`: Use pretrained weights (flag)

### 3. Object Tracking with YOLOv8 and DeepSORT

Track objects in videos using the trained YOLOv8 model and DeepSORT:

```bash
python scripts/track_objects.py --model_path models/riskbench_yolov8s/weights/best.pt --video_path data/riskbench/sample_video.mp4 --save_video
```

For batch processing of multiple videos:
```bash
python scripts/track_objects.py --model_path models/riskbench_yolov8s/weights/best.pt --video_path data/riskbench/videos --save_video
```

### 4. Train Risk Object Mining (ROM) Classifier

Train a neural network to identify risk objects based on tracking data:

```bash
python scripts/train_rom_classifier.py --tracking_dir data/tracking_results --riskbench_annotations data/riskbench/risk_annotations.json
```

### 5. Run the Complete System Demo

Demonstrate the complete system on a video:

```bash
python scripts/demo_risk_object_detection.py --yolo_model models/riskbench_yolov8s/weights/best.pt --rom_model models/rom_model/rom_model.pt --rom_config models/rom_model/model_config.json --video_path data/riskbench/test_video.mp4
```

## Evaluation

The system is evaluated using:
- Accuracy, precision, recall, and F1 score for the ROM classifier
- Visualizations of identified risk objects in traffic scenes

## Directory Structure

```
AI_Capstone_final_project/
├── data/               # Data directory
│   └── riskbench/      # RiskBench dataset
│   └── riskbench_yolo/ # Processed dataset for YOLOv8
│   └── tracking_results/ # Results from object tracking
├── demo/               # Demo videos and visualizations
├── features/           # Feature extraction code
├── models/             # Trained models
├── scripts/            # Main scripts
│   ├── prepare_riskbench.py       # Dataset preparation
│   ├── train_yolo_on_riskbench.py # YOLOv8 training
│   ├── track_objects.py           # Object tracking
│   ├── train_rom_classifier.py    # ROM classifier training
│   └── demo_risk_object_detection.py # Demo script
└── requirements.txt    # Project dependencies
```

## License

[Specify the license information here]

## Acknowledgments

- This project uses the RiskBench dataset from [HCIS-Lab](https://github.com/HCIS-Lab/RiskBench)
- Object detection is performed using [YOLOv8](https://github.com/ultralytics/ultralytics)
- Object tracking uses [DeepSORT](https://github.com/levan92/deep_sort_realtime)