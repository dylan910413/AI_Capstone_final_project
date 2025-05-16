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

### Running the Complete Pipeline

This project includes a unified pipeline manager (`main.py`) that orchestrates all steps from data preparation to final visualization:

```bash
python main.py --riskbench-dir data/riskbench
```

This will execute the complete pipeline with default settings:
1. Prepare the RiskBench dataset for YOLOv8
2. Train YOLOv8 on the prepared dataset
3. Perform object tracking with DeepSORT
4. Train the Risk Object Mining (ROM) classifier
5. Run the demo visualization

### Running Specific Pipeline Stages

You can run specific stages of the pipeline by specifying start and end stages:

```bash
python main.py --start-stage train_yolo --end-stage track_objects
```

Available stages: `prepare_data`, `train_yolo`, `track_objects`, `train_rom`, `run_demo`

### Common Configuration Options

#### Dataset Preparation
```bash
python main.py --start-stage prepare_data --end-stage prepare_data --riskbench-dir path/to/riskbench --val-split 0.2
```

#### YOLOv8 Training
```bash
python main.py --start-stage train_yolo --end-stage train_yolo --model-size s --epochs 50 --batch-size 16 --pretrained --device 0
```

#### Object Tracking
```bash
python main.py --start-stage track_objects --end-stage track_objects --test-video data/riskbench/sample_video.mp4 --conf-threshold 0.3 --save-video
```

#### ROM Classifier Training
```bash
python main.py --start-stage train_rom --end-stage train_rom --rom-epochs 50 --rom-batch-size 32 --learning-rate 0.001 --sequence-length 30
```

#### Running Demo
```bash
python main.py --start-stage run_demo --end-stage run_demo --demo-video data/riskbench/test_video.mp4 --risk-threshold 0.6
```

### Forcing Re-execution of Stages

By default, the pipeline skips stages that have already been completed. To force re-execution of stages:

```bash
python main.py --start-stage train_rom --force
```

### Advanced Pipeline Features

- **Progress Tracking**: The pipeline tracks progress in `pipeline_status.json`
- **Error Handling**: Detailed logging is saved to the `logs` directory
- **Resumable**: You can resume a failed pipeline from any stage
- **Parameter Management**: All parameters are stored for reproducibility

### Individual Script Usage

The component scripts can still be run individually if needed:

#### 1. Prepare RiskBench Dataset for YOLOv8
```bash
python scripts/prepare_riskbench.py --riskbench_dir path/to/riskbench --output_dir data/riskbench_yolo
```

#### 2. Train YOLOv8
```bash
python scripts/train_yolo_on_riskbench.py --data_yaml data/riskbench_yolo/dataset.yaml --model_size s --epochs 50
```

#### 3. Object Tracking
```bash
python scripts/track_objects.py --model_path models/riskbench_yolov8s/weights/best.pt --video_path data/riskbench/sample_video.mp4 --save_video
```

#### 4. Train ROM Classifier
```bash
python scripts/train_rom_classifier.py --tracking_dir data/tracking_results --riskbench_annotations data/riskbench/risk_annotations.json
```

#### 5. Run Demo
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
├── logs/               # Pipeline execution logs
├── models/             # Trained models
├── scripts/            # Component scripts
│   ├── prepare_riskbench.py       # Dataset preparation
│   ├── train_yolo_on_riskbench.py # YOLOv8 training
│   ├── track_objects.py           # Object tracking
│   ├── train_rom_classifier.py    # ROM classifier training
│   └── demo_risk_object_detection.py # Demo script
├── main.py             # Main pipeline manager
├── pipeline_status.json # Pipeline execution status
└── requirements.txt    # Project dependencies
```

## License

[Specify the license information here]

## Acknowledgments

- This project uses the RiskBench dataset from [HCIS-Lab](https://github.com/HCIS-Lab/RiskBench)
- Object detection is performed using [YOLOv8](https://github.com/ultralytics/ultralytics)
- Object tracking uses [DeepSORT](https://github.com/levan92/deep_sort_realtime)