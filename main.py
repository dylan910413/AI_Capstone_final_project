import os
import sys
import argparse
import logging
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Setup logging
def setup_logging(log_level=logging.INFO):
    """Configure logging for the application."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("rom_pipeline")

class PipelineStatus:
    """Class to track and manage pipeline execution status."""
    def __init__(self, status_file="pipeline_status.json"):
        self.status_file = status_file
        self.status = self._load_status()
    
    def _load_status(self):
        """Load pipeline status from file if exists."""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return self._create_default_status()
        else:
            return self._create_default_status()
    
    def _create_default_status(self):
        """Create default pipeline status structure."""
        return {
            "stages": {
                "prepare_data": {"completed": False, "timestamp": None},
                "train_yolo": {"completed": False, "timestamp": None},
                "track_objects": {"completed": False, "timestamp": None},
                "train_rom": {"completed": False, "timestamp": None},
                "run_demo": {"completed": False, "timestamp": None}
            },
            "parameters": {},
            "last_run": None
        }
    
    def update_stage(self, stage, completed=True):
        """Update the status of a pipeline stage."""
        if stage in self.status["stages"]:
            self.status["stages"][stage]["completed"] = completed
            self.status["stages"][stage]["timestamp"] = datetime.now().isoformat()
            self.status["last_run"] = datetime.now().isoformat()
            self._save_status()
        else:
            raise ValueError(f"Invalid stage name: {stage}")
    
    def is_stage_completed(self, stage):
        """Check if a stage has been completed."""
        if stage in self.status["stages"]:
            return self.status["stages"][stage]["completed"]
        else:
            raise ValueError(f"Invalid stage name: {stage}")
    
    def set_parameters(self, parameters):
        """Store pipeline parameters."""
        self.status["parameters"] = parameters
        self._save_status()
    
    def get_parameters(self):
        """Get stored pipeline parameters."""
        return self.status["parameters"]
    
    def reset_all(self):
        """Reset all pipeline stages to not completed."""
        for stage in self.status["stages"]:
            self.status["stages"][stage]["completed"] = False
            self.status["stages"][stage]["timestamp"] = None
        self._save_status()
    
    def reset_from_stage(self, stage):
        """Reset all stages starting from the specified stage."""
        stages = list(self.status["stages"].keys())
        if stage in stages:
            start_idx = stages.index(stage)
            for i in range(start_idx, len(stages)):
                current_stage = stages[i]
                self.status["stages"][current_stage]["completed"] = False
                self.status["stages"][current_stage]["timestamp"] = None
            self._save_status()
        else:
            raise ValueError(f"Invalid stage name: {stage}")
    
    def _save_status(self):
        """Save current status to file."""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

class ROMPipeline:
    """Main class to manage and execute the ROM pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.logger = setup_logging(
            logging.DEBUG if args.verbose else logging.INFO
        )
        self.status = PipelineStatus()
        
        # Store parameters for future reference and resuming
        self.status.set_parameters(vars(args))
        
        # Define paths
        self.scripts_dir = Path("scripts")
        self.data_dir = Path("data")
        self.models_dir = Path("models")
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Paths for each stage output
        self.riskbench_dir = self.data_dir / "riskbench"
        self.prepared_data_dir = self.data_dir / "riskbench_yolo"
        self.tracking_results_dir = self.data_dir / "tracking_results"
        self.yolo_model_dir = self.models_dir / f"riskbench_yolov8{args.model_size}"
        self.rom_model_dir = self.models_dir / "rom_model"
        
        # Check if required directories exist based on stage
        self._check_required_directories()
    
    def _check_required_directories(self):
        """Check if required directories exist based on the starting stage."""
        stages = ["prepare_data", "train_yolo", "track_objects", "train_rom", "run_demo"]
        start_stage_idx = stages.index(self.args.start_stage)
        
        # If not starting from the beginning, check if previous stage outputs exist
        if start_stage_idx > 0:
            if start_stage_idx >= 1 and not self.riskbench_dir.exists():
                self.logger.error(f"Missing required directory: {self.riskbench_dir}")
                sys.exit(1)
            
            if start_stage_idx >= 2 and not self.prepared_data_dir.exists():
                self.logger.error(f"Missing required directory: {self.prepared_data_dir}")
                sys.exit(1)
            
            if start_stage_idx >= 3 and not self.yolo_model_dir.exists():
                self.logger.error(f"Missing required directory: {self.yolo_model_dir}")
                sys.exit(1)
            
            if start_stage_idx >= 4 and not self.tracking_results_dir.exists():
                self.logger.error(f"Missing required directory: {self.tracking_results_dir}")
                sys.exit(1)
            
            if start_stage_idx >= 5 and not self.rom_model_dir.exists():
                self.logger.error(f"Missing required directory: {self.rom_model_dir}")
                sys.exit(1)
    
    def _run_script(self, script_name, args_list, stage_name, check_error=True):
        """Execute a Python script with arguments and handle errors."""
        script_path = self.scripts_dir / script_name
        cmd = [sys.executable, str(script_path)] + args_list
        
        self.logger.info(f"Running {script_name} with args: {' '.join(args_list)}")
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Stream output in real-time
            for line in process.stdout:
                self.logger.info(line.strip())
            
            # Wait for process to complete
            process.wait()
            
            # Check for errors
            if process.returncode != 0 and check_error:
                stderr = process.stderr.read()
                self.logger.error(f"Error in {script_name}: {stderr}")
                return False
            
            # Mark stage as completed
            self.status.update_stage(stage_name, completed=True)
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing {script_name}: {str(e)}")
            return False
    
    def prepare_data(self):
        """Run the data preparation stage."""
        self.logger.info("Starting data preparation stage")
        
        if self.status.is_stage_completed("prepare_data") and not self.args.force:
            self.logger.info("Data preparation already completed. Use --force to rerun.")
            return True
        
        args_list = [
            "--riskbench_dir", str(self.riskbench_dir),
            "--output_dir", str(self.prepared_data_dir),
            "--split_ratio", str(self.args.val_split)
        ]
        
        success = self._run_script("prepare_riskbench.py", args_list, "prepare_data")
        if success:
            self.logger.info("Data preparation completed successfully")
        else:
            self.logger.error("Data preparation failed")
        
        return success
    
    def train_yolo(self):
        """Run the YOLOv8 training stage."""
        self.logger.info("Starting YOLOv8 training stage")
        
        if self.status.is_stage_completed("train_yolo") and not self.args.force:
            self.logger.info("YOLOv8 training already completed. Use --force to rerun.")
            return True
        
        data_yaml = str(self.prepared_data_dir / "dataset.yaml")
        
        args_list = [
            "--data_yaml", data_yaml,
            "--model_size", self.args.model_size,
            "--epochs", str(self.args.epochs),
            "--batch_size", str(self.args.batch_size),
            "--output_dir", str(self.models_dir)
        ]
        
        if self.args.pretrained:
            args_list.append("--pretrained")
        
        if self.args.device:
            args_list.extend(["--device", self.args.device])
        
        success = self._run_script("train_yolo_on_riskbench.py", args_list, "train_yolo")
        if success:
            self.logger.info("YOLOv8 training completed successfully")
        else:
            self.logger.error("YOLOv8 training failed")
        
        return success
    
    def track_objects(self):
        """Run the object tracking stage."""
        self.logger.info("Starting object tracking stage")
        
        if self.status.is_stage_completed("track_objects") and not self.args.force:
            self.logger.info("Object tracking already completed. Use --force to rerun.")
            return True
        
        model_path = str(self.yolo_model_dir / "weights" / "best.pt")
        
        # Determine video path (file or directory)
        video_path = self.args.test_video if self.args.test_video else str(self.riskbench_dir)
        
        args_list = [
            "--model_path", model_path,
            "--video_path", video_path,
            "--conf_threshold", str(self.args.conf_threshold),
            "--output_dir", str(self.tracking_results_dir)
        ]
        
        if self.args.save_video:
            args_list.append("--save_video")
        
        success = self._run_script("track_objects.py", args_list, "track_objects")
        if success:
            self.logger.info("Object tracking completed successfully")
        else:
            self.logger.error("Object tracking failed")
        
        return success
    
    def train_rom(self):
        """Run the ROM classifier training stage."""
        self.logger.info("Starting ROM classifier training stage")
        
        if self.status.is_stage_completed("train_rom") and not self.args.force:
            self.logger.info("ROM training already completed. Use --force to rerun.")
            return True
        
        # Path to RiskBench risk annotations
        risk_annotations = str(self.riskbench_dir / "risk_annotations.json")
        
        args_list = [
            "--tracking_dir", str(self.tracking_results_dir),
            "--riskbench_annotations", risk_annotations,
            "--output_dir", str(self.rom_model_dir),
            "--epochs", str(self.args.rom_epochs),
            "--batch_size", str(self.args.rom_batch_size),
            "--learning_rate", str(self.args.learning_rate),
            "--sequence_length", str(self.args.sequence_length),
            "--test_split", str(self.args.test_split)
        ]
        
        success = self._run_script("train_rom_classifier.py", args_list, "train_rom")
        if success:
            self.logger.info("ROM classifier training completed successfully")
        else:
            self.logger.error("ROM classifier training failed")
        
        return success
    
    def run_demo(self):
        """Run the demo visualization stage."""
        self.logger.info("Starting demo visualization stage")
        
        # If no demo video is specified, skip this stage
        if not self.args.demo_video:
            self.logger.info("No demo video specified. Skipping demo stage.")
            return True
        
        yolo_model_path = str(self.yolo_model_dir / "weights" / "best.pt")
        rom_model_path = str(self.rom_model_dir / "rom_model.pt")
        rom_config_path = str(self.rom_model_dir / "model_config.json")
        
        args_list = [
            "--yolo_model", yolo_model_path,
            "--rom_model", rom_model_path,
            "--rom_config", rom_config_path,
            "--video_path", self.args.demo_video,
            "--output_video", self.args.output_video,
            "--conf_threshold", str(self.args.conf_threshold),
            "--risk_threshold", str(self.args.risk_threshold)
        ]
        
        success = self._run_script("demo_risk_object_detection.py", args_list, "run_demo")
        if success:
            self.logger.info("Demo visualization completed successfully")
            self.logger.info(f"Output video saved to: {self.args.output_video}")
        else:
            self.logger.error("Demo visualization failed")
        
        return success
    
    def run_pipeline(self):
        """Execute the complete pipeline or a subset of stages."""
        self.logger.info("Starting ROM pipeline execution")
        
        stages = {
            "prepare_data": self.prepare_data,
            "train_yolo": self.train_yolo,
            "track_objects": self.track_objects,
            "train_rom": self.train_rom,
            "run_demo": self.run_demo
        }
        
        # Get the list of stages to run
        start_idx = list(stages.keys()).index(self.args.start_stage)
        end_idx = list(stages.keys()).index(self.args.end_stage)
        
        if start_idx > end_idx:
            self.logger.error(f"Start stage '{self.args.start_stage}' comes after end stage '{self.args.end_stage}'")
            return False
        
        stages_to_run = list(stages.keys())[start_idx:end_idx+1]
        
        # Reset status for stages that will be run
        if self.args.force:
            self.status.reset_from_stage(self.args.start_stage)
        
        # Run each stage
        for stage in stages_to_run:
            self.logger.info(f"=" * 80)
            self.logger.info(f"PIPELINE STAGE: {stage}")
            self.logger.info(f"=" * 80)
            
            stage_func = stages[stage]
            success = stage_func()
            
            if not success:
                self.logger.error(f"Pipeline failed at stage: {stage}")
                return False
        
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
        return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Risk Object Mining Pipeline")
    
    # Pipeline control
    parser.add_argument("--start-stage", type=str, choices=["prepare_data", "train_yolo", "track_objects", "train_rom", "run_demo"], 
                        default="prepare_data", help="Stage to start the pipeline from")
    parser.add_argument("--end-stage", type=str, choices=["prepare_data", "train_yolo", "track_objects", "train_rom", "run_demo"], 
                        default="run_demo", help="Stage to end the pipeline at")
    parser.add_argument("--force", action="store_true", help="Force re-run of completed stages")
    
    # Dataset preparation
    parser.add_argument("--riskbench-dir", type=str, default="data/riskbench", 
                        help="Path to RiskBench dataset")
    parser.add_argument("--val-split", type=float, default=0.2, 
                        help="Validation split ratio")
    
    # YOLOv8 training
    parser.add_argument("--model-size", type=str, choices=["n", "s", "m", "l", "x"], 
                        default="s", help="YOLOv8 model size")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs for YOLOv8")
    parser.add_argument("--batch-size", type=int, default=16, 
                        help="Batch size for YOLOv8 training")
    parser.add_argument("--pretrained", action="store_true", 
                        help="Use pretrained weights for YOLOv8")
    parser.add_argument("--device", type=str, default="", 
                        help="Device to use for training ('' for auto, '0' for GPU 0, 'cpu' for CPU)")
    
    # Object tracking
    parser.add_argument("--test-video", type=str, default="", 
                        help="Path to video file for testing tracking (if empty, will process all videos in RiskBench)")
    parser.add_argument("--conf-threshold", type=float, default=0.3, 
                        help="Confidence threshold for object detection")
    parser.add_argument("--save-video", action="store_true", 
                        help="Save output videos with tracking visualizations")
    
    # ROM training
    parser.add_argument("--rom-epochs", type=int, default=50, 
                        help="Number of training epochs for ROM classifier")
    parser.add_argument("--rom-batch-size", type=int, default=32, 
                        help="Batch size for ROM classifier training")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                        help="Learning rate for ROM classifier training")
    parser.add_argument("--sequence-length", type=int, default=30, 
                        help="Length of sequence for temporal features")
    parser.add_argument("--test-split", type=float, default=0.2, 
                        help="Proportion of data to use for testing ROM model")
    
    # Demo
    parser.add_argument("--demo-video", type=str, default="", 
                        help="Path to video file for demo visualization")
    parser.add_argument("--output-video", type=str, default="demo/risk_detection_output.mp4", 
                        help="Path to save output demo video")
    parser.add_argument("--risk-threshold", type=float, default=0.6, 
                        help="Threshold for classifying an object as risky")
    
    # Other
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    pipeline = ROMPipeline(args)
    success = pipeline.run_pipeline()
    
    if success:
        print("\nPipeline completed successfully!")
        return 0
    else:
        print("\nPipeline failed. See logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
