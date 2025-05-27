import os
import json
import shutil
import argparse
import urllib.request
import tarfile
import zipfile
import hashlib
import time
import sys
import random
from tqdm import tqdm
from pathlib import Path
import cv2
import re
import yaml

class Vedio:
    def __init__(self, vedio_source = '5_i-7_1_c_f_sl_0', variant_scenario = 'ClearSunset_high_'):
        self.vedio_source = vedio_source
        self.variant_scenario = variant_scenario
        self.images :dict[str,str] = {}
        self.crash_image_id: str = ""
        self.collision: dict = {}   
        self.annotation = {}

    def add_frame(self, image_path, annotation, frame_id):
        self.images[frame_id] = image_path
        self.annotation[frame_id] = annotation

    def set_collision(self, collision_data):
        self.collision = collision_data
        self.crash_image_id = collision_data['frame']

    def __str__(self):
        return f"Vedio: {self.vedio_source}, Variant Scenario: {self.variant_scenario}, Crash Image ID: {self.crash_image_id}, Collision: {self.collision} Image size: {len(self.images)}" 
class RiskBench:
    """
    A class to handle RiskBench dataset processing for YOLOv8.
    This class processes the RiskBench dataset and converts it to YOLOv8 format.
    """
    
    def __init__(self, riskbench_dir='../data/riskbench', output_dir='../data/riskbench_yolo', 
                 split_ratio=0.2, no_prepare=False):
        """
        Initialize the RiskBench processor.
        
        Args:
            riskbench_dir (str): Path to the root directory of RiskBench dataset
            output_dir (str): Output directory for the prepared dataset
            split_ratio (float): Validation split ratio
            no_prepare (bool): Flag to only download the dataset without preparing it for YOLOv8
        """
        self.riskbench_dir = riskbench_dir
        self.output_dir = output_dir
        self.split_ratio = split_ratio
        self.no_prepare = no_prepare
        self.vedio_data: list[Vedio] = []
        
        # Define class mappings
        self.class_mapping = {
            'vehicle': 0,
            'pedestrian': 1,
            'cyclist': 2,
            'traffic_sign': 3,
            'traffic_light': 4,
            # Add more classes as needed
        }
        
    @staticmethod
    def parse_args():
        """
        Parse command line arguments.
        
        Returns:
            argparse.Namespace: Parsed arguments
        """
        parser = argparse.ArgumentParser(description="Prepare RiskBench dataset for YOLOv8")
        parser.add_argument('--riskbench_dir', type=str, default='../data/riskbench',
                            help='Path to the root directory of RiskBench dataset')
        parser.add_argument('--output_dir', type=str, default='../data/riskbench_yolo',
                            help='Output directory for the prepared dataset')
        parser.add_argument('--split_ratio', type=float, default=0.2, 
                            help='Validation split ratio')
        parser.add_argument('--no_prepare', action='store_true',
                            help='Only download the dataset without preparing it for YOLOv8')
        return parser.parse_args()

    def create_yolo_label(self, annotation, img_width, img_height, output_label_path):
        """Convert RiskBench annotations to YOLO format.
        
        Args:
            annotation (dict): Annotation data containing objects with category and bbox
            img_width (int): Width of the image
            img_height (int): Height of the image
            output_label_path (str): Path to save the YOLO format label
        """
        with open(output_label_path, 'w') as f:
            for obj in annotation['objects']:
                # Get class ID from the class mapping
                class_id = self.get_class_id(obj['category'])
                
                # Get normalized bounding box coordinates (x_center, y_center, width, height)
                # RiskBench provides [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = obj['bbox']
                
                # Convert to YOLO format (normalized coordinates)
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # Write to file
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    def get_class_id(self, category_name):
        """Map RiskBench category names to class IDs.
        
        Args:
            category_name (str): Category name to map to ID
            
        Returns:
            int: Class ID for the category (-1 if unknown)
        """
        # Default to -1 for unknown classes
        return self.class_mapping.get(category_name.lower(), -1)
    
    def map_object_id_to_category(self, obj_id):
        """Map RiskBench object IDs to category names using actor_attribute.json.
        
        This method uses the preloaded actor_data to determine the category of an object
        based on its ID. The actor_attribute.json file contains detailed information about
        each actor in the scene, including its type.
        
        Args:
            obj_id (str or int): Object ID from RiskBench dataset
            
        Returns:
            str: Category name for the object ID ('vehicle', 'pedestrian', 'traffic_sign', 'traffic_light', 'cyclist' or 'unknown')
        """
        # Convert to string for dictionary lookup
        obj_id = str(obj_id)
        
        # Use the preloaded actor_data if available
        if hasattr(self, 'actor_data') and self.actor_data is not None:
            # Check if object ID exists in the vehicle category and is a bicycle/motorcycle (cyclist)
            if 'vehicle' in self.actor_data and obj_id in self.actor_data['vehicle']:
                actor_info = self.actor_data['vehicle'][obj_id]
                attributes = actor_info.get('attributes', {})
                base_type = attributes.get('base_type', '')
                special_type = attributes.get('special_type', '')
                
                if base_type == 'bicycle' or base_type == 'motorcycle' or special_type == 'motorcycle':
                    return 'cyclist'
                return 'vehicle'
            
            # Check if object ID exists in the pedestrian category
            if 'pedestrian' in self.actor_data and obj_id in self.actor_data['pedestrian']:
                return 'pedestrian'
            
            # Check if object ID exists in the traffic sign category
            if 'traffic.traffic_sign' in self.actor_data and obj_id in self.actor_data['traffic.traffic_sign']:
                return 'traffic_sign'
            
            # Check if object ID exists in the traffic light category
            if 'traffic.traffic_light' in self.actor_data and obj_id in self.actor_data['traffic.traffic_light']:
                return 'traffic_light'
        
        # Fallback to the original mapping if actor_attribute.json is not available or the ID is not found
            return 'unknown'

    def extract_frames_from_video(self, video_path, output_dir, frame_interval=1):
        """Extract frames from video files in RiskBench.
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Directory to save extracted frames
            frame_interval (int): Interval between extracted frames
            
        Returns:
            list: List of paths to extracted frames
        """
        video = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted_count = 0
        
        if not video.isOpened():
            print(f"Error opening video file: {video_path}")
            return []
        
        frame_paths = []
        
        while True:
            success, frame = video.read()
            if not success:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = f"{Path(video_path).stem}_{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                extracted_count += 1
            
            frame_count += 1
        
        video.release()
        print(f"Extracted {extracted_count} frames from {video_path}")
        return frame_paths

    def process_dataset(self):
        """Process RiskBench dataset and convert to YOLOv8 format.
        
        Finds all scenario directories, extracts images and annotations,
        and converts them to YOLO format.
        """
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        images_dir = os.path.join(self.output_dir, 'images')
        labels_dir = os.path.join(self.output_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Create train/val directories
        for split in ['train', 'val']:
            os.makedirs(os.path.join(images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
        
        # Find all scenario directories in RiskBench
        scenario_dirs = []
        
        # First look for variant_scenario directories which contain the actual data
        for root, dirs, files in os.walk(self.riskbench_dir):
            if 'variant_scenario' in dirs:
                variant_dir = os.path.join(root, 'variant_scenario')
                # Get all weather variants
                weather_variants = [os.path.join(variant_dir, d) for d in os.listdir(variant_dir) 
                                  if os.path.isdir(os.path.join(variant_dir, d))]
                scenario_dirs.extend(weather_variants)
        
        print(f"Found {len(scenario_dirs)} scenario varidants in RiskBench dataset")
        
        # Process each scenario
        all_data = []
        for scenario_dir in scenario_dirs:
            # In RiskBench, the structure is:
            # variant_scenario/[weather_condition]/rgb/front/[image files]
            # variant_scenario/[weather_condition]/bbox.json
            splited_scenario_dir = scenario_dir.split('\\')
            # splited_scenario_dir = splited_scenario_dir[-2].split('/') 

            vedio = Vedio(vedio_source=splited_scenario_dir[-3].split('/')[-1],variant_scenario=splited_scenario_dir[-1])
            self.vedio_data.append(vedio)
        
            rgb_dir = os.path.join(scenario_dir, 'rgb', 'front')
            bbox_path = os.path.join(scenario_dir, 'bbox.json')
            collision_path = os.path.join(scenario_dir, 'collision_frame.json')
            actor_attribute_path = os.path.join(scenario_dir, 'actor_attribute.json')
            
            if not os.path.exists(rgb_dir) or not os.path.exists(bbox_path) or not os.path.exists(collision_path):
                print(f"Missing rgb or bbox.json or collision_frame.json in {scenario_dir}, skipping...")
                continue
                
            # Set actor_attribute_path to None if file doesn't exist (for fallback behavior)
            if not os.path.exists(actor_attribute_path):
                print(f"Warning: actor_attribute.json not found in {scenario_dir}")
                actor_attribute_path = None
            
            try:
                # Load the bbox.json which contains annotations for all frames
                with open(bbox_path, 'r') as f:
                    bbox_data = json.load(f)
                
                # Store the current bbox_path for reference
                self.current_bbox_path = bbox_path
                
                # Load actor_attribute.json if it exists
                self.actor_data = None
                if actor_attribute_path and os.path.exists(actor_attribute_path):
                    try:
                        with open(actor_attribute_path, 'r') as f:
                            self.actor_data = json.load(f)
                    except Exception as e:
                        print(f"Error reading actor_attribute.json: {e}")
                        self.actor_data = None
                
                # Load the collision_frame.json which contains the collision frame
                with open(collision_path, 'r') as f:
                    collision_data = json.load(f)
                
                # Get all image files in the rgb/front directory
                image_files = [f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))]
                
                vedio.set_collision(collision_data)
                
                for img_file in sorted(image_files):
                    img_path = os.path.join(rgb_dir, img_file)
                    frame_id = img_file.split('.')[0]  # e.g., '00000001'
                    
                    # Check if annotation exists for this frame
                    if frame_id in bbox_data:
                        # Get image dimensions
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Could not read image: {img_path}, skipping...")
                            continue
                            
                        img_height, img_width = img.shape[:2]
                        
                        # Convert bbox data to a format our code can work with
                        objects = []
                        for obj_id, bbox in bbox_data[frame_id].items():
                            # Map object IDs to categories
                            category = self.map_object_id_to_category(obj_id)
                            
                            objects.append({
                                'category': category,
                                'bbox': bbox  # [x_min, y_min, x_max, y_max]
                            })
                        
                        # Create an annotation structure
                        annotation = {
                            'objects': objects
                        }
                        
                        # Add to dataset
                        all_data.append({
                            'image_path': img_path,
                            'annotation_path': bbox_path,  # Not used directly but kept for consistency
                            'width': img_width,
                            'height': img_height,
                            'annotation': annotation,
                            'frame_id': frame_id
                        })

                    vedio.add_frame(img_path, annotation, frame_id)
                # print(vedio)
            except Exception as e:
                print(f"Error processing scenario {scenario_dir}: {e}")
    
        # Split data into train/val
        random.shuffle(all_data)
        
        split_idx = int(len(all_data) * (1 - self.split_ratio))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        print(f"Split dataset: {len(train_data)} train, {len(val_data)} validation")
        
        self.map_path_to_data = {}
        # Process train data
        for i, item in enumerate(train_data):
            dest_img_path = os.path.join(images_dir, 'train', f"{i:06d}.jpg")
            dest_label_path = os.path.join(labels_dir, 'train', f"{i:06d}.txt")
            self.map_path_to_data[dest_img_path] = item
            # Copy image
            shutil.copy(item['image_path'], dest_img_path)
            
            # Create YOLO label
            self.create_yolo_label(item['annotation'], item['width'], item['height'], dest_label_path)
        
        # Process validation data
        for i, item in enumerate(val_data):
            dest_img_path = os.path.join(images_dir, 'val', f"{i:06d}.jpg")
            dest_label_path = os.path.join(labels_dir, 'val', f"{i:06d}.txt")
            self.map_path_to_data[dest_img_path] = item

            # Copy image
            shutil.copy(item['image_path'], dest_img_path)
            
            # Create YOLO label
            self.create_yolo_label(item['annotation'], item['width'], item['height'], dest_label_path)
        
        # Create dataset.yaml file for YOLOv8
        dataset_yaml = {
            'path': os.path.abspath(self.output_dir),
            'train': os.path.join('images', 'train'),
            'val': os.path.join('images', 'val'),
            'nc': len(self.class_mapping),  # Number of classes
            'names': list(self.class_mapping.keys())  # Class names
        }
        
        with open(os.path.join(self.output_dir, 'dataset.yaml'), 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        print(f"Dataset prepared successfully at {self.output_dir}")
        print(f"Created dataset.yaml with {dataset_yaml['nc']} classes")

    def get_class_mapping(self):
        """Get the full class mapping.
        
        Returns:
            dict: Dictionary mapping class names to class IDs
        """
        return self.class_mapping

    def check_dataset_exists(self):
        """Check if the dataset directory exists and contains valid data.
        
        Returns:
            bool: True if the dataset exists and contains valid data, False otherwise
        """
        if not os.path.exists(self.riskbench_dir):
            return False
        
        # Check for key directories/files that would indicate the dataset is present
        # Based on the actual structure of RiskBench
        
        # First check if there are any variant_scenario directories
        for root, dirs, _ in os.walk(self.riskbench_dir):
            if 'variant_scenario' in dirs:
                variant_dir = os.path.join(root, 'variant_scenario')
                # Check if there are any weather variant directories
                weather_variants = [d for d in os.listdir(variant_dir) 
                                  if os.path.isdir(os.path.join(variant_dir, d))]
                if weather_variants:
                    return True
        
        # As a fallback, look for any bbox.json files which are used for annotations
        for root, _, files in os.walk(self.riskbench_dir):
            if 'bbox.json' in files:
                return True
        
        return False

    def run(self):
        """Run the RiskBench dataset processing pipeline.
        
        This method handles the entire pipeline from checking if the dataset exists,
        downloading it if needed, and processing it for YOLOv8.
        """
        # Check if dataset exists or needs to be downloaded
        dataset_exists = self.check_dataset_exists()
        
        
        if not dataset_exists:
            print(f"Error: RiskBench dataset not found at {self.riskbench_dir}")
            print("Please download the dataset and try again, or use the --download option.")
            sys.exit(1)
        
        if not self.no_prepare:
            print("\nProcessing RiskBench dataset for YOLOv8...")
            self.process_dataset()


if __name__ == "__main__":
    args = RiskBench.parse_args()
    
    # Create a RiskBench instance with the parsed arguments
    riskbench = RiskBench(
        riskbench_dir=args.riskbench_dir,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
          no_prepare=args.no_prepare
    )
    
    # Run the RiskBench processing pipeline
    riskbench.run()