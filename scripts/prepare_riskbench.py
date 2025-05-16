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
from tqdm import tqdm
from pathlib import Path
import cv2
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare RiskBench dataset for YOLOv8")
    parser.add_argument('--riskbench_dir', type=str, default='../data/riskbench',
                        help='Path to the root directory of RiskBench dataset')
    parser.add_argument('--output_dir', type=str, default='../data/riskbench_yolo',
                        help='Output directory for the prepared dataset')
    parser.add_argument('--split_ratio', type=float, default=0.2, 
                        help='Validation split ratio')
    parser.add_argument('--download', action='store_true',
                        help='Download RiskBench dataset if not available')
    parser.add_argument('--download_subset', type=str, choices=['DATA_FOR_Planning_Aware_Metric', 'DATASET_for_LBC_Training'],
                        help='Download a specific subset of RiskBench instead of full dataset')
    parser.add_argument('--direct_url', type=str, default='',
                        help='Directly provide a URL to download a specific RiskBench tar/zip file')
    parser.add_argument('--no_prepare', action='store_true',
                        help='Only download the dataset without preparing it for YOLOv8')
    return parser.parse_args()

def create_yolo_label(annotation, img_width, img_height, output_label_path):
    """Convert RiskBench annotations to YOLO format."""
    # RiskBench likely has a specific annotation format
    # Here we assume annotations contain objects with class name and bounding boxes
    
    with open(output_label_path, 'w') as f:
        for obj in annotation['objects']:
            # Get class ID (you'll need to define your class mapping)
            class_id = get_class_id(obj['category'])
            
            # Get normalized bounding box coordinates (x_center, y_center, width, height)
            # Assuming RiskBench provides [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = obj['bbox']
            
            # Convert to YOLO format (normalized coordinates)
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Write to file
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def get_class_id(category_name):
    """Map RiskBench category names to class IDs."""
    # Define your class mapping here
    # This will likely be based on the categories in your RiskBench dataset
    class_mapping = {
        'vehicle': 0,
        'pedestrian': 1,
        'cyclist': 2,
        # Add more classes as needed
    }
    
    # Default to -1 for unknown classes
    return class_mapping.get(category_name.lower(), -1)

def extract_frames_from_video(video_path, output_dir, frame_interval=1):
    """Extract frames from video files in RiskBench."""
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

def process_riskbench_dataset(riskbench_dir, output_dir, split_ratio=0.2):
    """Process RiskBench dataset and convert to YOLOv8 format."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Create train/val directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # Find all scenario directories in RiskBench
    scenario_dirs = []
    for root, dirs, files in os.walk(riskbench_dir):
        # Adjust this condition based on actual RiskBench structure
        if 'rgb' in dirs or 'annotation' in dirs:
            scenario_dirs.append(root)
    
    print(f"Found {len(scenario_dirs)} scenarios in RiskBench dataset")
    
    # Process each scenario
    all_data = []
    
    for scenario_dir in scenario_dirs:
        # Get RGB data and annotations
        # This assumes the RiskBench structure has 'rgb' and 'annotation' directories
        rgb_dir = os.path.join(scenario_dir, 'rgb')
        annotation_dir = os.path.join(scenario_dir, 'annotation')
        
        if not os.path.exists(rgb_dir) or not os.path.exists(annotation_dir):
            continue
        
        # Check if RGB directory contains videos or images
        rgb_files = [f for f in os.listdir(rgb_dir) 
                     if f.endswith(('.mp4', '.avi', '.jpg', '.png'))]
        
        for rgb_file in rgb_files:
            rgb_path = os.path.join(rgb_dir, rgb_file)
            
            # Handle videos by extracting frames
            if rgb_file.endswith(('.mp4', '.avi')):
                temp_img_dir = os.path.join(output_dir, 'temp_frames')
                os.makedirs(temp_img_dir, exist_ok=True)
                
                frame_paths = extract_frames_from_video(rgb_path, temp_img_dir)
                
                # Process each extracted frame
                for frame_path in frame_paths:
                    frame_name = os.path.basename(frame_path)
                    frame_id = int(frame_name.split('_')[-1].split('.')[0])
                    
                    # Find corresponding annotation
                    anno_filename = f"{Path(rgb_file).stem}_{frame_id}.json"
                    anno_path = os.path.join(annotation_dir, anno_filename)
                    
                    if os.path.exists(anno_path):
                        try:
                            with open(anno_path, 'r') as f:
                                annotation = json.load(f)
                            
                            # Get image dimensions
                            img = cv2.imread(frame_path)
                            img_height, img_width = img.shape[:2]
                            
                            # Add to dataset
                            all_data.append({
                                'image_path': frame_path,
                                'annotation_path': anno_path,
                                'width': img_width,
                                'height': img_height,
                                'annotation': annotation
                            })
                        except Exception as e:
                            print(f"Error processing annotation {anno_path}: {e}")
                
                # Clean up temp directory when done with this video
                shutil.rmtree(temp_img_dir)
            
            # Handle individual images
            elif rgb_file.endswith(('.jpg', '.png')):
                # Find corresponding annotation
                anno_filename = f"{Path(rgb_file).stem}.json"
                anno_path = os.path.join(annotation_dir, anno_filename)
                
                if os.path.exists(anno_path):
                    try:
                        with open(anno_path, 'r') as f:
                            annotation = json.load(f)
                        
                        # Get image dimensions
                        img = cv2.imread(rgb_path)
                        img_height, img_width = img.shape[:2]
                        
                        # Add to dataset
                        all_data.append({
                            'image_path': rgb_path,
                            'annotation_path': anno_path,
                            'width': img_width,
                            'height': img_height,
                            'annotation': annotation
                        })
                    except Exception as e:
                        print(f"Error processing annotation {anno_path}: {e}")
    
    # Split data into train/val
    import random
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * (1 - split_ratio))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Split dataset: {len(train_data)} train, {len(val_data)} validation")
    
    # Process train data
    for i, item in enumerate(train_data):
        dest_img_path = os.path.join(images_dir, 'train', f"{i:06d}.jpg")
        dest_label_path = os.path.join(labels_dir, 'train', f"{i:06d}.txt")
        
        # Copy image
        shutil.copy(item['image_path'], dest_img_path)
        
        # Create YOLO label
        create_yolo_label(item['annotation'], item['width'], item['height'], dest_label_path)
    
    # Process validation data
    for i, item in enumerate(val_data):
        dest_img_path = os.path.join(images_dir, 'val', f"{i:06d}.jpg")
        dest_label_path = os.path.join(labels_dir, 'val', f"{i:06d}.txt")
        
        # Copy image
        shutil.copy(item['image_path'], dest_img_path)
        
        # Create YOLO label
        create_yolo_label(item['annotation'], item['width'], item['height'], dest_label_path)
    
    # Create dataset.yaml file for YOLOv8
    dataset_yaml = {
        'path': os.path.abspath(output_dir),
        'train': os.path.join('images', 'train'),
        'val': os.path.join('images', 'val'),
        'nc': len(get_class_mapping()),  # Number of classes
        'names': list(get_class_mapping().keys())  # Class names
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"Dataset prepared successfully at {output_dir}")
    print(f"Created dataset.yaml with {dataset_yaml['nc']} classes")

def get_class_mapping():
    """Get the full class mapping."""
    # This should return a dictionary mapping class names to class IDs
    return {
        'vehicle': 0,
        'pedestrian': 1,
        'cyclist': 2,
        # Add more classes as needed
    }

# Progress bar for downloads
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path, desc=None):
    """Download a file from a URL with a progress bar."""
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_riskbench(output_dir, subset=None, direct_url=None):
    """Download RiskBench dataset or a subset of it."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Base GitHub URLs
    github_base = "https://github.com/HCIS-Lab/RiskBench"
    
    if direct_url:
        # Direct URL to a specific file
        print(f"Downloading RiskBench file from direct URL: {direct_url}")
        filename = os.path.basename(direct_url)
        output_path = os.path.join(output_dir, filename)
        
        if download_url(direct_url, output_path, desc=f"Downloading {filename}"):
            print(f"Downloaded {filename} successfully")
            # Extract the file based on its extension
            extract_archive(output_path, output_dir)
            return True
        else:
            print(f"Failed to download from {direct_url}")
            return False
    
    # Handle specific subsets
    if subset:
        if subset == "DATA_FOR_Planning_Aware_Metric":
            # Example URL - adjust based on actual repository structure
            url = f"{github_base}/raw/main/DATA_FOR_Planning_Aware_Metric.zip"
            output_path = os.path.join(output_dir, "DATA_FOR_Planning_Aware_Metric.zip")
            
            if download_url(url, output_path, desc="Downloading Planning Aware Metric dataset"):
                print("Downloaded Planning Aware Metric dataset successfully")
                extract_archive(output_path, output_dir)
                return True
        
        elif subset == "DATASET_for_LBC_Training":
            # Example URL - adjust based on actual repository structure
            url = f"{github_base}/raw/main/DATASET_for_LBC_Training.zip"
            output_path = os.path.join(output_dir, "DATASET_for_LBC_Training.zip")
            
            if download_url(url, output_path, desc="Downloading LBC Training dataset"):
                print("Downloaded LBC Training dataset successfully")
                extract_archive(output_path, output_dir)
                return True
        
        print(f"Failed to download subset: {subset}")
        return False
    
    # If no specific subset, try to download the full dataset
    print("Attempting to download full RiskBench dataset...")
    print("Note: This may take a while as the full dataset is large.")
    print("RiskBench provides multiple ways to download the dataset. Please visit:")
    print(f"{github_base}")
    print("\nThe following options are available:")
    print("1. Use the --direct_url option with a specific file URL")
    print("2. Use the --download_subset option to download a specific subset")
    print("3. Manually download the dataset and use the --riskbench_dir option")
    print("\nExamples:")
    print("python prepare_riskbench.py --download --direct_url https://example.com/riskbench_file.tar")
    print("python prepare_riskbench.py --download --download_subset DATA_FOR_Planning_Aware_Metric")
    
    return False

def extract_archive(archive_path, output_dir):
    """Extract tar/zip archives."""
    print(f"Extracting {archive_path} to {output_dir}...")
    
    if archive_path.endswith('.tar') or archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path) as tar:
            # Get the total size for the progress bar
            total_size = sum(member.size for member in tar.getmembers())
            
            # Set up progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                for member in tar.getmembers():
                    tar.extract(member, path=output_dir)
                    pbar.update(member.size)
    
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as zip_ref:
            # Get the total size for the progress bar
            total_size = sum(info.file_size for info in zip_ref.infolist())
            
            # Set up progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                for member in zip_ref.infolist():
                    zip_ref.extract(member, path=output_dir)
                    pbar.update(member.file_size)
    
    print(f"Extraction of {archive_path} completed.")

def check_dataset_exists(dataset_dir):
    """Check if the dataset directory exists and contains valid data."""
    if not os.path.exists(dataset_dir):
        return False
    
    # Check for key directories/files that would indicate the dataset is present
    # Adjust this based on the actual structure of RiskBench
    potential_indicators = ['rgb', 'annotation', 'scenarios']
    
    # Check if any of the indicator directories exist
    for indicator in potential_indicators:
        if os.path.exists(os.path.join(dataset_dir, indicator)):
            return True
    
    # Look for any .json files which might be annotations
    for root, _, files in os.walk(dataset_dir):
        if any(f.endswith('.json') for f in files):
            return True
    
    return False

if __name__ == "__main__":
    args = parse_args()
    
    # Check if dataset exists or needs to be downloaded
    dataset_exists = check_dataset_exists(args.riskbench_dir)
    
    if args.download or (not dataset_exists and args.direct_url):
        print("Starting RiskBench dataset download...")
        success = download_riskbench(args.riskbench_dir, args.download_subset, args.direct_url)
        
        if not success:
            print("\nDownload failed or was not executed. Please:")
            print("1. Check your internet connection")
            print("2. Verify the download URLs")
            print("3. Consider manually downloading from the RiskBench GitHub repository")
            sys.exit(1)
        
        # Re-check if dataset exists after download
        dataset_exists = check_dataset_exists(args.riskbench_dir)
    
    if not dataset_exists:
        print(f"Error: RiskBench dataset not found at {args.riskbench_dir}")
        print("Please download the dataset and try again, or use the --download option.")
        sys.exit(1)
    
    if not args.no_prepare:
        print("\nProcessing RiskBench dataset for YOLOv8...")
        process_riskbench_dataset(args.riskbench_dir, args.output_dir, args.split_ratio)