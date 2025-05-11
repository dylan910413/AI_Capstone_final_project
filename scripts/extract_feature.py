import pandas as pd
import os
import numpy as np

# Paths
input_csv = "features/tracking_output.csv"
output_csv = "features/motion_features.csv"

# Optional: resolution to compute center distance
FRAME_WIDTH = 640   # adjust based on your video
FRAME_HEIGHT = 360

# Ignore static classes
IGNORED_CLASSES = {"traffic light", "stop sign", "traffic sign", "bench", "pole", "building", "potted plant"}

# Load tracking data
df = pd.read_csv(input_csv)

# Compute center coordinates and bbox area
df["cx"] = (df["x1"] + df["x2"]) / 2
df["cy"] = (df["y1"] + df["y2"]) / 2
df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])

# Compute distance to image center
frame_cx = FRAME_WIDTH / 2
frame_cy = FRAME_HEIGHT / 2
df["dist_to_center"] = np.sqrt((df["cx"] - frame_cx)**2 + (df["cy"] - frame_cy)**2)

# Sort for consistent processing
df.sort_values(by=["track_id", "frame"], inplace=True)

# Compute speed (pixel/frame) per track_id
df["speed"] = 0.0
for tid in df["track_id"].unique():
    track = df[df["track_id"] == tid]
    dx = track["cx"].diff()
    dy = track["cy"].diff()
    speed = np.sqrt(dx**2 + dy**2)
    df.loc[track.index, "speed"] = speed.fillna(0)

# Filter out ignored classes
df = df[~df["class"].isin(IGNORED_CLASSES)]

# Optional: filter out static objects with very low speed
MIN_SPEED = 1.0  # adjust threshold as needed
grouped = df.groupby("track_id")
moving_ids = [tid for tid, g in grouped if g["speed"].max() >= MIN_SPEED]
df = df[df["track_id"].isin(moving_ids)]

# Save to new CSV
os.makedirs("features", exist_ok=True)
df.to_csv(output_csv, index=False)
print("Motion features extracted and saved to:", output_csv)
