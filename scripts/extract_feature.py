import pandas as pd
import numpy as np
import os

# Paths
input_csv = "features/tracking_output.csv"
output_csv = "features/track_summary.csv"

# Optional: resolution of the video
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Ignore non-moving object classes
IGNORED_CLASSES = {"traffic light", "stop sign", "traffic sign", "bench", "pole", "building", "potted plant"}

# Load CSV
df = pd.read_csv(input_csv)

# Compute center position and area
df["cx"] = (df["x1"] + df["x2"]) / 2
df["cy"] = (df["y1"] + df["y2"]) / 2
frame_cx = FRAME_WIDTH / 2
frame_cy = FRAME_HEIGHT / 2
df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
df["dist_to_center"] = np.sqrt((df["cx"] - frame_cx)**2 + (df["cy"] - frame_cy)**2)

# Sort by track/frame
df.sort_values(by=["track_id", "frame"], inplace=True)

# Compute speed per object (based on center movement)
speed_all = []
for tid, track in df.groupby("track_id"):
    dx = track["cx"].diff()
    dy = track["cy"].diff()
    speed = np.sqrt(dx**2 + dy**2).fillna(0)
    speed_all.extend(speed.values)
df["speed"] = speed_all

# Filter classes
df = df[~df["class"].isin(IGNORED_CLASSES)]

# Summarize per object (track_id)
summaries = []
for tid, group in df.groupby("track_id"):
    summary = {
        "track_id": tid,
        "class": group["class"].iloc[0],
        "max_speed": group["speed"].max(),
        "avg_speed": group["speed"].mean(),
        "max_area": group["area"].max(),
        "min_dist_to_center": group["dist_to_center"].min(),
        "num_frames": len(group),
        "start_frame": group["frame"].min(),
        "end_frame": group["frame"].max()
    }
    summaries.append(summary)

summary_df = pd.DataFrame(summaries)
os.makedirs("features", exist_ok=True)
summary_df.to_csv(output_csv, index=False)

print("Track summary features saved to:", output_csv)
