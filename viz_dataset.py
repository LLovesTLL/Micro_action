
import os
import cv2
import numpy as np
import decord
from decord import VideoReader, cpu

# Paths
dataset_root = "/home/xcguo/Project/Micro_action/datasets"
train_list_path = os.path.join(dataset_root, "annotations/train_list_videos.txt")
label_map_path = os.path.join(dataset_root, "annotations/label_name.txt")
video_root = os.path.join(dataset_root, "train")

# Load label map
label_map = {}
with open(label_map_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            idx = int(parts[0])
            name = " ".join(parts[1:])
            label_map[idx] = name

# Load training list
train_list = []
with open(train_list_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            filename = parts[0]
            label = int(parts[1])
            train_list.append((filename, label))

if not train_list:
    print("Error: train_list is empty. Check path:", train_list_path)
    exit(1)

# Sample 4 random videos
indices = np.random.choice(len(train_list), 4, replace=False)
samples = [train_list[i] for i in indices]

# Collect rows of images
rows = []
FRAME_SIZE = (224, 224) 

for idx, (filename, label_idx) in enumerate(samples):
    video_path = os.path.join(video_root, filename)
    label_name = label_map.get(label_idx, str(label_idx))
    
    row_images = []
    try:
        if not os.path.exists(video_path):
             print(f"File not found: {video_path}")
             raise FileNotFoundError
             
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        # Sample 8 frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, 8).astype(int)
        frames = vr.get_batch(frame_indices).asnumpy() # (8, H, W, 3) RGB

        for i, frame in enumerate(frames):
            # Resize for consistent grid
            img = cv2.resize(frame, FRAME_SIZE)
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Add label to first frame
            if i == 0:
                cv2.putText(img_bgr, f"{label_name}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            row_images.append(img_bgr)
            
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        # Create blank images on error
        for i in range(8):
            blank = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            if i == 0:
                cv2.putText(blank, "Error", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            row_images.append(blank)
    
    # Concatenate frames horizontally with some spacing
    spacer = np.zeros((FRAME_SIZE[1], 10, 3), dtype=np.uint8)
    row_concat = row_images[0]
    for img in row_images[1:]:
        row_concat = np.hstack((row_concat, spacer, img))
    
    rows.append(row_concat)

# Concatenate rows vertically with spacing
final_image = rows[0]
dataset_spacer = np.zeros((30, final_image.shape[1], 3), dtype=np.uint8)

for row in rows[1:]:
    final_image = np.vstack((final_image, dataset_spacer, row))

output_file = "/home/xcguo/Project/Micro_action/dataset_preview.png"
cv2.imwrite(output_file, final_image)
print(f"Dataset preview saved to {output_file}")
