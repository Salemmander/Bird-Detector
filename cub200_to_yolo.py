import os
import shutil
from PIL import Image

# Paths to CUB-200-2011 dataset
cub_root = "data/CUB_200_2011"  # Replace with your CUB-200-2011 directory
output_root = "cub200_yolo"  # Output directory for YOLO format
image_dir = os.path.join(cub_root, "images")
bbox_file = os.path.join(cub_root, "bounding_boxes.txt")
split_file = os.path.join(cub_root, "train_test_split.txt")
image_list_file = os.path.join(cub_root, "images.txt")
class_label_file = os.path.join(cub_root, "image_class_labels.txt")

# Create output directories
os.makedirs(os.path.join(output_root, "images/train"), exist_ok=True)
os.makedirs(os.path.join(output_root, "images/val"), exist_ok=True)
os.makedirs(os.path.join(output_root, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(output_root, "labels/val"), exist_ok=True)

# Read image list
image_list = {}
with open(image_list_file, "r") as f:
    for line in f:
        img_id, img_path = line.strip().split()
        image_list[int(img_id)] = img_path

# Read train/test split
train_split = {}
with open(split_file, "r") as f:
    for line in f:
        img_id, is_train = line.strip().split()
        train_split[int(img_id)] = int(is_train)

# Read class labels
class_labels = {}
with open(class_label_file, "r") as f:
    for line in f:
        img_id, class_id = line.strip().split()
        class_labels[int(img_id)] = int(class_id) - 1  # Convert to 0-based indexing

# Read bounding boxes
bboxes = {}
with open(bbox_file, "r") as f:
    for line in f:
        img_id, x, y, w, h = map(float, line.strip().split())
        bboxes[int(img_id)] = (x, y, w, h)

# Convert to YOLO format
for img_id in image_list:
    # Get image path and split
    img_rel_path = image_list[img_id]
    img_path = os.path.join(cub_root, "images", img_rel_path)
    is_train = train_split[img_id]
    split_dir = "train" if is_train else "val"

    # Copy image
    img_name = os.path.basename(img_rel_path)
    output_img_path = os.path.join(output_root, f"images/{split_dir}/{img_name}")
    shutil.copy(img_path, output_img_path)

    # Get bounding box and convert to YOLO format
    x, y, w, h = bboxes[img_id]
    img = Image.open(img_path)
    img_w, img_h = img.size
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Write YOLO annotation (class_id = 0 for "bird")
    output_label_path = os.path.join(
        output_root, f"labels/{split_dir}/{img_name.replace('.jpg', '.txt')}"
    )
    with open(output_label_path, "w") as f:
        f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")

    # Save species class for later use (optional metadata)
    species_class = class_labels[img_id]
    # You can save this to a separate file or use it during training

print(f"Dataset prepared in: {output_root}")
