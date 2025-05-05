from ultralytics import YOLO
import os
import cv2
import pandas as pd
import torch

classes = pd.read_csv(
    "data/CUB_200_2011/classes.txt", sep=" ", names=["index", "class"]
)

class_names = list(classes["class"].str.split(".").str[-1].str.replace("_", " "))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "runs/train/yolov8n_cub200/weights/best.pt"
if not os.path.exists(model_path):
    print(f"Error: Fune-tuned model {model_path} not found")
    exit(1)

model = YOLO(model_path)

model.to(device)

image_path = (
    "data/CUB_200_2011/images/039.Least_Flycatcher/Least_Flycatcher_0001_30221.jpg"
)

if not os.path.exists(image_path):
    print(f"Error: Test iamge {image_path} not found")
    exit(1)

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Faield to load test iamge {image_path}")
    exit(1)


results = model.predict(image, device=device, imgsz=640, verbose=False)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = class_names[class_id]
        confidence = box.conf.item()
        coords = box.xyxy[0].tolist()
        print(f"Detected: {class_name} (Confidence: {confidence:.2f}) at {coords}")

annotated_iamge = results[0].plot()
output_path = "visualizations/YOLOv8_Tuned_Output.jpg"
cv2.imwrite(output_path, annotated_iamge)
print(f"Annotated image saved to: {output_path}")
