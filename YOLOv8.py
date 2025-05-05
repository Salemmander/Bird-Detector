from ultralytics import YOLO
import cv2
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the YOLOv8 detection model

model_path = "models/"
os.makedirs(model_path, exist_ok=True)
model = YOLO(model_path + "yolov8n.pt")

# Load an image
image_path = "data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Process and print detected objects with their classes
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)  # Class ID
        class_name = model.names[class_id]  # Class name (e.g., "dog", "cat")
        confidence = box.conf.item()  # Confidence score
        coords = box.xyxy[
            0
        ].tolist()  # Bounding box coordinates [x_min, y_min, x_max, y_max]
        print(f"Detected: {class_name} (Confidence: {confidence:.2f}) at {coords}")

# Optionally, display the image with bounding boxes
annotated_image = results[0].plot()  # Draw bounding boxes and labels
cv2.imwrite("visualizations/YOLOv8 Detection.jpg", annotated_image)
