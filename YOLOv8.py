from ultralytics import YOLO, settings
import torch
import os

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set the custom directory for downloading models
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)
    settings.update({"weights_dir": model_path})

    # Load the pre-trained YOLOv8 model
    model_name = "yolov8n.pt"
    model = YOLO(os.path.join(model_path, model_name))
    model.to(device)

    # Path to the dataset configuration file
    data_yaml = "N:/datasets/cub200_yolo/cub200.yaml"

    # Fine-tune the model
    model.train(
        data=data_yaml,  # Path to dataset YAML
        epochs=50,  # Number of training epochs
        imgsz=640,  # Image size
        batch=16,  # Batch size
        name="yolov8n_cub200",  # Name for saving the model
        project="runs/train",  # Directory to save training results
        device=device,  # Use GPU or CPU
        pretrained=True,  # Use pre-trained weights
    )

    print(f"Model {model_name} fine-tuned and saved in runs/train/yolov8n_cub200")
    print(f"Model weights downloaded to: {model_path}")
