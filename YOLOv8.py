from logging.handlers import RotatingFileHandler
from ultralytics import YOLO, settings
import torch
import os
import sys
import logging


# Custom stream handler to redirect stdout/stderr to logger
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


if __name__ == "__main__":

    epochs = 50
    batch_size = 8
    img_size = 640
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler with rotation to manage log file size
    file_handler = RotatingFileHandler(
        "N:/Projects/Bird Detection/logs/training.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    # Redirect stdout/stderr to logger
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {img_size}")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set the custom directory for downloading models
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)
    settings.update({"weights_dir": model_path})
    # Load the pre-trained YOLOv8 model
    model_name = "yolov8l.pt"
    model = YOLO(os.path.join(model_path, model_name))
    model.to(device)
    logger.info(f"Model being tuned: {model_name}")
    # Path to the dataset configuration file
    data_yaml = "N:/Projects/Bird Detection/cub200_yolo/cub200.yaml"
    logger.info(f"Path to yaml: {data_yaml}")

    # Fine-tune the model
    model.train(
        data=data_yaml,  # Path to dataset YAML
        epochs=epochs,  # Number of training epochs
        imgsz=img_size,  # Image size
        batch=batch_size,  # Batch size
        name="yolov8l_cub200",  # Name for saving the model
        project="runs/train",  # Directory to save training results
        device=device,  # Use GPU or CPU
        pretrained=True,  # Use pre-trained weights
    )

    logger.info(f"Model {model_name} fine-tuned and saved in runs/train/yolov8l_cub200")
