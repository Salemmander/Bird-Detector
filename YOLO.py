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
    model_path = "models"
    model_name = "yolo11n.pt"

    # region Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler with rotation to manage log file size
    log_dir = "logs/"
    log_file = os.path.join(log_dir, f"{model_name.split('.')[0]}_training.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    # Redirect stdout/stderr to logger
    sys.stdout = StreamToLogger(logger, logging.INFO)
    # model.train() seems to log to stderr not stdout this is a patch for the logging
    sys.stderr = StreamToLogger(logger, logging.INFO)
    # endregion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Model: {model_name}, Device: {device}, Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {img_size}"
    )

    os.makedirs(model_path, exist_ok=True)

    model = YOLO(os.path.join(model_path, model_name))
    model.to(device)

    data_yaml = "cub200_yolo/cub200.yaml"
    logger.info(f"Path to yaml: {data_yaml}")

    model_name = f"{model_name.split('.')[0]}"

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=model_name,
        exist_ok=True,  # Will overwrite runs with the same name
        project="runs/train",  # Directory to save training results
        device=device,
        pretrained=True,
    )

    logger.info(f"Model {model_name} fine-tuned and saved in runs/train/{model_name}")
