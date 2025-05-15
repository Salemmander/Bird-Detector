from ultralytics import YOLO
import torch
import os
import logging
import sys
from logging.handlers import RotatingFileHandler


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
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(
        "N:/Projects/Bird Detection/logs/test_max_det_1.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(
        logger, logging.INFO
    )  # model.val() seems to log to error not info so this is a patch

    logging.info("Starting YOLOv8 model testing")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the fine-tuned model
    model_path = "N:/Projects/Bird Detection/runs/train/yolov8l_cub200/weights/best.pt"
    if not os.path.exists(model_path):
        logging.error(f"Model {model_path} not found")
        sys.exit(1)
    model = YOLO(model_path)
    model.to(device)
    logging.info(f"Loaded model: {model_path}")

    # Path to dataset configuration
    data_yaml = "N:/Projects/Bird Detection/cub200_yolo/cub200.yaml"
    if not os.path.exists(data_yaml):
        logging.error(f"Dataset YAML {data_yaml} not found")
        sys.exit(1)
    logging.info(f"Using dataset configuration: {data_yaml}")

    # Evaluate on validation set
    logging.info("Running validation on validation set")
    val_results = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        device=device,
        max_det=1,
        save_json=True,  # Save detailed results
    )
    logging.info("Validation results:")
    logging.info(f"mAP50: {val_results.box.map50:.4f}")
    logging.info(f"mAP50-95: {val_results.box.map:.4f}")
    logging.info(f"Validation results saved to {val_results.save_dir}")
