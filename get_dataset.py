import kaggle
import os

# Authenticate with Kaggle API (ensure Kaggle API token is set up in ~/.kaggle/kaggle.json)
kaggle.api.authenticate()

# Define dataset and download path
dataset = "wenewone/cub2002011"
download_path = "/media/salem/NVME/Projects/Bird-Detector/data"

# Create download directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# Download and unzip the dataset
kaggle.api.dataset_download_files(dataset, path=download_path, unzip=True)
