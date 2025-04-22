import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CUB200(Dataset):

    def __init__(self, data_root_path):

        self.img_size = 448
        self.num_grid = 7
        self.grid_size = self.img_size / self.num_grid
        self.pixel_mean = np.array([[[110.2801, 127.6312, 124.0555]]], dtype=np.float32)
        self.image_root_path = os.path.join(data_root_path, "CUB_200_2011/images")

        self.data_df = pd.read_csv(
            os.path.join(data_root_path, "CUB_200_2011/images.txt"),
            sep=" ",
            names=["index", "file_path"],
        )
        bboxes = pd.read_csv(
            os.path.join(data_root_path, "CUB_200_2011/bounding_boxes.txt"),
            sep=" ",
            names=["index", "x", "y", "width", "height"],
        )
        classes = pd.read_csv(
            os.path.join(data_root_path, "CUB_200_2011/image_class_labels.txt"),
            sep=" ",
            names=["index", "class"],
        )

        self.data_df = self.data_df.merge(bboxes, left_on="index", right_on="index")
        self.data_df = self.data_df.merge(classes, left_on="index", right_on="index")

    def __getitem__(self, index):
        data = self.data_df.iloc[index]
        file_path = data["file_path"]
        # region Image
        img_path = os.path.join(self.image_root_path, file_path)
        image = cv2.imread(img_path)
        img_height, img_width, _ = image.shape

        image = cv2.resize(image, (self.img_size, self.img_size))
        image = (image - self.pixel_mean) / 255
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image)
        # endregion

        # region Box
        # 1.0 is confidence value
        bbox = [data["x"], data["y"], data["width"], data["height"], 1.0]
        # region Scaling
        width_factor = self.img_size / img_width
        height_factor = self.img_size / img_height
        bbox[0] *= width_factor
        bbox[1] *= height_factor
        bbox[2] *= width_factor
        bbox[3] *= height_factor
        # endregion

        # region Refactor
        cx = bbox[0] + (bbox[2] / 2)
        cy = bbox[1] + (bbox[3] / 2)
        bbox[0] = cx
        bbox[1] = cy
        # endregion

        # region Normalization
        x_grid = math.floor(bbox[0] / self.grid_size)
        y_grid = math.floor(bbox[1] / self.grid_size)

        x_offset = (bbox[0] - x_grid * self.grid_size) / self.grid_size
        y_offset = (bbox[1] - y_grid * self.grid_size) / self.grid_size
        w_offset = bbox[2] / self.img_size
        h_offset = bbox[3] / self.img_size

        bbox[0] = x_offset
        bbox[1] = y_offset
        bbox[2] = w_offset
        bbox[3] = h_offset
        # endregion
        gt_box = torch.zeros(5, 7, 7)
        gt_box[:, x_grid, y_grid] = torch.tensor(bbox)
        # endregion

        # region Mask
        gt_mask = torch.zeros(7, 7)
        gt_mask[x_grid, y_grid] = 1
        # endregion

        return {"image": image, "gt_box": gt_box, "gt_mask": gt_mask}

    def __len__(self):
        return len(self.data_df)


def draw_image(image, bbox, mask, pixel_mean, img_name="test"):
    image = image.numpy().transpose((1, 2, 0))
    image = image * 255 + pixel_mean
    image = image[:, :, ::-1]
    image = image.astype(np.uint8)
    bbox = bbox.numpy()
    mask = mask.numpy()

    x, y = np.where(mask == 1)
    cx = bbox[0, x, y] * 64 + x * 64
    cy = bbox[1, x, y] * 64 + y * 64
    w = bbox[2, x, y] * 448
    h = bbox[3, x, y] * 448
    x1 = cx - (w / 2)
    y1 = cy - (h / 2)

    fig, ax = plt.subplots()

    ax.imshow(image)
    ax.axis("off")

    # Draw bounding box
    rect = patches.Rectangle(
        (x1, y1), w, h, linewidth=2, edgecolor="g", facecolor="none"
    )
    ax.add_patch(rect)
    ax.plot(cx, cy, "ro", markersize=8)

    plt.savefig(f"visualizations/{img_name}", bbox_inches="tight", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    dataset = CUB200("data")
    image, bbox, mask = dataset[2].values()
    draw_image(image, bbox, mask, dataset.pixel_mean)
