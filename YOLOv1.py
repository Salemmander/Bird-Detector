import torch
import torch.nn as nn
import math
import numpy as np


class YOLOv1(nn.Module):
    def __init__(self, num_boxes, num_classes, device):
        super(YOLOv1, self).__init__()

        # Bounding Boxes per cell
        self.num_boxes = num_boxes
        # Number of classes for detection
        self.num_classes = num_classes
        # GPU/CPU
        self.device = device
        # height/width of images (after resizing/not original size)
        self.image_size = 448
        # height/width of each grid
        self.grid_size = 64
        # Creating the Networks layers
        self.network = self.create_modules()
        # Initialize Weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def create_modules(self):
        modules = nn.Sequential()
        in_channels = 3
        out_channels = 16
        for i in range(6):
            i += 1
            modules.add_module(f"Conv{i}", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)) # fmt: skip
            modules.add_module(f"Relu{i}", nn.ReLU())
            modules.add_module(f"MaxPool{i}", nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels  # Ends at 512
            out_channels *= 2  # Ends at 1024
        for i in range(3):
            i += 7
            modules.add_module(f"Conv{i}", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)) # fmt: skip
            modules.add_module(f"Relu{i}", nn.ReLU())
            in_channels = out_channels
        # Should result as 50176 (1024 * 7 * 7)
        modules.add_module("Flatten", nn.Flatten())
        modules.add_module("FC1", nn.Linear(50176, 256))
        modules.add_module("FC2", nn.Linear(256, 256))

        output = 7 * 7 * (5 * self.num_boxes + self.num_classes)

        modules.add_module("FC Output", nn.Linear(256, output))
        modules.add_module("Sigmoid", nn.Sigmoid())

        return modules

    def transform_predictions(self, output):
        batch_size = output.shape[0]
        x = torch.linspace(0, 384, steps=7, device=self.device)
        y = torch.linspace(0, 384, steps=7, device=self.device)
        corner_x, corner_y = torch.meshgrid(x, y, indexing="xy")
        corner_x = torch.unsqueeze(corner_x, dim=0)
        corner_y = torch.unsqueeze(corner_y, dim=0)
        corners = torch.cat((corner_x, corner_y), dim=0)
        corners = corners.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        pred_box = output.clone()
        # for each bounding box
        for i in range(self.num_boxes):
            # x and y
            pred_box[:, i * 5, :, :] = (
                corners[:, 0, :, :] + output[:, i * 5, :, :] * self.grid_size
            )
            pred_box[:, i * 5 + 1, :, :] = (
                corners[:, 1, :, :] + output[:, i * 5 + 1, :, :] * self.grid_size
            )
            # w and h
            pred_box[:, i * 5 + 2, :, :] = output[:, i * 5 + 2, :, :] * self.image_size
            pred_box[:, i * 5 + 3, :, :] = output[:, i * 5 + 3, :, :] * self.image_size
        return pred_box

    def forward(self, x):
        output = self.network(x).reshape(
            -1, self.num_boxes * 5 + self.num_classes, 7, 7
        )

        pred_box = self.transform_predictions(output)
        return output, pred_box


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = YOLOv1(num_boxes=2, num_classes=1, device=device)
    network.to(device)
    print(network)

    image = np.random.uniform(-0.5, 0.5, size=(1, 3, 448, 448)).astype(np.float32)
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.to(device)
    print("input image:", image_tensor.shape)

    output, pred_box = network(image_tensor)
    print("network output:", output.shape, pred_box.shape)
