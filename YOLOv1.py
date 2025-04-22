import torch
import torch.nn as nn
import math


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

    def forward(self, x):
        pass