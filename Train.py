import os
import matplotlib.pyplot as plt
import numpy as np
from Loss import compute_loss
from YOLOv1 import YOLOv1
from Dataset import CUB200
import torch
from torch.utils.data import DataLoader


def plot_losses(losses, filename="visualizations/train_loss.pdf"):
    num_epoches = losses.shape[0]
    l = np.mean(losses, axis=1)
    plt.plot(range(1, num_epoches + 1), l, marker="o", alpha=0.5, ms=4)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.savefig(filename, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    num_epochs = 50
    batch_size = 16
    learning_rate = 0.0001
    num_workers = 4

    dataset = CUB200("data", "train")

    train_dataset = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    epoch_size = len(train_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    num_classes = 200
    num_boxes = 1
    model = YOLOv1(num_boxes, num_classes, device)
    model.to(device)
    image_size = model.image_size
    grid_size = model.grid_size
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    output_dir = "models"
    print(f"Output will be saved to {output_dir}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    losses = np.zeros((num_epochs, epoch_size), dtype=np.float32)

    for epoch in range(num_epochs):
        for i, sample in enumerate(train_dataset):
            image = sample["image"].to(device)
            gt_box = sample["gt_box"].to(device)
            gt_mask = sample["gt_mask"].to(device)
            gt_class = sample["gt_class"].to(device)

            output, pred_box = model(image)

            loss = compute_loss(
                output,
                pred_box,
                gt_box,
                gt_mask,
                gt_class,
                num_boxes,
                num_classes,
                grid_size,
                image_size,
                device,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[epoch, i] = loss

            print(
                f"Epoch {epoch+1}/{num_epochs}, Iteration {i+1}/{epoch_size}, lr {learning_rate}, Loss {loss}"
            )

    state = model.state_dict()
    filename = "YOLOv1.pth"
    torch.save(state, os.path.join(output_dir, filename))
    print(filename)
    plot_losses(losses)
