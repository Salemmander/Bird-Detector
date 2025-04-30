import torch
from Dataset import CUB200
from YOLOv1 import YOLOv1
from torch.utils.data import DataLoader
import numpy as np


# from the network prediction, extract the bounding boxes with confidences larger than threshold
# pred_box: (batch_size, num_boxes * 5 + num_classes, 7, 7), predicted bounding boxes from the network (see the forward() function)
def extract_detections(pred_box, threshold, num_boxes, num_classes=200):
    boxes_all = []
    class_scores_all = []
    for i in range(num_boxes):
        confidence = pred_box[0, 5 * i + 4].detach().cpu().numpy()
        x, y = np.where(confidence > threshold)
        boxes = pred_box[0, 5 * i : 5 * i + 5, x, y].detach().cpu().numpy().transpose()
        # Get class probabilities for these cells
        class_probs = (
            pred_box[0, 5 * num_boxes : 5 * num_boxes + num_classes, x, y]
            .detach()
            .cpu()
            .numpy()
            .T
        )  # (N_detected, 200)
        class_scores = confidence[y, x][:, None] * class_probs  # (N_detected, 200)
        boxes_all.append(boxes)
        class_scores_all.append(class_scores)

    boxes_all = np.concatenate(boxes_all, axis=0)
    class_scores_all = np.concatenate(class_scores_all, axis=0)
    # convert to (x1, y1, x2, y2)
    boxes = boxes_all.copy()
    boxes[:, 0] = boxes_all[:, 0] - boxes_all[:, 2] * 0.5
    boxes[:, 2] = boxes_all[:, 0] + boxes_all[:, 2] * 0.5
    boxes[:, 1] = boxes_all[:, 1] - boxes_all[:, 3] * 0.5
    boxes[:, 3] = boxes_all[:, 1] + boxes_all[:, 3] * 0.5
    return boxes, class_scores_all


dataset = CUB200("data", "test")

batch_size = 1
num_workers = 0
num_boxes = 2
num_classes = 200

test_dataset = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

epoch_size = len(test_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

threshold = 0.1
results_gt = []
results_pred = []

model = YOLOv1(num_boxes, num_classes, device)
model.to(device)
model.load_state_dict(torch.load("models/YOLOv1.pth"))
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for i, sample in enumerate(test_dataset):
        if i == 0:
            image = sample["image"].to(device)

            output, pred_box = model(image)

            gt_box = sample["gt_box"][0].numpy()
            gt_mask = sample["gt_mask"][0].numpy()
            gt_class = sample["gt_class"][0].numpy()
            x, y = np.where(gt_mask == 1)
            cx = gt_box[0, x, y] * dataset.grid_size + x * dataset.grid_size
            cy = gt_box[1, x, y] * dataset.grid_size + y * dataset.grid_size
            w = gt_box[2, x, y] * dataset.img_size
            h = gt_box[3, x, y] * dataset.img_size
            x1 = cx - (w / 2)
            x2 = cx + (w / 2)
            y1 = cy - (h / 2)
            y2 = cy + (h / 2)
            gt = np.array([x1, y1, x2, y2]).reshape((1, 4))

            cls, _, _ = np.where(gt_class == 1)
            print(f"truth classes: {cls}")


            max_value, pred_class = torch.max(output[:, 10:210, x, y], dim=1)
            print("Predicted Class: ", pred_class, max_value)
            results_gt.append(gt)

            detections, class_scores = extract_detections(
                pred_box, threshold, num_boxes
            )
            # For simplicity, get top class per detection
            pred_classes = np.argmax(class_scores, axis=1)
            print("Detected Classes:", pred_classes)
            results_pred.append((detections, class_scores))

            print(f"Image {i+1}/{epoch_size}, {detections.shape[0]} objects detected")
