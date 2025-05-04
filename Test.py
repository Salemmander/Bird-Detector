import torch
from Dataset import CUB200
from YOLOv1 import YOLOv1
from torch.utils.data import DataLoader
from torchvision.ops import nms
import numpy as np
from Loss import compute_iou


def box_to_cxcywh(box):
    """Convert box from (x1, y1, x2, y2) to (cx, cy, w, h)."""
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    return np.array([cx, cy, w, h])


# from the network prediction, extract the bounding boxes with confidences larger than threshold
# pred_box: (batch_size, num_boxes * 5 + num_classes, 7, 7), predicted bounding boxes from the network (see the forward() function)
def extract_detections(pred_box, threshold, num_boxes, num_classes=200):
    confidence = pred_box[0, 4].detach().cpu().numpy()
    x, y = np.where(confidence > threshold)
    boxes = pred_box[0, 0:5, x, y].detach().cpu().numpy().transpose()
    # Get class probabilities for these cells
    class_probs = (
        pred_box[0, 5 * num_boxes : 5 * num_boxes + num_classes, x, y]
        .detach()
        .cpu()
        .numpy()
        .T
    )  # (N_detected, 200)
    class_scores = confidence[y, x][:, None] * class_probs  # (N_detected, 200)

    if boxes.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    # convert to (x1, y1, x2, y2)
    boxes_xyxy = boxes[:, :4].copy()
    boxes_xyxy[:, 0] = boxes_xyxy[:, 0] - boxes_xyxy[:, 2] * 0.5
    boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2] * 0.5
    boxes_xyxy[:, 1] = boxes_xyxy[:, 1] - boxes_xyxy[:, 3] * 0.5
    boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3] * 0.5

    max_class_scores = np.max(class_scores, axis=1)
    pred_classes = np.argmax(class_scores, axis=1)
    boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
    scores_tensor = torch.tensor(
        max_class_scores * confidence[x, y], dtype=torch.float32
    )
    keep = nms(boxes_tensor, scores_tensor, iou_threshold=threshold)
    detections = boxes[keep.numpy()]
    class_scores = class_scores[keep.numpy()]
    pred_classes = pred_classes[keep.numpy()]
    return detections, class_scores, pred_classes


dataset = CUB200("data", "test")

batch_size = 1
num_workers = 0
num_boxes = 1
num_classes = 200

test_dataset = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

epoch_size = len(test_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

threshold = 0.3
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
        gt_cxcywh = box_to_cxcywh(gt[0])

        cls, _, _ = np.where(gt_class == 1)

        detections, class_scores, pred_classes = extract_detections(
            pred_box, threshold, num_boxes
        )
        results_gt.append(gt)
        results_pred.append((detections, class_scores))

        if detections.shape[0] == 0:
            print(f"Image {i+1}/{epoch_size}, no objects detected")
            total += 1
            continue

        # Compute IoU for each detection with ground truth
        ious = []
        for det in detections:
            det_cxcywh = box_to_cxcywh(det)
            iou = compute_iou(det_cxcywh, gt_cxcywh)
            ious.append(iou)
        ious = np.array(ious)

        # Select the detection with the highest IoU
        max_iou_idx = np.argmax(ious)
        top_pred_class = pred_classes[max_iou_idx]

        print(
            f"Image {i+1}/{epoch_size}, GT Class: {dataset.class_list[cls]}, Predicted Class: {dataset.class_list[top_pred_class]}, IoU: {ious[max_iou_idx]:.2f}"
        )
        if top_pred_class == cls[0]:
            correct += 1
        total += 1

    print(f"{correct}/{total} correct class predictions. {correct/total * 100}%")
    print(f"Avg IOU: {np.mean(iou)}")
