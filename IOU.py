import torch


def IOU(ground_truth, bbox_pred):
    xmin = torch.max(ground_truth[0], bbox_pred[0])
    ymin = torch.max(ground_truth[1], bbox_pred[1])
    xmax = torch.min(ground_truth[2], bbox_pred[2])
    ymax = torch.min(ground_truth[3], bbox_pred[3])

    intersection_height = ymax - ymin
    intersection_width = xmax - xmin

    area_of_intersection = intersection_width * intersection_height

    gt_height = ground_truth[3] - ground_truth[1]
    gt_width = ground_truth[2] - ground_truth[0]

    gt_area = gt_width * gt_height

    pred_height = bbox_pred[3] - bbox_pred[1]
    pred_width = bbox_pred[2] - bbox_pred[0]

    pred_area = pred_width * pred_height

    iou = area_of_intersection / (gt_area + pred_area - area_of_intersection)

    return iou


def batch_IOU(ground_truth, bbox_pred):
    # Input shapes: [batch_size, 4], [batch_size, 4] in [x, y, w, h]
    gt_xmin = ground_truth[:, 0]
    gt_ymin = ground_truth[:, 1]
    gt_xmax = gt_xmin + ground_truth[:, 2]
    gt_ymax = gt_ymin + ground_truth[:, 3]
    
    pred_xmin = bbox_pred[:, 0]
    pred_ymin = bbox_pred[:, 1]
    pred_xmax = pred_xmin + bbox_pred[:, 2]
    pred_ymax = pred_ymin + bbox_pred[:, 3]
    
    # Intersection coordinates
    xmin = torch.max(gt_xmin, pred_xmin)
    ymin = torch.max(gt_ymin, pred_ymin)
    xmax = torch.min(gt_xmax, pred_xmax)
    ymax = torch.min(gt_ymax, pred_ymax)
    
    # Intersection area
    intersection = torch.clamp(xmax - xmin, min=0) * torch.clamp(ymax - ymin, min=0)
    
    # Areas of ground truth and prediction
    gt_area = ground_truth[:, 2] * ground_truth[:, 3]
    pred_area = bbox_pred[:, 2] * bbox_pred[:, 3]
    
    # Union area
    union = gt_area + pred_area - intersection
    iou = intersection / (union + 1e-6)  # Add epsilon to avoid division by zero
    return iou