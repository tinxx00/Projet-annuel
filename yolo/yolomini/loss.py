import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def greedy_match(pred_boxes, gt_boxes):
    """
    Associe chaque bbox ground-truth avec la bbox prédite la plus proche (L1).
    Retourne les indices des correspondances.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], []

    cost = torch.cdist(gt_boxes, pred_boxes, p=1).cpu().detach().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind


def yolo_loss(preds, targets, num_classes=10):
    """
    preds : tuple (bbox_pred, conf_pred, class_pred)
    targets : liste de tensors (N, 5), chaque ligne = [class_id, x, y, w, h]
    """
    bbox_pred, conf_pred, class_pred = preds

    # ✅ Fonctions de perte
    bbox_loss_fn = nn.SmoothL1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    total_loss = 0.0

    for b in range(len(targets)):
        target = targets[b]  # (N, 5)
        if target.ndim != 2 or target.shape[1] != 5:
            continue

        labels = target[:, 0].long()
        boxes = target[:, 1:5].float()

        pred_bbox = bbox_pred[b].reshape(-1, 4)
        pred_conf = conf_pred[b].reshape(-1)
        pred_cls = class_pred[b].reshape(-1, num_classes)

        n = min(len(boxes), pred_bbox.shape[0])
        if n == 0:
            continue

        # 🔄 Greedy Matching (distance L1)
        matched_gt, matched_pred = greedy_match(pred_bbox, boxes)
        matched_gt = torch.tensor(matched_gt, dtype=torch.long, device=pred_bbox.device)
        matched_pred = torch.tensor(matched_pred, dtype=torch.long, device=pred_bbox.device)

        # ✅ Localisation
        loss_bbox = bbox_loss_fn(pred_bbox[matched_pred], boxes[matched_gt])

        # ✅ Confiance (objectness)
        obj_mask = torch.zeros_like(pred_conf)
        obj_mask[matched_pred] = 1.0
        loss_conf = bce_loss(pred_conf, obj_mask)

        # ✅ Classification
        matched_labels = labels[matched_gt]
        pred_logits = pred_cls[matched_pred]
        loss_cls = ce_loss(pred_logits, matched_labels)

        # ✅ Total
        weighted_loss = 10.0 * loss_bbox + 1.0 * loss_conf + 1.0 * loss_cls
        total_loss += weighted_loss

    return total_loss
