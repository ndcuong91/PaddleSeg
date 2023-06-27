import numpy as np

def calculate_iou(prediction, target, ignore_class=None):
    num_classes_target = np.unique(target)
    class_iou = {}
    for class_id in num_classes_target:
        if class_id != ignore_class:
            pred_mask = (prediction == class_id).numpy()
            target_mask = (target == class_id).numpy()

            intersection = np.sum(np.logical_and(target_mask, pred_mask))
            union = np.sum(np.logical_or(target_mask, pred_mask))

            if (union == 0):
                iou = 0
            else:
                iou = intersection / union

            class_iou[class_id] = iou

    return class_iou