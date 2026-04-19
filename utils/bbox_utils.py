import numpy as np

def normalize_bbox(bbox, width, height):
    """
    Normalizes a bounding box [x0, y0, x1, y1] to [0, 1000].
    """
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def denormalize_bbox(bbox, width, height):
    """
    Converts normalized [0, 1000] bbox back to pixel coordinates.
    """
    return [
        int(width * (bbox[0] / 1000)),
        int(height * (bbox[1] / 1000)),
        int(width * (bbox[2] / 1000)),
        int(height * (bbox[3] / 1000)),
    ]

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two boxes.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(area1 + area2 - intersection_area)
    return iou
