# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "./")
def nms_centerface(
    boxes : np.ndarray,
    scores : np.ndarray,
    nms_thresh: float) -> list:
    """
    Perform Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes.

    This function takes a set of bounding boxes and their corresponding scores,
    and eliminates the boxes that overlap significantly with higher scoring boxes.

    Args:
        boxes (np.ndarray): An array of shape (N, 4) representing bounding boxes,
                            where each box is defined as (x1, y1, x2, y2).
        scores (np.ndarray): An array of shape (N,) representing the scores for each bounding box.
        nms_thresh (float): Threshold for determining whether two boxes overlap too much.

    Returns:
        list: A list of indices of the bounding boxes that are kept after NMS.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    suppressed = np.zeros((num_detections,), dtype=np.bool_)

    keep = []
    for _i in range(num_detections):
        i = order[_i]
        if suppressed[i]:
            continue
        keep.append(i)

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, num_detections):
            j = order[_j]
            if suppressed[j]:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_thresh:
                suppressed[j] = True

    return keep

def postprocess(
    heatmap : np.ndarray,
    lms : np.ndarray,
    offset : np.ndarray,
    scale : float,
    threshold : float,
    img_h_new : int,
    img_w_new : int,
    scale_w : float,
    scale_h : float,
    det_scale : float
    ) -> tuple:
    """
    Post-process the outputs from the detection model to decode bounding boxes and 
    landmark points, and scale them back to the original image size.

    Args:
        heatmap (numpy.ndarray): The heatmap output from the model indicating the presence of faces.
        lms (numpy.ndarray): Initial landmark outputs from the model for detected faces.
        offset (numpy.ndarray): Offset values used for adjusting bounding box locations.
        scale (numpy.ndarray): Scale factors for bounding boxes.
        threshold (float): Confidence threshold to filter out weak detections.
        img_h_new (int): Height of the new input image after processing.
        img_w_new (int): Width of the new input image after processing.
        scale_w (float): Scale factor for width adjustment when resizing bounding boxes.
        scale_h (float): Scale factor for height adjustment when resizing bounding boxes.
        det_scale (float): Additional scale factor for the final bounding box size.

    Returns:
        tuple: A tuple containing:
            - dets (numpy.ndarray): The decoded and scaled bounding boxes of detected faces.
            - lms (numpy.ndarray): The landmarks associated with the detected faces, scaled to the original image size.
    """

    dets, lms = decode(heatmap, scale, offset, lms, (img_h_new, img_w_new), threshold,det_scale)

    if len(dets) > 0:
        dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / scale_w, dets[:, 1:4:2] / scale_h
        lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / scale_w, lms[:, 1:10:2] / scale_h

    return dets, lms


def decode(
    heatmap : np.ndarray,
    scale : np.ndarray,
    offset : np.ndarray,
    landmark : np.ndarray,
    size : tuple,
    threshold : float,
    det_scale : float
    ) -> tuple:
    """
    Decode the output from the heatmap, scale, and offset to get bounding boxes
    and landmark points for detected objects.

    Args:
        heatmap (numpy.ndarray): The heatmap output from the detector, indicating the presence of objects.
        scale (numpy.ndarray): The scale information for bounding box dimensions.
        offset (numpy.ndarray): The offset information for bounding box coordinates.
        landmark (numpy.ndarray): The landmark positions for detected objects.
        size (tuple): The size of the original image as (height, width).
        threshold (float): Confidence threshold for filtering out weak detections.
        det_scale (float): Scale factor for resizing the bounding boxes and landmarks.

    Returns:
        tuple: A tuple containing:
            - boxes (numpy.ndarray): The bounding boxes of detected objects in the format 
                                    [x1, y1, x2, y2, score].
            - lms (numpy.ndarray): The landmark positions associated with the detected objects.
    """
    heatmap = np.squeeze(heatmap)
    scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
    offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
    c0, c1 = np.where(heatmap > threshold)
    boxes, lms = [], []

    if len(c0) > 0:
        for i in range(len(c0)):
            s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
            o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
            s = heatmap[c0[i], c1[i]]
            x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
            x1, y1 = min(x1, size[1]), min(y1, size[0])
            boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])

            lm = []
            for j in range(5):
                lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
            lms.append(lm)
        boxes = np.asarray(boxes, dtype=np.float32)
        keep = nms_centerface(boxes[:, :4], boxes[:, 4], 0.3)
        boxes = boxes[keep, :]

        lms = np.asarray(lms, dtype=np.float32)
        lms = lms[keep, :]

    boxes = np.vstack(boxes) / det_scale
    lms = np.vstack(lms) / det_scale
    return boxes, lms

def area_of(left_top : np.ndarray, right_bottom : np.ndarray) -> np.ndarray:
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0 : np.ndarray, boxes1 : np.ndarray, eps : float = 1e-5) -> np.ndarray:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores : np.ndarray, iou_threshold : float, top_k : int = -1, candidate_size : int = 200) -> np.ndarray:
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


def scale_box(box : np.ndarray) -> np.ndarray:
    """
    Scale the bounding box to square
    Args:
        box (N, 4): bounding box in corner-form
    Returns:
        bboxes (N, 4): scaled bounding box in corner-form
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width)/2)
    dy = int((maximum - height)/2)
    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes


def postprocess_rfb(width : int, height : int, confidences : np.ndarray, boxes : np.ndarray, prob_threshold : float, iou_threshold : float=0.5, top_k : int = -1)-> tuple:
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]
