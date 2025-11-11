# ---------------------------------------------------------------------
# Copyright 2025 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
import numpy as np
import torch
from PIL import Image
import cv2


def tensor2labelim(label_tensor: torch.Tensor, impalette: list, imtype=np.uint8)-> np.ndarray:
    """Converts a label Tensor into an image array (numpy),
    we use a palette to color the label images"""
    if len(label_tensor.shape) == 4:
        _, label_tensor = torch.max(label_tensor.data.cpu(), 1)
    label_numpy = label_tensor[0].cpu().float().detach().numpy()
    label_image = Image.fromarray(label_numpy.astype(np.uint8))
    label_image = label_image.convert("P")
    label_image.putpalette(impalette)
    label_image = label_image.convert("RGB")
    return np.array(label_image).astype(imtype)

def merge_rgb_to_bev(img_rgb : np.ndarray, img_bev: np.ndarray, img_pre: np.ndarray,output_width: int)-> np.ndarray:
    """Merges the RGB image, BEV image and pre-processed image into a single BEV image
    Args:
        img_rgb: RGB image
        img_bev: BEV image
        img_pre: pre-processed image
        output_width: output width of the BEV image

    Returns:
        np.ndarray:
            Merged BEV image
    """
    img_rgb_h, img_rgb_w = img_rgb.shape[:2]
    ratio_rgb = output_width / img_rgb_w
    output_rgb_h = int(ratio_rgb * img_rgb_h)
    ret_img_rgb = cv2.resize(img_rgb, (output_width, output_rgb_h))
    img_bev_h, img_bev_w = img_bev.shape[:2]
    img_ = np.zeros((img_bev_h, 60, 3), dtype=np.uint8)
    img_lidar=np.concatenate((img_bev, img_, img_pre), axis=1)
    out_img = np.zeros((output_rgb_h +img_bev_h, output_width, 3), dtype=np.uint8)
    # Upper: cam --> BEV
    out_img[:output_rgb_h, ...] = ret_img_rgb
    out_img[output_rgb_h:, ...] = img_lidar
    return out_img