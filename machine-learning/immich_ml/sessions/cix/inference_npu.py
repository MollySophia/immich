# ---------------------------------------------------------------------
# Copyright 2025 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""
This is the script of cix noe umd api for inference over npu.
"""
import os
import argparse
import clip
import numpy as np
import torch
import sys

from utils.tools import get_file_list
from utils.image_process import preprocess_image_deeplabv3
from utils.text_process import compute_clip_similarity
from utils.NOE_Engine import EngineInfer
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--txt_model_path",
        default="clip_txt.cix",
        help="path to the model file",
    )
    parser.add_argument(
        "--image_model_path",
        default="clip_visual.cix",
        help="path to the model file",
    )
    parser.add_argument(
        "--image_path",
        default="test_data",
        help="path to the model file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    texts = ["a person", "a dog", "a bird"]
    txt_model = EngineInfer(args.txt_model_path)
    img_model = EngineInfer(args.image_model_path)
    images_list = get_file_list(args.image_path)
    text_features = np.empty((len(texts),512))
    for i in range(len(texts)):
        message = texts[i]
        text = clip.tokenize(message).numpy().astype(np.int32)
        text_feature = txt_model.forward([text.astype(np.int32)])[0]
        text_feature = np.reshape(text_feature, (1,512))
        text_features[i] = text_feature
    for image_path in images_list:
        image = preprocess_image_deeplabv3(image_path,mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], target_size=(224, 224))
        img_feature = img_model.forward([image])[0]
        img_feature = np.reshape(img_feature, (1,512))
        logits_per_image, logits_per_text = compute_clip_similarity(img_feature, text_features)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        print(probs)
        print(f"{image_path}, max similarity: {texts[np.argmax(probs)]}")
    txt_model.clean()
    img_model.clean()