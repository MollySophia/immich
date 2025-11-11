# ---------------------------------------------------------------------
# Copyright 2024-2025 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
import torch
import clip
import numpy as np

def embed_text(text : str):
    """
    Embed text using CLIP model.
    Args:
        text (str or list of str):
            the text to be embedded.
    Returns:
        torch.Tensor:
            the embedded text
    """
    clip_model, _ = clip.load("ViT-B/32", device="cpu")
    if not isinstance(text, list):
        text = [text]
    text_token = clip.tokenize(text).to("cpu")
    txt_feats = [clip_model.encode_text(token).detach() for token in text_token.split(1)]
    txt_feats = torch.cat(txt_feats, dim=0)
    txt_feats /= txt_feats.norm(dim=1, keepdim=True)
    txt_feats = txt_feats.unsqueeze(0)
    return txt_feats


def prepare_embeddings(class_embeddings : torch.Tensor, num_classes: int):
    """
    Prepare class embeddings for inference.
    Args:
        class_embeddings (torch.Tensor):
            class embeddings of shape (num_classes, 512)
        num_classes (int):
            total number of classes
    Returns:
        class_embeddings: numpy.ndarray of shape (num_classes, 512)
    """
    if class_embeddings.shape[1] != num_classes:
        class_embeddings = torch.nn.functional.pad(class_embeddings, (0, 0, 0, num_classes - class_embeddings.shape[1]), mode='constant', value=0)

    return class_embeddings.cpu().numpy().astype(np.float32)


def compute_clip_similarity(image: np.ndarray, text: np.ndarray) -> tuple:
    """
    Compute the similarity between an image and a text using CLIP model.
    Args:
        image (numpy.ndarray):
            the image to be compared.
        text (numpy.ndarray):
            the text to be compared.
    Returns:
        tuple:
            the similarity between the image and the text.
    """
    logit_scale = 100.0
    image_features = torch.from_numpy(image).float()
    text_features = torch.from_numpy(text).float()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    return logits_per_image, logits_per_text