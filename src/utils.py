# File: src/utils.py

import os
import torch
from torch import nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torchvision.transforms as T
from torchvision import models

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Labels ─────────────────────────────────────────────────────────────────────
LABEL_NAMES = [
    "safe",
    "animal_cruelty",
    "animal_gore",
    "animal_sexuality_nudity",
]

# ── BLIP Captioning ─────────────────────────────────────────────────────────────

def load_caption_model(model_dir=None):
    processor = BlipProcessor.from_pretrained(
        model_dir or "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        model_dir or "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    return processor, model


def generate_blip_caption(processor, model, pil_img: Image.Image) -> str:
    inputs = processor(pil_img, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ── Explicit Classifier ─────────────────────────────────────────────────────────

def build_explicit_classifier(num_classes: int):
    """Instantiate an EfficientNet-B3 backbone with custom classifier head."""
    backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    in_feats = backbone.classifier[1].in_features
    backbone.classifier[1] = nn.Linear(in_feats, num_classes)
    return backbone


def prepare_image_for_classifier(image_path: str):
    """
    Load image from path and apply transforms matching training.
    """
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    return tensor


def load_explicit_classifier(model_path: str, num_classes: int):
    """
    Build the classifier and load saved weights.
    """
    model = build_explicit_classifier(num_classes=num_classes).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    if 'state_dict' in state:
        state = state['state_dict']
    # Handle keys with or without module prefix
    new_state = {}
    for k, v in state.items():
        key = k.replace('module.', '')
        new_state[key] = v
    model.load_state_dict(new_state)
    model.eval()
    return model


