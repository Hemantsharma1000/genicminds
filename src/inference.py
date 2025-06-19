# File: src/inference.py

import argparse
import torch
import torch.nn.functional as F
from PIL import Image

from utils import (
    load_caption_model,
    load_explicit_classifier,
    generate_blip_caption,
    prepare_image_for_classifier,
    LABEL_NAMES,
    DEVICE,
)

def inference(image_path: str,
              caption_model_dir: str = None,
              classifier_path: str = "animal_explicit_classifier.pth",
              threshold: float = 0.5):
    """
    1. Load BLIP (either default or fine-tuned if caption_model_dir given).
    2. Load ResNet50 classifier.
    3. Generate caption, get probabilities, decide on explicit tag, print results.
    """
    # 1) Load models
    processor, caption_model = load_caption_model(caption_model_dir)
    classifier_model = load_explicit_classifier(classifier_path, num_classes=len(LABEL_NAMES))

    # 2) Open & generate caption
    img = Image.open(image_path).convert("RGB")
    caption = generate_blip_caption(processor, caption_model, img)

    # 3) Prepare for classifier & predict
    img_tensor = prepare_image_for_classifier(image_path)  # shape 1×3×224×224
    logits = classifier_model(img_tensor)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()  # list of 5 floats

    # 4) Find top label
    top_idx = int(torch.argmax(torch.tensor(probs)))
    top_label = LABEL_NAMES[top_idx]
    top_prob = probs[top_idx]

    # 5) Prefix if needed
    if top_label != "safe" and top_prob >= threshold:
        final_caption = f"[{top_label.upper()}] {caption}"
    else:
        final_caption = caption

    # 6) Print summary
    print("\n===== Inference Results =====")
    print(f"Image: {image_path}")
    print(f"Generated caption: {caption}")
    print(f"Predicted label: {top_label} (p={top_prob:.2f})")
    print(f"Final output   : {final_caption}\n")
    print("All probabilities:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:20s}: {probs[i]:.3f}")

    return final_caption, {LABEL_NAMES[i]: probs[i] for i in range(len(LABEL_NAMES))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on one image")
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to the image file to run inference on"
    )
    parser.add_argument(
        "--caption_model_dir", type=str, default=None,
        help="If you have a fine-tuned BLIP, point here; otherwise leave blank to use default BLIP"
    )
    parser.add_argument(
        "--classifier_path", type=str, default="animal_explicit_classifier.pth",
        help="Path to the trained explicit classifier .pth file"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold above which to tag a non-safe label"
    )
    args = parser.parse_args()

    inference(
        image_path=args.image_path,
        caption_model_dir=args.caption_model_dir,
        classifier_path=args.classifier_path,
        threshold=args.threshold
    )
