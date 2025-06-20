# File: app.py

import os
import random
import tempfile
import torch
import numpy as np
import streamlit as st
import cv2
from PIL import Image

from src.utils import (
    load_caption_model,
    load_explicit_classifier,
    generate_blip_caption,
    prepare_image_for_classifier,
    LABEL_NAMES,
    DEVICE,
)
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="Animal Captioner & Classifier", layout="wide")

# ── 1. Load models once ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    processor, caption_model = load_caption_model()
    classifier_model = load_explicit_classifier(
        "animal_explicit_classifier.pth", num_classes=len(LABEL_NAMES)
    )

    prompts = [
        "a safe animal photo",
        "animal cruelty scene",
        "animal gore with blood and wounds",
        "animal sexual activity"
    ]
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

    text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(**text_inputs)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    return (
        processor,
        caption_model,
        classifier_model,
        clip_processor,
        clip_model,
        text_embeds,
    )

processor, caption_model, classifier_model, clip_processor, clip_model, text_embeds = load_models()

def zero_shot_clip(pil_img: Image.Image):
    img_inputs = clip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        img_embeds = clip_model.get_image_features(**img_inputs)
    img_embeds /= img_embeds.norm(dim=-1, keepdim=True)
    sims = (img_embeds @ text_embeds.T).squeeze(0)
    return sims.softmax(dim=-1).cpu().tolist()

# ── 2. Image classification logic ─────────────────────────────────────────────
def classify_image(pil_img: Image.Image) -> str:
    caption = generate_blip_caption(processor, caption_model, pil_img)
    cap_low = caption.lower()

    gore_kw = {
        "blood", "meat", "flesh", "dead", "carcass",
        "wound", "gore", "kill", "predator", "butcher", "slaughter"
    }
    cruelty_kw = {
        "chain", "cage", "tie", "torture", "abuse",
        "hit", "beat", "kick", "whip", "punish", "restraint"
    }

    if any(k in cap_low for k in gore_kw):
        display = "animal_gore"
    elif any(k in cap_low for k in cruelty_kw):
        display = "animal_cruelty"
    else:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil_img.save(tmp.name)
            tensor = prepare_image_for_classifier(tmp.name)
        os.unlink(tmp.name)

        probs_cls = torch.softmax(classifier_model(tensor), dim=1).squeeze(0)
        probs_clip = torch.tensor(zero_shot_clip(pil_img), device=DEVICE)
        probs = 0.6 * probs_cls + 0.4 * probs_clip

        arr = np.array(pil_img.resize((224, 224)))
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        if float(((r > 150) & (r > 1.2 * g) & (r > 1.2 * b)).mean()) > 0.02:
            original = "animal_gore"
        else:
            original = LABEL_NAMES[int(probs.argmax())]

        swap = {
            "safe": "animal_cruelty",
            "animal_cruelty": "animal_gore",
            "animal_gore": "safe",
            "animal_sexuality_nudity": "animal_sexuality_nudity",
        }
        display = swap.get(original, original)

    return f"[ANIMAL_{display.upper()}] {caption}"

# ── 3. Video frame extraction with OpenCV ────────────────────────────────────
def extract_frames_with_cv2(video_path, outdir, target_fps=5, max_duration=30.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video file")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / vid_fps
    end_time = min(duration, max_duration)

    # sample at target_fps over [0, end_time]
    intervals = np.linspace(0, end_time, int(target_fps * end_time), endpoint=False)
    for idx, t in enumerate(intervals):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        # convert BGR→RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(outdir, f"frame_{idx:04d}.jpg"))

    cap.release()

def classify_video_bytes(video_bytes) -> str:
    tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_vid.write(video_bytes.read())
    tmp_vid.flush()
    tmp_vid.close()

    outdir = tempfile.mkdtemp()
    # extract frames at 2 FPS up to 30 seconds
    extract_frames_with_cv2(tmp_vid.name, outdir, target_fps=2, max_duration=30.0)

    frames = sorted(f for f in os.listdir(outdir) if f.endswith(".jpg"))
    # group into chunks of 5
    batches = [frames[i : i + 5] for i in range(0, len(frames), 5)]
    picks = [random.choice(b) for b in batches if b]

    results = []
    for name in picks:
        img = Image.open(os.path.join(outdir, name)).convert("RGB")
        results.append(f"{name}: {classify_image(img)}")

    # cleanup
    for filename in os.listdir(outdir):
        os.unlink(os.path.join(outdir, filename))
    os.rmdir(outdir)
    os.unlink(tmp_vid.name)

    return "\n".join(results)

# ── 4. Streamlit UI ───────────────────────────────────────────────────────────
st.title("🦁 Animal Captioner & Classifier")

mode = st.sidebar.radio("Mode", ["Image", "Video"])
if mode == "Image":
    img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img:
        pil = Image.open(img).convert("RGB")
        st.image(pil, use_column_width=True)
        st.markdown(f"**Caption:** {classify_image(pil)}")
else:
    vid = st.file_uploader("Upload a video (<=30s)", type=["mp4", "mov", "avi"])
    if vid:
        st.video(vid)
        st.markdown("**Results:**")
        st.text(classify_video_bytes(vid))






