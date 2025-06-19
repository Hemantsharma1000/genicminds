# File: app_streamlit.py

import os
import random
import tempfile
import torch
import cv2
import numpy as np
import streamlit as st
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
    classifier_model = load_explicit_classifier("animal_explicit_classifier.pth", num_classes=len(LABEL_NAMES))

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

    return (processor, caption_model, classifier_model,
            clip_processor, clip_model, text_embeds)

processor, caption_model, classifier_model, clip_processor, clip_model, text_embeds = load_models()

def zero_shot_clip(pil_img):
    img_inputs = clip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        img_embeds = clip_model.get_image_features(**img_inputs)
    img_embeds /= img_embeds.norm(dim=-1, keepdim=True)
    sims = (img_embeds @ text_embeds.T).squeeze(0)
    return sims.softmax(dim=-1).cpu().tolist()

# ── 2. Classification logic ────────────────────────────────────────────────────
def classify_image(pil_img: Image.Image) -> str:
    caption = generate_blip_caption(processor, caption_model, pil_img)
    cap_low = caption.lower()

    # Keyword rules
    gore_kw = {"blood","meat","flesh","dead","carcass","wound","gore","kill","predator","butcher","slaughter"}
    cruelty_kw = {"chain","cage","tie","torture","abuse","hit","beat","kick","whip","punish","restraint"}

    if any(k in cap_low for k in gore_kw):
        display = "animal_gore"
    elif any(k in cap_low for k in cruelty_kw):
        display = "animal_cruelty"
    else:
        # CNN+CLIP fusion
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil_img.save(tmp.name)
            tensor = prepare_image_for_classifier(tmp.name)
        os.unlink(tmp.name)

        probs_cls = torch.softmax(classifier_model(tensor), dim=1).squeeze(0)
        probs_clip = torch.tensor(zero_shot_clip(pil_img), device=DEVICE)
        probs = 0.6 * probs_cls + 0.4 * probs_clip

        arr = np.array(pil_img.resize((224,224)))
        r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        if float(((r>150)&(r>1.2*g)&(r>1.2*b)).mean()) > 0.02:
            original = "animal_gore"
        else:
            original = LABEL_NAMES[int(probs.argmax())]

        swap = {
            "safe":"animal_cruelty",
            "animal_cruelty":"animal_gore",
            "animal_gore":"safe",
            "animal_sexuality_nudity":"animal_sexuality_nudity",
        }
        display = swap.get(original, original)

    return f"[ANIMAL_{display.upper()}] {caption}"

def classify_video_bytes(video_bytes) -> str:
    # write to temp file
    tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_vid.write(video_bytes.read())
    tmp_vid.flush()
    tmp_vid.close()

    cap = cv2.VideoCapture(tmp_vid.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = cnt / fps if fps else 0.0
    cap.release()

    end = min(30.0, dur)
    from video_utils import extract_frames
    outdir = tempfile.mkdtemp()
    extract_frames(tmp_vid.name, outdir, int(fps), 0.0, end)

    jpgs = sorted(f for f in os.listdir(outdir) if f.endswith(".jpg"))
    batches = [jpgs[i:i+5] for i in range(0,len(jpgs),5)]
    picks = [random.choice(b) for b in batches if b]

    results = []
    for name in picks:
        img = Image.open(os.path.join(outdir,name)).convert("RGB")
        results.append(f"{name}: {classify_image(img)}")

    # cleanup
    for f in os.listdir(outdir):
        os.unlink(os.path.join(outdir,f))
    os.rmdir(outdir)
    os.unlink(tmp_vid.name)

    return "\n".join(results)

# ── 3. Streamlit UI ───────────────────────────────────────────────────────────
st.title("🦁 Animal Captioner & Classifier")

mode = st.sidebar.radio("Mode", ["Image", "Video"])
if mode == "Image":
    img = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if img:
        pil = Image.open(img).convert("RGB")
        st.image(pil, use_column_width=True)
        caption = classify_image(pil)
        st.markdown(f"**Caption:** {caption}")
else:
    vid = st.file_uploader("Upload a video (<=30s)", type=["mp4","mov","avi"])
    if vid:
        st.video(vid)
        st.markdown("**Results:**")
        res = classify_video_bytes(vid)
        st.text(res)



