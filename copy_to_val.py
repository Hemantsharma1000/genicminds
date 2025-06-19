# copy_to_val.py
import os, shutil
from glob import glob

DATA = "data"
CLASSES = ["safe","cruelty","gore","sexuality_nudity"]
N = 5  # how many per class to copy

for cls in CLASSES:
    src = os.path.join(DATA,"train",cls)
    dst = os.path.join(DATA,"val",cls)
    os.makedirs(dst, exist_ok=True)
    imgs = glob(os.path.join(src,"*.jpg")) + glob(os.path.join(src,"*.png"))
    for img in imgs[:N]:
        shutil.copy(img, dst)
    print(f"Copied {min(len(imgs),N)} images to {dst}")
