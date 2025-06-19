import os

DATA_DIR = "data"
CLASSES = ["safe", "cruelty", "gore", "sexuality_nudity"]

def count_images(folder):
    if not os.path.isdir(folder):
        return -1
    return sum(1 for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png")))

for split in ("train","val"):
    print(f"--- {split.upper()} ---")
    for cls in CLASSES:
        path = os.path.join(DATA_DIR, split, cls)
        cnt = count_images(path)
        status = f"{cnt} image(s)" if cnt>=0 else "MISSING"
        print(f"  {path:40s} → {status}")
    print()
