# File: video_utils.py

import cv2
import os

def extract_frames(video_path, out_dir, every_nth, start_time=0.0, end_time=None):
    """
    Extract frames from video_path into out_dir, capturing one every `every_nth` frames,
    between `start_time` and `end_time` (seconds).
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / fps if fps else 0.0

    start = max(0.0, start_time)
    end   = min(duration, end_time if end_time is not None else duration)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    saved = 0
    frame_idx = int(start * fps)
    while True:
        ret, frame = cap.read()
        if not ret or (frame_idx / fps) > end:
            break
        if frame_idx % every_nth == 0:
            fname = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames from {start}s to {end}s into {out_dir}")



