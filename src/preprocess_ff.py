"""
Preprocessing script for FaceForensics++ → F3Net (yyk-wew repo)
================================================================
Reads raw .mp4 videos from the FF++ directory structure,
extracts 270 frames per video, detects faces with MTCNN,
applies a 1.3x bounding-box crop, and saves JPEGs into
the exact tree expected by utils.py / FFDataset:

    <output_root>/
        train/real/<vid_id>/frame0.jpg ...
        train/fake/Deepfakes/<vid_id>/frame0.jpg ...
        train/fake/Face2Face/...
        train/fake/FaceSwap/...
        train/fake/NeuralTextures/...
        valid/...
        test/...

"""

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# ─────────────────────────────────────────────
# CONFIG 
# ─────────────────────────────────────────────

# Root of the FF++ download (contains original_sequences/ and manipulated_sequences/)
FF_ROOT = os.path.expanduser("~/F3Net/data/faceforensics")

# Where to write the preprocessed dataset
OUTPUT_ROOT = os.path.expanduser("~/F3Net/data/FF++_preprocessed")

# Compression level downloaded ("c40" = low quality, "c23" = high quality, "raw" = no compression)
COMPRESSION = "c40"

# Number of frames to extract per video (paper uses 270)
FRAMES_PER_VIDEO = 270

# Face crop scale factor (paper: 1.3×)
CROP_SCALE = 1.3

# Output image size fed to the model
IMG_SIZE = 299

# FF++ split files — 720 train / 140 valid / 140 test  (standard FF++ split)
TRAIN_IDS = [f"{i:03d}" for i in range(720)]
VALID_IDS  = [f"{i:03d}" for i in range(720, 860)]
TEST_IDS   = [f"{i:03d}" for i in range(860, 1000)]

SPLIT_MAP = {
    "train": TRAIN_IDS,
    "valid": VALID_IDS,
    "test":  TEST_IDS,
}

# Manipulation methods (folder names inside FF++)
FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

# ─────────────────────────────────────────────
# MTCNN face detector (GPU if available)
# ─────────────────────────────────────────────
from facenet_pytorch import MTCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# keep_all=False, only the most confident face per frame
detector = MTCNN(
    keep_all=False,
    device=device,
    select_largest=True,   # pick the biggest face if multiple detected
    post_process=False,
)


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def get_video_path_real(ff_root: str, vid_id: str, compression: str) -> str:
    """Returns the path to a real video file."""
    # FF++ layout: original_sequences/youtube/c40/videos/000.mp4
    # (some downloads use 'actors' too, but we target youtube originals)
    p = os.path.join(ff_root, "original_sequences", "youtube", compression, "videos", f"{vid_id}.mp4")
    if not os.path.exists(p):
        # fallback: sometimes the extension differs or there's no subfolder
        p = os.path.join(ff_root, "original_sequences", "youtube", compression, "videos", f"{vid_id}.avi")
    return p


def get_video_path_fake(ff_root: str, method: str, vid_id: str, compression: str) -> list:
    """
    Returns a list of candidate video paths for a fake video.
    A fake video for source 000 can be named 000_XXX.mp4 (any target).
    We return ALL matching videos (one source can have multiple targets).
    """
    base = os.path.join(ff_root, "manipulated_sequences", method, compression, "videos")
    if not os.path.isdir(base):
        return []
    matches = [
        os.path.join(base, f)
        for f in os.listdir(base)
        if f.startswith(f"{vid_id}_") and f.endswith((".mp4", ".avi"))
    ]
    return matches


def sample_frames(video_path: str, n_frames: int) -> list:
    """
    Opens a video and returns `n_frames` PIL Images sampled at regular intervals.
    Returns an empty list if the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num=min(n_frames, total), dtype=int)
    frames = []
    prev_idx = -1

    for idx in indices:
        if idx != prev_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        prev_idx = idx

    cap.release()
    return frames


def crop_face(pil_img: Image.Image, scale: float = 1.3, out_size: int = 299):
    """
    Detects the primary face in `pil_img`, expands the bounding box by `scale`,
    crops, and resizes to `out_size × out_size`.
    Returns None if no face is detected.
    """
    boxes, _ = detector.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return None

    x1, y1, x2, y2 = boxes[0]  # most confident / largest face
    W, H = pil_img.size

    # Expand bounding box by scale factor around its center
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale

    nx1 = max(0, int(cx - bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    nx2 = min(W, int(cx + bw / 2))
    ny2 = min(H, int(cy + bh / 2))

    if nx2 <= nx1 or ny2 <= ny1:
        return None

    face = pil_img.crop((nx1, ny1, nx2, ny2))
    face = face.resize((out_size, out_size), Image.LANCZOS)
    return face


def process_video(video_path: str, out_dir: str, n_frames: int, crop_scale: float, img_size: int):
    """
    Full pipeline for one video: sample frames → detect face → crop → save.
    Skips frames where no face is found.
    Returns the number of saved frames.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Skip if already processed (allows resuming)
    existing = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
    if len(existing) >= n_frames // 2:   # at least half done → skip
        return len(existing)

    frames = sample_frames(video_path, n_frames)
    saved = 0

    for i, frame in enumerate(frames):
        face = crop_face(frame, scale=crop_scale, out_size=img_size)
        if face is None:
            continue
        out_path = os.path.join(out_dir, f"frame{saved}.jpg")
        face.save(out_path, quality=95)
        saved += 1

    return saved


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main():
    total_videos = 0
    total_frames = 0
    total_skipped = 0

    for split, vid_ids in SPLIT_MAP.items():
        print(f"\n{'='*60}")
        print(f"  Processing split: {split.upper()}  ({len(vid_ids)} base videos)")
        print(f"{'='*60}")

        # ── REAL ──────────────────────────────────────────────────
        print(f"\n  [REAL]")
        for vid_id in tqdm(vid_ids, desc=f"{split}/real"):
            video_path = get_video_path_real(FF_ROOT, vid_id, COMPRESSION)
            if not os.path.exists(video_path):
                tqdm.write(f"    [SKIP] Missing real video: {video_path}")
                total_skipped += 1
                continue

            out_dir = os.path.join(OUTPUT_ROOT, split, "real", vid_id)
            saved = process_video(video_path, out_dir, FRAMES_PER_VIDEO, CROP_SCALE, IMG_SIZE)
            total_frames += saved
            total_videos += 1

        # ── FAKE ──────────────────────────────────────────────────
        for method in FAKE_METHODS:
            print(f"\n  [FAKE / {method}]")
            for vid_id in tqdm(vid_ids, desc=f"{split}/fake/{method}"):
                video_paths = get_video_path_fake(FF_ROOT, method, vid_id, COMPRESSION)

                if not video_paths:
                    # Try the symmetric pair (target_source) as fallback
                    tqdm.write(f"    [SKIP] No fake video found for {method}/{vid_id}_*")
                    total_skipped += 1
                    continue

                for vp in video_paths:
                    # Folder name = stem of video filename, e.g. "000_167"
                    folder_name = os.path.splitext(os.path.basename(vp))[0]
                    out_dir = os.path.join(OUTPUT_ROOT, split, "fake", method, folder_name)
                    saved = process_video(vp, out_dir, FRAMES_PER_VIDEO, CROP_SCALE, IMG_SIZE)
                    total_frames += saved
                    total_videos += 1

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Videos processed : {total_videos}")
    print(f"  Total frames saved: {total_frames}")
    print(f"  Videos skipped   : {total_skipped}")
    print(f"  Output directory : {OUTPUT_ROOT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
