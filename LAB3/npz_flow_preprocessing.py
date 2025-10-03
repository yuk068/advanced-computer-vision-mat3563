import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================
PROCESSED_DIR = "data/UCF50_processed/flows"
NPZ_DIR = "data/UCF50_npz"

IMG_SIZE = None
TEST_RATIO = 0.2
RANDOM_SEED = 42

# ==============================
# UTILS
# ==============================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_flow_frames(flow_dir, size=IMG_SIZE):
    """
    Load optical flow frames from folder as numpy array [T, H, W, 2].
    Each frame has 2 images: _flowx and _flowy.
    """
    files = sorted([f for f in os.listdir(flow_dir) if f.endswith("_flowx.jpg")])
    frames = []

    for fx in files:
        fy = fx.replace("_flowx", "_flowy")
        fx_path = os.path.join(flow_dir, fx)
        fy_path = os.path.join(flow_dir, fy)
        if not os.path.exists(fx_path) or not os.path.exists(fy_path):
            continue

        # Load grayscale (single channel)
        img_x = cv2.imread(fx_path, cv2.IMREAD_GRAYSCALE)
        img_y = cv2.imread(fy_path, cv2.IMREAD_GRAYSCALE)

        if size is not None:
            img_x = cv2.resize(img_x, size)
            img_y = cv2.resize(img_y, size)

        # Stack as H x W x 2
        frame = np.stack([img_x, img_y], axis=-1)
        frames.append(frame)

    if len(frames) == 0:
        return None

    # Optional: pad first frame with zeros if needed
    frames = np.array(frames, dtype=np.uint8)
    pad_frame = np.zeros_like(frames[0], dtype=np.uint8)
    frames = np.concatenate([pad_frame[None], frames], axis=0)

    return frames

# ==============================
# MAIN
# ==============================
def main():
    ensure_dir(NPZ_DIR)

    classes = sorted(os.listdir(PROCESSED_DIR))
    print(f"📂 Found {len(classes)} classes: {classes}")

    X, y, video_ids = [], [], []

    for label, class_name in enumerate(classes):
        class_dir = os.path.join(PROCESSED_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        videos = sorted(os.listdir(class_dir))
        print(f"➡️ Processing class {class_name} ({len(videos)} videos)")

        for vid in videos:
            vid_dir = os.path.join(class_dir, vid)
            frames = load_flow_frames(vid_dir)
            if frames is None:
                print(f"⚠️ Skipping empty {vid}")
                continue

            X.append(frames)
            y.append(label)
            video_ids.append(f"{class_name}/{vid}")

        print(f"✅ Done class {class_name}")

    X = np.array(X, dtype=object)
    y = np.array(y)
    print(f"📊 Loaded {len(X)} videos in total.")

    # Train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y
    )

    X_train, y_train, ids_train = X[train_idx], y[train_idx], [video_ids[i] for i in train_idx]
    X_test, y_test, ids_test = X[test_idx], y[test_idx], [video_ids[i] for i in test_idx]

    print(f"✂️ Split into {len(X_train)} train and {len(X_test)} test videos.")

    # Save npz
    train_path = os.path.join(NPZ_DIR, "train_flow.npz")
    test_path = os.path.join(NPZ_DIR, "test_flow.npz")

    np.savez_compressed(train_path, X=X_train, y=y_train, ids=ids_train)
    np.savez_compressed(test_path, X=X_test, y=y_test, ids=ids_test)

    print(f"💾 Saved train set → {train_path}")
    print(f"💾 Saved test set → {test_path}")


if __name__ == "__main__":
    main()