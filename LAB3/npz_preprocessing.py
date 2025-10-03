import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================
PROCESSED_DIR = "data/UCF50_processed"
NPZ_DIR = "data/UCF50_npz"

IMG_SIZE = None       # None → keep original size (your preprocessing already made 224x224)
TEST_RATIO = 0.2
RANDOM_SEED = 42

# ==============================
# UTILS
# ==============================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_video_frames(frame_dir, size=IMG_SIZE):
    """Load all frames from one video folder as numpy array [T, H, W, C]."""
    frames = []
    files = sorted(os.listdir(frame_dir))
    for file in files:
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(frame_dir, file))
        if size is not None:
            img = cv2.resize(img, size)   # keep option if you ever want smaller
        frames.append(img)
    if len(frames) == 0:
        return None
    return np.array(frames, dtype=np.uint8)

# ==============================
# MAIN
# ==============================
def main():
    ensure_dir(NPZ_DIR)

    frames_root = os.path.join(PROCESSED_DIR, "frames")
    classes = sorted(os.listdir(frames_root))
    print(f"📂 Found {len(classes)} classes: {classes}")

    X, y, video_ids = [], [], []

    for label, class_name in enumerate(classes):
        class_dir = os.path.join(frames_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        videos = sorted(os.listdir(class_dir))
        print(f"➡️ Processing class {class_name} ({len(videos)} videos)")

        for vid in videos:
            vid_dir = os.path.join(class_dir, vid)
            frames = load_video_frames(vid_dir)
            if frames is None:
                print(f"⚠️ Skipping empty {vid}")
                continue

            X.append(frames)
            y.append(label)
            video_ids.append(f"{class_name}/{vid}")

        print(f"✅ Done class {class_name}")

    # Convert to numpy arrays (object dtype because videos have variable length)
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
    train_path = os.path.join(NPZ_DIR, "train.npz")
    test_path = os.path.join(NPZ_DIR, "test.npz")

    np.savez_compressed(train_path, X=X_train, y=y_train, ids=ids_train)
    np.savez_compressed(test_path, X=X_test, y=y_test, ids=ids_test)

    print(f"💾 Saved train set → {train_path}")
    print(f"💾 Saved test set → {test_path}")


if __name__ == "__main__":
    main()
