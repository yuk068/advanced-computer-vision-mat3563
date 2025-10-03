import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================
SRC_DIR = "data/SYSU-CEUS-FLL"   # Original dataset
PROCESSED_DIR = "data/SYSU-CEUS-FLL_processed"
NPZ_DIR = "data/SYSU-CEUS-FLL_npz"

FPS = 5
SIZE = (112, 112)  # Resize frames
TEST_RATIO = 0.2
RANDOM_SEED = 42

DO_DOWNSAMPLE = False
DO_PROCESS = False

# ==============================
# UTILS
# ==============================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resize_and_resample_video(src_path, dst_path):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"❌ Cannot open {src_path}")
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dst_path, fourcc, FPS, SIZE)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = FPS
    frame_interval = max(1, round(input_fps / FPS))

    count, kept = 0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 Processing {os.path.basename(src_path)}: {total_frames} frames at {input_fps:.2f} fps → target {FPS} fps")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_resized = cv2.resize(frame, SIZE)
            out.write(frame_resized)
            kept += 1
        count += 1

        if count % 50 == 0:
            print(f"   ⏳ {count}/{total_frames} frames processed...")

    cap.release()
    out.release()
    print(f"✅ Resampled {os.path.basename(src_path)} → {dst_path} ({kept} frames kept)\n")
    return True

def process_video_to_images(video_path, frame_dir):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞 Extracting frames from {os.path.basename(video_path)} ({total_frames} frames)")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, SIZE)
        frames.append(frame_resized)
        count += 1
        if count % 50 == 0:
            print(f"   ⏳ {count}/{total_frames} frames extracted...")

    cap.release()

    if len(frames) == 0:
        print(f"⚠️ Skipping empty video: {os.path.basename(video_path)}")
        return

    ensure_dir(frame_dir)
    for idx, frame in enumerate(frames, start=1):
        frame_name = f"f{idx:04d}.jpg"
        cv2.imwrite(os.path.join(frame_dir, frame_name), frame)

    print(f"✅ Saved {len(frames)} frames to {frame_dir}\n")

def load_video_frames(frame_dir):
    frames = []
    files = sorted(os.listdir(frame_dir))
    for file in files:
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(frame_dir, file))
        frames.append(img)
    if len(frames) == 0:
        return None
    return np.array(frames, dtype=np.uint8)

# ==============================
# MAIN
# ==============================
def main():
    ensure_dir(PROCESSED_DIR)
    ensure_dir(NPZ_DIR)

    classes = sorted([c for c in os.listdir(SRC_DIR)
                      if os.path.isdir(os.path.join(SRC_DIR, c))])
    print(f"📂 Found {len(classes)} classes: {classes}\n")

    # Step 1: Downsample videos
    if DO_DOWNSAMPLE:
        print("🔹 Starting downsampling of videos...\n")
        for class_name in classes:
            print(f"📁 Class: {class_name}")
            src_class_dir = os.path.join(SRC_DIR, class_name)
            dst_class_dir = os.path.join(PROCESSED_DIR, class_name)
            ensure_dir(dst_class_dir)

            avi_files = [f for f in os.listdir(src_class_dir) if f.endswith(".avi")]
            print(f"   {len(avi_files)} videos to process...")

            for file in avi_files:
                src_path = os.path.join(src_class_dir, file)
                dst_path = os.path.join(dst_class_dir, file.replace(".avi", ".mp4"))
                resize_and_resample_video(src_path, dst_path)
        print("🔹 Downsampling completed.\n")

    # Step 2: Extract frames
    if DO_PROCESS:
        print("🔹 Starting frame extraction...\n")
        for class_name in classes:
            class_dir = os.path.join(PROCESSED_DIR, class_name)
            video_files = [f for f in os.listdir(class_dir) if f.endswith(".mp4")]
            print(f"📁 Class: {class_name} has {len(video_files)} videos to extract frames from...")

            for file in video_files:
                video_name = file.replace(".mp4", "")
                video_path = os.path.join(class_dir, file)
                frame_dir = os.path.join(class_dir, video_name)
                process_video_to_images(video_path, frame_dir)
        print("🔹 Frame extraction completed.\n")

    # Step 3: Load frames and create npz
    print("🔹 Loading frames into memory and creating npz...\n")
    X, y, video_ids = [], [], []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(PROCESSED_DIR, class_name)
        vids = [v for v in sorted(os.listdir(class_dir)) if os.path.isdir(os.path.join(class_dir, v))]
        print(f"📁 Loading class {class_name}: {len(vids)} videos")

        for vid in vids:
            vid_dir = os.path.join(class_dir, vid)
            frames = load_video_frames(vid_dir)
            if frames is None:
                print(f"   ⚠️ Skipping empty video folder {vid}")
                continue
            X.append(frames)
            y.append(label)
            video_ids.append(f"{class_name}/{vid}")
            print(f"   ✅ Loaded {len(frames)} frames from {vid}")

    X = np.array(X, dtype=object)
    y = np.array(y)
    print(f"\n📊 Loaded total {len(X)} videos.")

    # Train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y
    )

    X_train, y_train, ids_train = X[train_idx], y[train_idx], [video_ids[i] for i in train_idx]
    X_test, y_test, ids_test = X[test_idx], y[test_idx], [video_ids[i] for i in test_idx]

    np.savez_compressed(os.path.join(NPZ_DIR, "train.npz"), X=X_train, y=y_train, ids=ids_train)
    np.savez_compressed(os.path.join(NPZ_DIR, "test.npz"), X=X_test, y=y_test, ids=ids_test)
    print(f"\n💾 Saved train/test npz in {NPZ_DIR}")
    print(f"   Train videos: {len(X_train)}, Test videos: {len(X_test)}\n")

if __name__ == "__main__":
    main()
