import os
import cv2
import numpy as np

# ==============================
# CONFIG
# ==============================
SRC_DIR = "data/UCF50" 				# Original dataset
DOWNSAMPLED_DIR = "data/UCF50_downsampled"
PROCESSED_DIR = "data/UCF50_processed"

FPS = 5 							# Target FPS
SIZE = (112, 112) 					# Target resize

# Number of classes to use (selected uniformly from sorted list)
# Example: CLASS_NUM = 10 → picks 10 evenly spaced classes alphabetically
CLASS_NUM = 5

# Flags
DO_DOWNSAMPLE = True 	# True if you want resizing+fps
DO_PROCESS = True 		# True = extract frames + optical flow


# ==============================
# UTILS
# ==============================
def ensure_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def resize_and_resample_video(src_path, dst_path):
	"""Resize to fixed size and FPS, save to mp4."""
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
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if count % frame_interval == 0:
			frame_resized = cv2.resize(frame, SIZE)
			out.write(frame_resized)
			kept += 1
		count += 1

	cap.release()
	out.release()
	print(f"✅ Resampled {os.path.basename(src_path)} → {dst_path} "
		  f"({kept} frames at {FPS} fps)")
	return True


def process_video_to_images(video_path, frame_dir, flow_dir, video_name):
	"""Extract frames and optical flow, save as JPEG images."""
	cap = cv2.VideoCapture(video_path)
	frames = []

	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frame_resized = cv2.resize(frame, SIZE)
		frames.append(frame_resized)
	cap.release()

	total_frames = len(frames)
	if total_frames == 0:
		print(f"⚠️ Skipping empty video: {video_name}")
		return

	ensure_dir(frame_dir)
	ensure_dir(flow_dir)

	# Save frames
	for idx, frame in enumerate(frames, start=1):
		frame_name = f"{video_name}_f{idx:04d}.jpg"
		cv2.imwrite(os.path.join(frame_dir, frame_name), frame)

	# Compute and save optical flow (Farneback)
	gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
	for i in range(1, len(gray_frames)):
		flow = cv2.calcOpticalFlowFarneback(
			gray_frames[i-1], gray_frames[i],
			None, 0.5, 3, 15, 3, 5, 1.2, 0
		)
		fx, fy = flow[..., 0], flow[..., 1]

		# Normalize flow to 0–255 for storage
		fx_norm = cv2.normalize(fx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		fy_norm = cv2.normalize(fy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

		base_name = f"{video_name}_f{i:04d}"
		cv2.imwrite(os.path.join(flow_dir, f"{base_name}_flowx.jpg"), fx_norm)
		cv2.imwrite(os.path.join(flow_dir, f"{base_name}_flowy.jpg"), fy_norm)

	print(f"✅ Processed {video_name}: {total_frames} frames → {total_frames-1} flow pairs")


# ==============================
# MAIN
# ==============================
def main():
	# Ensure DOWNSAMPLED_DIR exists if DO_DOWNSAMPLE is True or if DO_PROCESS is True and expects pre-downsampled data
	# It will be created below in Step 1 if DO_DOWNSAMPLE is True, but adding it here is safer.
	ensure_dir(DOWNSAMPLED_DIR)
	# Ensure PROCESSED_DIR exists if DO_PROCESS is True
	ensure_dir(PROCESSED_DIR)

	# Pick classes uniformly from sorted list
	all_classes = sorted([c for c in os.listdir(DOWNSAMPLED_DIR)
						  if os.path.isdir(os.path.join(DOWNSAMPLED_DIR, c))])
	
	# If DOWNSAMPLED_DIR was just created, it will be empty.
	# We should ideally check SRC_DIR for classes first if DO_DOWNSAMPLE is True.
	# For simplicity, we stick to the original logic which assumes the class list
	# will be used to create the downsampled structure in the next step.
	
	if not all_classes and DO_DOWNSAMPLE:
		# Check SRC_DIR for classes if we intend to downsample
		all_classes = sorted([c for c in os.listdir(SRC_DIR)
							  if os.path.isdir(os.path.join(SRC_DIR, c))])
		if not all_classes:
			print("❌ No classes found in SRC_DIR or DOWNSAMPLED_DIR.")
			return
		
	elif not all_classes and DO_PROCESS and not DO_DOWNSAMPLE:
		# No downsampling, and DOWNSAMPLED_DIR is empty. Processing won't work.
		print("❌ No classes found in DOWNSAMPLED_DIR. Cannot process videos.")
		return
	
	elif not all_classes:
		# Fallback if both are False or if DOWNSAMPLED_DIR exists but is empty
		print("❌ No classes found in DOWNSAMPLED_DIR.")
		return

	stride = max(1, len(all_classes) // CLASS_NUM)
	selected_classes = [all_classes[i] for i in range(0, len(all_classes), stride)][:CLASS_NUM]

	print(f"📂 Selected {len(selected_classes)} classes (uniformly sampled): {selected_classes}")

	# Step 1: Downsample (resize + resample to FPS)
	if DO_DOWNSAMPLE:
		# Directory DOWNSAMPLED_DIR ensured above
		for class_name in selected_classes:
			class_dir = os.path.join(SRC_DIR, class_name)
			if not os.path.isdir(class_dir):
				continue

			down_class_dir = os.path.join(DOWNSAMPLED_DIR, class_name)
			ensure_dir(down_class_dir)

			for file in os.listdir(class_dir):
				if not file.endswith(".avi"):
					continue
				src_path = os.path.join(class_dir, file)
				dst_path = os.path.join(down_class_dir, file.replace(".avi", ".mp4"))
				resize_and_resample_video(src_path, dst_path)

	# Step 2: Extract frames + optical flow
	if DO_PROCESS:
		# Directory PROCESSED_DIR ensured above
		for class_name in selected_classes:
			class_dir = os.path.join(DOWNSAMPLED_DIR, class_name)
			if not os.path.isdir(class_dir):
				continue

			for file in os.listdir(class_dir):
				if not file.endswith(".mp4"):
					continue
				video_name = file.replace(".mp4", "")
				video_path = os.path.join(class_dir, file)

				frame_dir = os.path.join(PROCESSED_DIR, "frames", class_name, video_name)
				flow_dir = os.path.join(PROCESSED_DIR, "flows", class_name, video_name)

				# process_video_to_images internally calls ensure_dir for frame_dir and flow_dir
				process_video_to_images(video_path, frame_dir, flow_dir, video_name)


if __name__ == "__main__":
	main()