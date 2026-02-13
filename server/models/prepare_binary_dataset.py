import kagglehub
import os
import shutil
import yaml
import glob
import random
from pathlib import Path

# --- Configuration ---
TARGET_CLASSES = ["Human", "Gun"]  # We only want these
NEW_CLASS_MAP = {"Human": 0, "Gun": 1}  # Remap to clean 0 and 1
DATASET_DIR = "datasets/cleaned_gun_data"


def prepare_dataset():
    print("⬇️  Downloading dataset from Kaggle...")
    # Download dataset
    raw_path = kagglehub.dataset_download("ugorjiir/gun-detection")
    base_path = os.path.join(raw_path, "Gunmen Dataset", "All")

    print(f"📂 Raw data located at: {base_path}")

    # 1. Read the original dirty classes.txt
    old_classes_path = os.path.join(base_path, "classes.txt")
    if not os.path.exists(old_classes_path):
        raise FileNotFoundError("Could not find classes.txt in the dataset!")

    with open(old_classes_path, "r") as f:
        old_classes = [line.strip() for line in f.readlines()]

    # Map OLD index to NEW index (e.g., if 'Gun' was 16, map 16 -> 1)
    # If a class isn't in TARGET_CLASSES, it won't be in this map.
    id_map = {}
    for idx, name in enumerate(old_classes):
        if name in TARGET_CLASSES:
            id_map[idx] = NEW_CLASS_MAP[name]

    print(f"ℹ️  Class Mapping: {id_map} (Old ID -> New ID)")

    # 2. Setup standard YOLO directory structure
    images_train = os.path.join(DATASET_DIR, "images/train")
    images_val = os.path.join(DATASET_DIR, "images/val")
    labels_train = os.path.join(DATASET_DIR, "labels/train")
    labels_val = os.path.join(DATASET_DIR, "labels/val")

    for p in [images_train, images_val, labels_train, labels_val]:
        os.makedirs(p, exist_ok=True)

    # 3. Process Images and Labels
    all_images = glob.glob(os.path.join(base_path, "*.jpg")) + glob.glob(
        os.path.join(base_path, "*.png")
    )
    random.shuffle(all_images)

    split_idx = int(len(all_images) * 0.8)  # 80% train, 20% val
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    print("🧹 Cleaning labels and moving files...")

    def process_files(image_list, img_dest_dir, lbl_dest_dir):
        for img_path in image_list:
            # Copy Image
            shutil.copy(img_path, img_dest_dir)

            # Process Label
            txt_path = img_path.rsplit(".", 1)[0] + ".txt"
            if os.path.exists(txt_path):
                new_lines = []
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        class_id = int(parts[0])

                        # Only keep if it's a Human or Gun
                        if class_id in id_map:
                            new_class_id = id_map[class_id]
                            # Reconstruct line with new ID
                            new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
                            new_lines.append(new_line)

                # Write clean label file
                dest_txt_path = os.path.join(lbl_dest_dir, os.path.basename(txt_path))
                with open(dest_txt_path, "w") as f:
                    f.writelines(new_lines)

    process_files(train_imgs, images_train, labels_train)
    process_files(val_imgs, images_val, labels_val)

    # 4. Create data.yaml
    yaml_content = {
        "path": os.path.abspath(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["Human", "Gun"],  # 0: Human, 1: Gun
    }

    with open("data.yaml", "w") as f:
        yaml.dump(yaml_content, f)

    print("✅ Dataset preparation complete.")


def train_yolo():
    # Clone YOLOv5 if not present
    if not os.path.exists("yolov5"):
        print("⬇️  Cloning YOLOv5 repository...")
        os.system("git clone https://github.com/ultralytics/yolov5")
        os.system("pip install -r yolov5/requirements.txt")

    print("🚀 Starting Training on GPU...")
    # Run training command
    # img 640, batch 16, epochs 50, data.yaml, weights yolov5s.pt
    os.system(
        f"python yolov5/train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt --device 0 --name weapon_detect"
    )


if __name__ == "__main__":
    prepare_dataset()
    train_yolo()
    print(
        "\n🎉 Training finished! Check 'yolov5/runs/train/weapon_detect/weights/best.onnx' (You may need to export it first)."
    )
    print("To export to ONNX for your C++ server, run:")
    print(
        "python yolov5/export.py --weights yolov5/runs/train/weapon_detect/weights/best.pt --include onnx"
    )
