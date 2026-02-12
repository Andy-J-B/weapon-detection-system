#!/usr/bin/env python3
# ----------------------------------------------------------------------
# train_binary.py  –  Binary weapon detector (YOLO‑v5 / Ultralytics v8)
#
#   •  Input:  weapon_data/weapon.yaml  (nc = 1, name = ['weapon'])
#   •  Output: runs/train/weapon_binary/weights/best.onnx
#   •  GPU is used automatically when available (`--device 0`).
# ----------------------------------------------------------------------
import argparse
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# Helper: print a nice banner
# ----------------------------------------------------------------------
def banner(txt: str) -> None:
    line = "#" * (len(txt) + 8)
    print(f"\n{line}\n##  {txt}  ##\n{line}\n")


# ----------------------------------------------------------------------
# Main training routine (Ultralytics YOLO API)
# ----------------------------------------------------------------------
def train(
    data_yaml: str,
    epochs: int = 150,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",  # "0" = first CUDA device, "-1" = CPU
    weights: str = "yolov5s.pt",  # pretrained backbone (downloaded automatically)
    project: str = "runs/train",
    name: str = "weapon_binary",
    patience: int = 30,
) -> Path:
    """
    Trains a YOLO‑v5 model for the binary weapon detection task
    and exports the best checkpoint to ONNX.

    Returns the path to the exported ONNX file.
    """
    # --------------------------------------------------------------
    # 1️⃣  Import the high‑level Ultralytics API (this will download the
    #     official YOLO‑v5 repo the first time it runs)
    # --------------------------------------------------------------
    try:
        from ultralytics import YOLO
    except Exception as e:  # pip install ultralytics
        raise RuntimeError(
            "❌  Ultralytics not installed. Run: pip install ultralytics"
        ) from e

    # --------------------------------------------------------------
    # 2️⃣  Load a pretrained backbone (yolov5s is tiny & fast)
    # --------------------------------------------------------------
    banner("INITIALISING YOLO‑v5")
    model = YOLO(weights)  # weights can be .pt or .yaml

    # --------------------------------------------------------------
    # 3️⃣  Train
    # --------------------------------------------------------------
    banner("STARTING TRAINING")
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        patience=patience,
        # The following arguments are optional but useful for reproducibility
        cache=False,  # do not cache all images (saves RAM)
        workers=8,  # number of dataloader workers
        # hyper‑parameters you can tweak later
        # lr0=0.01,  lrf=0.01,  momentum=0.937,  weight_decay=5e-4,
        # augment=True,  # default – mosaic/HSV etc.
    )

    # --------------------------------------------------------------
    # 4️⃣  Export the best checkpoint to ONNX (ops‑et 12 → most compatible)
    # --------------------------------------------------------------
    best_pt = Path(project) / name / "weights" / "best.pt"
    if not best_pt.is_file():
        raise RuntimeError(f"❌  Expected checkpoint not found: {best_pt}")

    banner("EXPORTING BEST CHECKPOINT TO ONNX")
    onnx_path = Path(project) / name / "weights" / "best.onnx"
    model.export(
        format="onnx",
        opset=12,
        simplify=True,
        imgsz=imgsz,
        batch=1,
        device=device,
        half=False,  # keep FP32 (easier for OpenCV DNN)
        include=["model"],  # only the core model
        checkpoint=best_pt,
        save_dir=onnx_path.parent,
    )
    print(f"✅  ONNX model written to: {onnx_path.resolve()}")
    return onnx_path


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary weapon detector (YOLO‑v5) on the GPU. "
            "The script expects a YOLO‑v5 style dataset descriptor "
            "(nc=1, names=['weapon'])."
        )
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="weapon_data/weapon.yaml",
        help="Path to the dataset yaml file (default: weapon_data/weapon.yaml)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs (default 150)",
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=16, help="Batch size per GPU (default 16)"
    )
    parser.add_argument(
        "-s",
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training / inference (default 640)",
    )
    parser.add_argument(
        "-g",
        "--device",
        type=str,
        default="0",
        help='CUDA device id ("0" for first GPU, "-1" for CPU).',
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="yolov5s.pt",
        help="Pre‑trained backbone (default yolov5s.pt).",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default="runs/train",
        help="Root folder for all training runs.",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="weapon_binary",
        help="Sub‑folder name under the project directory.",
    )
    parser.add_argument(
        "-P",
        "--patience",
        type=int,
        default=30,
        help="Early‑stop patience (default 30 epochs).",
    )
    args = parser.parse_args()

    # Run training – everything else is handled inside `train()`
    train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        weights=args.weights,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )
