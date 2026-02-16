#!/usr/bin/env python3
# ----------------------------------------------------------------------
# train_binary.py  –  Binary weapon detector (YOLO‑v5 / Ultralytics v8)
#
#   •  Input:  weapon_data/weapon.yaml  (nc = 1, name = ['weapon'])
#   •  Output: runs/train/weapon_binary/weights/best.onnx
#   •  GPU is used automatically when available (`--device 0`).
#   •  New: you can stop training (Ctrl‑C) and later resume it.
# ----------------------------------------------------------------------
import argparse
import signal
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# Helper: print a nice banner
# ----------------------------------------------------------------------
def banner(txt: str) -> None:
    line = "#" * (len(txt) + 8)
    print(f"\n{line}\n##  {txt}  ##\n{line}\n")


# ----------------------------------------------------------------------
# SIGINT handling – allows a graceful stop and checkpoint write
# ----------------------------------------------------------------------
def _install_sigint_handler():
    """Install a handler that converts SIGINT into KeyboardInterrupt."""

    def _handler(signum, frame):
        # the Ultralytics trainer catches KeyboardInterrupt and exits cleanly
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)


# ----------------------------------------------------------------------
# Main training routine (Ultralytics YOLO API)
# ----------------------------------------------------------------------
def train(
    data_yaml: str,
    epochs: int = 150,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",  # "0" = first CUDA device, "-1" = CPU
    weights: str = "yolov5s.pt",
    project: str = "runs/train",
    name: str = "weapon_binary",
    patience: int = 30,
    resume: bool = False,
    ckpt: str | None = None,
) -> Path:
    """
    Trains a YOLO‑v5 model for the binary weapon detection task,
    optionally resuming from a previous checkpoint, and finally
    exports the best checkpoint to ONNX.

    Returns the path to the exported ONNX file (or the checkpoint path
    if training was interrupted).
    """
    # ------------------------------------------------------------------
    # 1️⃣  Import the high‑level Ultralytics API (downloads YOLO‑v5 repo on first run)
    # ------------------------------------------------------------------
    try:
        from ultralytics import YOLO
    except Exception as e:  # pip install ultralytics
        raise RuntimeError(
            "❌  Ultralytics not installed. Run: pip install ultralytics"
        ) from e

    # ------------------------------------------------------------------
    # 2️⃣  Resolve what weights to load:
    #     * If a checkpoint is supplied (`ckpt`) → use it.
    #     * Else if ``resume`` is True → look for the automatic `last.pt`.
    #     * Otherwise fall back to the pretrained backbone (`weights`).
    # ------------------------------------------------------------------
    exp_dir = Path(project) / name
    weights_to_load = weights  # default

    if ckpt:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"❌  Specified checkpoint not found: {ckpt_path}")
        weights_to_load = str(ckpt_path)
        banner(f"LOADING USER‑SPECIFIED CHECKPOINT: {ckpt_path}")
    elif resume:
        last_pt = exp_dir / "weights" / "last.pt"
        if last_pt.is_file():
            weights_to_load = str(last_pt)
            banner(f"RESUMING FROM LAST CHECKPOINT: {last_pt}")
        else:
            print(
                "⚠️  No 'last.pt' found in the run folder – starting from the "
                f"pre‑trained backbone ({weights}) instead."
            )

    # ------------------------------------------------------------------
    # 3️⃣  Initialise the model
    # ------------------------------------------------------------------
    banner("INITIALISING YOLO‑v5")
    model = YOLO(weights_to_load)  # loads either .pt or .yaml

    # ------------------------------------------------------------------
    # 4️⃣  Install graceful‑shutdown handler (Ctrl‑C → KeyboardInterrupt)
    # ------------------------------------------------------------------
    _install_sigint_handler()

    # ------------------------------------------------------------------
    # 5️⃣  Train (wrapped so we can catch a KeyboardInterrupt)
    # ------------------------------------------------------------------
    banner("STARTING TRAINING")
    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            patience=patience,
            resume=resume,  # ✅ let Ultralytics reload the checkpoint if present
            cache=False,  # do not cache all images (saves RAM)
            workers=8,
            # hyper‑parameters you can tweak later (default values are fine)
            # lr0=0.01,  lrf=0.01,  momentum=0.937,  weight_decay=5e-4,
            # augment=True,
        )
        training_completed = True
    except KeyboardInterrupt:
        # ------------------------------------------------------------------
        # We have been interrupted – the trainer already saved a 'last.pt'
        # checkpoint before bubbling the exception up, so we just exit gracefully.
        # ------------------------------------------------------------------
        print(
            "\n⚠️  Training was interrupted by the user – "
            "the latest checkpoint (last.pt) has been saved."
        )
        training_completed = False

    # ------------------------------------------------------------------
    # 6️⃣  Export ONNX *only* if the training run finished normally.
    #     If it was interrupted you can resume later with `--resume`.
    # ------------------------------------------------------------------
    if training_completed:
        best_pt = exp_dir / "weights" / "best.pt"
        if not best_pt.is_file():
            raise RuntimeError(f"❌  Expected 'best.pt' not found: {best_pt}")

        banner("EXPORTING BEST CHECKPOINT TO ONNX")
        onnx_path = exp_dir / "weights" / "best.onnx"
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
    else:
        # Return the latest checkpoint so the caller can know where to resume from.
        last_pt = exp_dir / "weights" / "last.pt"
        return last_pt


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
        "-b",
        "--batch",
        type=int,
        default=16,
        help="Batch size per GPU (default 16)",
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
    # ---- NEW options ----------------------------------------------------
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint (last.pt) if it exists.",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        default=None,
        help="Path to a specific checkpoint to resume from (overrides --resume).",
    )
    # --------------------------------------------------------------------
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
        resume=args.resume,
        ckpt=args.ckpt,
    )
