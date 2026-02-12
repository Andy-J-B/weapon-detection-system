# --------------------------------------------------------------
#  train_resume.py – YOLOv5 weapon detection with resumable training
# --------------------------------------------------------------
#   Usage:
#       python train_resume.py               # fresh start from yolov5su.pt
#       python train_resume.py --resume      # continue from last.pt if it exists
#
#   You can also override a few defaults (epochs, batch, img size) on
#   the command line – see the argparse help.
# --------------------------------------------------------------

import argparse
import signal
import sys
from pathlib import Path

from ultralytics import YOLO


# ----------------------------------------------------------------------
# 1️⃣  SIGINT (Ctrl‑C) handler – lets YOLO finish the current epoch
# ----------------------------------------------------------------------
def _sigint_handler(sig, frame):
    """Gracefully exit on Ctrl‑C – the trainer will finish the epoch and
    write `last.pt`.  We just raise SystemExit after the signal."""
    print(
        "\n⚠️  Caught interrupt (SIGINT). "
        "YOLO will finish the current epoch and save a checkpoint …"
    )
    raise SystemExit(0)


signal.signal(signal.SIGINT, _sigint_handler)


# ----------------------------------------------------------------------
# 2️⃣  Argument parsing – expose a `--resume` flag and a few common knobs
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2‑class (knife / handgun) YOLOv5 model "
        "with automatic resume support."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the most recent checkpoint (runs/.../weights/last.pt). "
        "If no checkpoint is found we start from the pretrained yolov5su.pt.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train (default: 100).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use – e.g. cpu, 0 (first GPU), 0,1 (multiple GPUs).",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Base learning rate (default 0.001 – you can increase to 0.01).",
    )
    parser.add_argument(
        "--project",
        default="server/models/runs/train",
        help="Folder where training runs are stored.",
    )
    parser.add_argument(
        "--name",
        default="weapon_v5s_640_cpu",
        help="Name of the sub‑folder for this run (inside `project`).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# 3️⃣  Find an existing checkpoint (if any) and decide which model file to load
# ----------------------------------------------------------------------
def locate_checkpoint(args) -> Path:
    """
    Return a pathlib.Path to the checkpoint we should load.
    If `--resume` is set and a checkpoint exists we return that.
    Otherwise we return the pretrained `yolov5su.pt` checkpoint that ships with Ultralytics.
    """
    # The checkpoint lives in the folder that the trainer creates:
    #   runs/train/<project>/<name>/weights/last.pt
    ckpt_dir = Path(args.project) / args.name / "weights"
    last_pt = ckpt_dir / "last.pt"

    if args.resume:
        if last_pt.is_file():
            print(f"🔁 Resuming training from checkpoint: {last_pt}")
            return last_pt
        else:
            print(
                "⚠️  --resume was given but no checkpoint found. "
                "Starting from the pretrained 'yolov5su.pt'."
            )
    # fresh start – use the official pretrained checkpoint
    return Path("yolov5su.pt")


# ----------------------------------------------------------------------
# 4️⃣  Main training routine – everything is wrapped in a function
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 4‑a. Choose which .pt file to initialise the YOLO model with
    # ------------------------------------------------------------------
    model_path = locate_checkpoint(args)
    model = YOLO(str(model_path))

    # ------------------------------------------------------------------
    # 4‑b. Train – note the `resume=args.resume` flag.
    # ------------------------------------------------------------------
    train_results = model.train(
        data=r"/Users/Andy_1/dev/code/programs/GitHub/weapon-detection-system/server/models/dataset.yaml",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        optimizer="AdamW",
        lr0=args.lr0,
        weight_decay=0.0005,
        patience=10,
        save_period=5,
        augment=True,
        flipud=0.5,
        fliplr=0.5,
        project=args.project,
        name=args.name,
        # **Key** – tell the trainer we want it to be able to resume
        resume=args.resume,
    )

    # ------------------------------------------------------------------
    # 4‑c. OPTIONAL – Export the *final* best.pt to ONNX (you probably want
    #           this only once the training *actually* finishes).
    # ------------------------------------------------------------------
    best_pt = Path(train_results.save_dir) / "weights" / "best.pt"
    if best_pt.is_file():
        print(f"\n✅ Training finished – best checkpoint: {best_pt}")
        print("🚀 Exporting to ONNX …")
        # Re‑load the best model (ensures any tiny modifications from the trainer are applied)
        best_model = YOLO(str(best_pt))
        best_model.export(
            format="onnx",
            opset=12,
            imgsz=args.imgsz,
            nms=False,  # we will do NMS in C++ (see my earlier answer)
            simplify=True,
            project="server",  # puts the ONNX file into `server/`
            name="best",  # final name will be  server/best.onnx
        )
        print("📦 ONNX export complete → server/best.onnx")
    else:
        print("❌  No best.pt found – something went wrong during training.")


# ----------------------------------------------------------------------
# 5️⃣  Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # This block is a safety‑net – if the signal handler fails for some reason,
        # we still exit cleanly.
        print("\n⚠️  KeyboardInterrupt caught – exiting.")
        sys.exit(0)
