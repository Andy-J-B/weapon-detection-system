# --------------------------------------------------------------
#  train_binary.py  –  GPU‑accelerated training of a 1‑class YOLO‑v5
# --------------------------------------------------------------
import argparse
import sys
from pathlib import Path

# Ultralytics YOLO‑v5 repo must be checked out next to this file.
YOLOV5_ROOT = Path(__file__).resolve().parent / "yolov5"
if not YOLOV5_ROOT.is_dir():
    raise RuntimeError(
        f"YOLO‑v5 source not found at {YOLOV5_ROOT}. "
        "Clone it once with:  git clone https://github.com/ultralytics/yolov5.git"
    )
sys.path.append(str(YOLOV5_ROOT))


def train(
    data_yaml: str,
    epochs: int = 150,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    weights: str = "yolov5s.pt",
    project: str = "runs/train",
    name: str = "weapon_binary",
):
    """
    Trains a YOLO‑v5 model for the binary weapon detector.
    The best checkpoint is automatically exported to ONNX as best.onnx.
    """
    from yolov5 import train as yolov5_train  # type: ignore

    # ----------------------------------------------------------
    # Build the argument namespace expected by yolov5.train()
    # ----------------------------------------------------------
    args = argparse.Namespace(
        data=data_yaml,
        cfg="yolov5s.yaml",  # architecture – you can also try yolov5n.yaml for an even smaller model
        weights=weights,
        batch_size=batch,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        augment=True,
        rect=False,
        cache=False,
        workers=8,
        single_cls=False,  # keep false – we already have nc=1 in yaml
        patience=30,
        resume=False,
        nosave=False,
        noval=False,
        evolve=False,
        freeze=0,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=0.05,
        cls=0.5,
        dfl=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # -------------------  Misc  -------------------
        save_dir=None,  # let YOLO decide (project/name)
        exist_ok=False,
        verbose=True,
    )
    # Run the training job
    yolov5_train.run(args)

    # ----------------------------------------------------------
    # Export the best checkpoint to ONNX
    # ----------------------------------------------------------
    best_pt = Path(project) / name / "weights" / "best.pt"
    if not best_pt.is_file():
        raise RuntimeError(f"Training finished but {best_pt} not found")

    from yolov5 import models  # type: ignore

    model = models.common.AutoShape(str(best_pt))  # wrapper that provides .export()
    onnx_path = Path(project) / name / "weights" / "best.onnx"
    model.export(format="onnx", opset=12, simplify=True)
    print(f"\n✅ Exported ONNX model → {onnx_path.resolve()}\n")
    return onnx_path


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a binary weapon detector (YOLO‑v5) on the GPU."
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="weapon_data/weapon.yaml",
        help="Path to dataset yaml (default: weapon_data/weapon.yaml)",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=150, help="Number of epochs (default 150)"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=16, help="Batch size per GPU (default 16)"
    )
    parser.add_argument(
        "-s",
        "--imgsz",
        type=int,
        default=640,
        help="Training / inference image dimension (default 640)",
    )
    parser.add_argument(
        "-g",
        "--device",
        type=str,
        default="0",
        help='CUDA device id (e.g. "0" for first GPU, "-1" for CPU)',
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="yolov5s.pt",
        help="Pre‑trained backbone (default yolov5s.pt)",
    )
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        weights=args.weights,
    )
