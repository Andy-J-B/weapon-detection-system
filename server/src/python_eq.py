#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binary weapon detector – ONNX inference (YOLO‑v5 / Ultralytics v8)

Key changes:
* Handles both (1,6,N) and (1,N,6) output layouts.
* No extra sigmoid – the ONNX already contains it.
* Debug prints the highest raw confidence and top‑5 objectness scores.
* Default confidence lowered to 0.001 for quick debugging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
log = logging.getLogger(__name__)
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log.addHandler(ch)
log.setLevel(logging.INFO)

# ----------------------------------------------------------------------
# Path to the exported ONNX model
# ----------------------------------------------------------------------
_MODEL_PATH = Path(
    "/Users/Andy_1/dev/code/programs/GitHub/weapon-detection-system/runs/detect/runs/train/weapon_binary7/weights/best.onnx"
)
_NET: cv2.dnn_Net | None = None  # global cache


def _load_net() -> cv2.dnn_Net:
    """Load the ONNX model once and cache it."""
    global _NET
    if _NET is None:
        if not _MODEL_PATH.is_file():
            raise FileNotFoundError(f"ONNX model not found at {_MODEL_PATH}")
        _NET = cv2.dnn.readNetFromONNX(str(_MODEL_PATH))
        _NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        log.info(f"✅ Loaded ONNX model from '{_MODEL_PATH}'")
    return _NET


def _letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, int, int]:
    """
    Resize + pad exactly like the YOLO‑v5 ``letterbox`` function.
    Returns (padded_image, scale, pad_w, pad_h).
    """
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)

    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))

    img_resized = cv2.resize(
        img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR
    )

    dw = new_shape[1] - new_unpad_w
    dh = new_shape[0] - new_unpad_h
    dw //= 2
    dh //= 2

    padded = cv2.copyMakeBorder(
        img_resized,
        top=dh,
        bottom=dh,
        left=dw,
        right=dw,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, r, dw, dh


def _postprocess(
    raw_pred: np.ndarray,
    orig_shape: Tuple[int, int],
    scale: float,
    pad_w: int,
    pad_h: int,
    conf_thresh: float = 0.001,
    nms_thresh: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    Decode the ONNX output (which can be (1,6,N) or (1,N,6)).
    Returns a list of detections – each detection is a dict with keys:
    ``bbox`` (x, y, w, h), ``confidence``, ``class_id``, ``class_name``.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Put the tensor into shape (N, 6)
    # ------------------------------------------------------------------
    pred = raw_pred[0]  # (6, N)  or  (N, 6)
    if pred.shape[0] == 6:  # channel‑first → transpose
        pred = pred.T  # (N, 6)

    # ------------------------------------------------------------------
    # 2️⃣  Gather boxes & confidence (the model already applied sigmoid)
    # ------------------------------------------------------------------
    boxes: List[List[int]] = []
    scores: List[float] = []

    # Keep the whole column of objectness for a quick debug later
    all_obj = pred[:, 4] if pred.shape[1] > 4 else None

    for row in pred:
        cx, cy, w, h = map(float, row[:4])

        obj_conf = float(row[4])  # already 0‑1
        cls_conf = float(row[5]) if row.shape[0] > 5 else 1.0
        conf = obj_conf * cls_conf

        if conf < conf_thresh:
            continue

        # centre‑xywh → top‑left‑right (pixel coordinates in the padded 640×640)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        # Undo the padding and scaling that we added in `_letterbox`
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # Clip to original image size
        img_h, img_w = orig_shape
        x1 = max(0.0, min(x1, img_w))
        y1 = max(0.0, min(y1, img_h))
        x2 = max(0.0, min(x2, img_w))
        y2 = max(0.0, min(y2, img_h))

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)

        boxes.append([int(round(x1)), int(round(y1)), int(round(bw)), int(round(bh))])
        scores.append(conf)

    # ------------------------------------------------------------------
    # 3️⃣  DEBUG: show the best raw objectness scores
    # ------------------------------------------------------------------
    if scores:
        log.debug(f"DEBUG: max confidence after threshold = {max(scores):.6f}")
    else:
        log.debug("DEBUG: no detections survived the confidence threshold")

    if all_obj is not None:
        top5 = np.sort(all_obj)[-5:][::-1]
        log.debug(f"DEBUG: top‑5 raw objectness scores (all anchors) = {top5}")

    # ------------------------------------------------------------------
    # 4️⃣  NMS (exactly like the C++ implementation)
    # ------------------------------------------------------------------
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)

    detections: List[Dict[str, Any]] = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detections.append(
                {
                    "bbox": (x, y, w, h),
                    "confidence": scores[i],
                    "class_id": 0,
                    "class_name": "weapon",
                }
            )
    return detections


def detect_weapon(
    image: np.ndarray,
    conf_thresh: float = 0.001,
    nms_thresh: float = 0.45,
    return_detections: bool = False,
) -> bool | Tuple[bool, List[Dict[str, Any]]]:
    """Run the detector on a single BGR image."""
    if image is None or image.size == 0:
        log.warning("⚠️  Received an empty image.")
        return (False, []) if return_detections else False

    net = _load_net()
    padded, scale, pad_w, pad_h = _letterbox(image, new_shape=(640, 640))

    blob = cv2.dnn.blobFromImage(
        padded,
        scalefactor=1.0 / 255.0,
        size=(640, 640),
        mean=(0, 0, 0),
        swapRB=True,  # BGR→RGB
        crop=False,
    )
    net.setInput(blob)

    try:
        raw_out = net.forward()  # (1, 6, N)  or (1, N, 6)
    except cv2.error as exc:
        log.error(f"❌ Inference failed: {exc}")
        return (False, []) if return_detections else False

    detections = _postprocess(
        raw_out,
        orig_shape=image.shape[:2],
        scale=scale,
        pad_w=pad_w,
        pad_h=pad_h,
        conf_thresh=conf_thresh,
        nms_thresh=nms_thresh,
    )

    if not detections:
        log.info("🟢 No weapon detected (background).")
        return (False, []) if return_detections else False

    best = max(detections, key=lambda d: d["confidence"])
    log.info(
        f"🔴 Weapon detected! bbox={best['bbox']} confidence={best['confidence']:.3f}"
    )
    return (True, detections) if return_detections else True


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Binary weapon detector demo")
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to an image (BGR). If omitted a synthetic black image is generated.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default 0.001 – lower it for debugging).",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.45,
        help="IoU NMS threshold (default 0.45).",
    )
    args = parser.parse_args()

    if args.image_path:
        img = cv2.imread(args.image_path)
        if img is None:
            sys.exit(f"❌ Could not read image '{args.image_path}'")
    else:
        img = np.zeros((240, 320, 3), dtype=np.uint8)  # black synthetic image

    has_weapon = detect_weapon(img, conf_thresh=args.conf, nms_thresh=args.nms)
    print("\nRESULT:", "Weapon present ✅" if has_weapon else "No weapon ❌")
