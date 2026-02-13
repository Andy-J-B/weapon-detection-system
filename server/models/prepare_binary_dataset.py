#!/usr/bin/env python3
# ----------------------------------------------------------------------
#   prepare_binary_dataset.py
#
#   What it does
#   -------------
#   * Downloads the Kaggle *Gun‑detection* archive (if it is not already
#     present locally).
#   * Finds the `classes.txt` file – it may be anywhere inside the
#     downloaded folder (the original dataset puts it in
#     `<root>/Gunmen Dataset/All/classes.txt`).  If it cannot be found we
#     fall back to the known numeric ID for the **Gun** class (16).
#   * Reads the class names you want to treat as “weapon” (default = ["Gun"])
#     and converts those original IDs into a **single new class 0 = weapon**.
#   * Optionally merges a second YOLO‑v5‑style knife dataset.
#   * Writes a binary‑class YOLO‑v5 dataset (`weapon.yaml`) where every
#     weapon annotation has class id 0 and all other objects are discarded,
#     producing an empty label file → “background / nothing”.
#   * Splits the data 80 % train / 20 % val (random, seed‑controlled) and
#     prints a short report.
#
#   Usage
#   -----
#   python prepare_binary_dataset.py -o ./weapon_data
#   python prepare_binary_dataset.py -o ./weapon_data -k /path/to/knife_dataset
#   python prepare_binary_dataset.py -o ./weapon_data -t Gun Knife   # if you also have a Knife class in the original dataset
#
# ----------------------------------------------------------------------
import argparse
import random
import shutil
from pathlib import Path
import kagglehub
import tqdm
from typing import List, Dict, Tuple, Optional


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------
def read_yolo_txt(txt_path: Optional[Path]) -> List[Tuple[str, ...]]:
    """
    Return a list of tuples (class_id, x_center, y_center, w, h) read from a
    YOLO‑v5 txt file.  If the file does not exist or is empty an empty list
    is returned – this signals a background‑only image.
    """
    if txt_path is None or not txt_path.is_file():
        return []  # background image
    with txt_path.open("r") as f:
        lines = f.read().strip().splitlines()
    # Keep everything as strings; later we only cast the class id to int.
    return [tuple(l.split()) for l in lines if l]


def write_yolo_txt(txt_path: Path, rows: List[Tuple]):
    """Write a YOLO‑v5 txt file – rows may be empty (background)."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w") as f:
        for r in rows:
            f.write(" ".join(map(str, r)) + "\n")


def copy_image(src: Path, dst_dir: Path):
    """Copy a JPEG/PNG image to the destination folder."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)


# ----------------------------------------------------------------------
# Build a stem → .txt‑file lookup (fast O(1) access)
# ----------------------------------------------------------------------
def build_label_lookup(root: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for txt_path in root.rglob("*.txt"):
        lookup[txt_path.stem] = txt_path
    return lookup


# ----------------------------------------------------------------------
# Find the (first) classes.txt file anywhere under ``root``.
# ----------------------------------------------------------------------
def locate_classes_file(root: Path) -> Optional[Path]:
    candidates = list(root.rglob("classes.txt"))
    if not candidates:
        return None
    # Prefer the deepest path (most specific) – that is usually the correct one.
    return sorted(candidates, key=lambda p: len(p.parts))[-1]


# ----------------------------------------------------------------------
# Parse the classes file and return the numeric IDs that correspond to the
# supplied ``target_names`` (case‑insensitive).  If the file cannot be found,
# we fall back to a hard‑coded map where "Gun" → 16.
# ----------------------------------------------------------------------
def get_target_ids(classes_txt: Optional[Path], target_names: List[str]) -> List[int]:
    """
    Return the list of integer class IDs that belong to the weapon classes
    you asked for.  ``target_names`` may be e.g. ["Gun"] or ["Gun","Knife"].
    """
    # ------------------------------------------------------------------
    # If we have a real classes.txt file, read it line‑by‑line.
    # ------------------------------------------------------------------
    if classes_txt is not None and classes_txt.is_file():
        with classes_txt.open("r") as f:
            class_names = [ln.strip() for ln in f if ln.strip()]
        name_to_id = {name.lower(): idx for idx, name in enumerate(class_names)}
        missing = [n for n in target_names if n.lower() not in name_to_id]
        if missing:
            raise ValueError(
                f"The following target class(es) were NOT found in {classes_txt}:\n"
                f"{missing}\nAvailable classes: {class_names}"
            )
        return [name_to_id[n.lower()] for n in target_names]

    # ------------------------------------------------------------------
    # No classes.txt – fall back to known numeric IDs.
    # The original Gun‑detection dataset uses the following order
    # (0‑based indexing).  “Gun” is at index 16.
    # ------------------------------------------------------------------
    fallback = {
        "gun": 16,
        "knife": 0,  # only relevant if you happen to have a separate knife set
        "human": 15,
    }
    ids: List[int] = []
    for name in target_names:
        key = name.lower()
        if key in fallback:
            ids.append(fallback[key])
        else:
            raise ValueError(
                f"Cannot find class '{name}' – no classes.txt and no fallback entry."
            )
    print(
        "⚠️  classes.txt not found – using hard‑coded IDs "
        f"{list(zip(target_names, ids))}"
    )
    return ids


# ----------------------------------------------------------------------
# Recursive image finder (case‑insensitive extensions)
# ----------------------------------------------------------------------
def find_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


# ----------------------------------------------------------------------
# Main conversion routine
# ----------------------------------------------------------------------
def main(
    out_root: Path,
    knife_src: Optional[Path] = None,
    target_names: List[str] = None,
    seed: int = 42,
):
    """
    Create a binary (weapon / background) YOLO‑v5 data set.

    Parameters
    ----------
    out_root : Path
        Destination folder (will contain images/, labels/ and weapon.yaml).
    knife_src : Path | None
        Optional second dataset (already in YOLO‑v5 format) that contains knives.
    target_names : list[str] | None
        Human‑readable class names that should be treated as weapons.
        Default = ["Gun"].
    seed : int
        Random seed for the train/val split.
    """
    random.seed(seed)

    # ==============================================================
    # 1️⃣  Download the gun‑detection Kaggle archive
    # ==============================================================

    print("🔽 Downloading gun‑detection dataset from Kaggle …")
    gun_root = Path(kagglehub.dataset_download("ugorjiir/gun-detection"))

    # --------------------------------------------------------------
    # 2️⃣  Locate ``classes.txt`` (may be anywhere under the root)
    # --------------------------------------------------------------
    classes_txt_path = locate_classes_file(gun_root)
    if classes_txt_path:
        print(f"🔍  Found classes.txt at: {classes_txt_path}")
    else:
        print("⚠️  Unable to locate classes.txt – will use a built‑in fallback map.")

    # --------------------------------------------------------------
    # 3️⃣  Which original class IDs map to the single “weapon” class?
    # --------------------------------------------------------------
    if target_names is None:
        target_names = ["Gun"]  # default
    target_ids = get_target_ids(classes_txt_path, target_names)
    print(f"🔎  Mapping original class IDs {target_ids} → new class 0 (weapon)")

    # --------------------------------------------------------------
    # 4️⃣  Build fast lookup tables for label files
    # --------------------------------------------------------------
    gun_label_lookup = build_label_lookup(gun_root)
    gun_imgs = find_images(gun_root)

    # --------------------------------------------------------------
    # 5️⃣  OPTIONAL: second knife dataset
    # --------------------------------------------------------------
    knife_imgs: List[Path] = []
    knife_label_lookup: Dict[str, Path] = {}
    if knife_src:
        knife_src = Path(knife_src)
        knife_label_lookup = build_label_lookup(knife_src)
        knife_imgs = find_images(knife_src)
        print(f"🔪  Added {len(knife_imgs)} knife images (all will become class 0).")

    # --------------------------------------------------------------
    # 6️⃣  Merge all image paths
    # --------------------------------------------------------------
    all_imgs = gun_imgs + knife_imgs

    # --------------------------------------------------------------
    # 7️⃣  Train / validation split (80 % train / 20 % val)
    # --------------------------------------------------------------
    indices = list(range(len(all_imgs)))
    random.shuffle(indices)
    n_val = int(0.20 * len(all_imgs))
    val_idx = set(indices[:n_val])
    train_idx = set(indices[n_val:])

    # --------------------------------------------------------------
    # 8️⃣  Create folder hierarchy
    # --------------------------------------------------------------
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # 9️⃣  Process each image
    # --------------------------------------------------------------
    weapon_img_cnt = 0
    for i, img_path in enumerate(
        tqdm.tqdm(all_imgs, total=len(all_imgs), desc="Re‑labelling")
    ):
        # ----------------------------------------------------------
        # Choose the proper label lookup (gun vs knife)
        # ----------------------------------------------------------
        if img_path in gun_imgs:
            label_lookup = gun_label_lookup
        else:
            label_lookup = knife_label_lookup

        # ----------------------------------------------------------
        # Load original YOLO annotations (may be empty)
        # ----------------------------------------------------------
        orig_txt_path = label_lookup.get(img_path.stem)  # None → no file
        rows = read_yolo_txt(orig_txt_path)

        # ----------------------------------------------------------
        # Keep only rows that belong to the weapon IDs (or all rows for knife)
        # ----------------------------------------------------------
        weapon_rows: List[Tuple] = []

        # –‑ Gun‑dataset —
        for r in rows:
            old_id = int(r[0])
            if old_id in target_ids:
                weapon_rows.append((0, *r[1:]))  # new class id = 0 (weapon)

        # –‑ Knife‑dataset (if any) —
        #   In most public knife datasets the original class ids are 0 (knife)
        #   and 1 (handgun).  Both are weapons for us, so we just map them to 0.
        if img_path in knife_imgs:
            for r in rows:  # keep the original bbox
                weapon_rows.append((0, *r[1:]))

        # ----------------------------------------------------------
        # Destination split
        # ----------------------------------------------------------
        if i in train_idx:
            img_dst = out_root / "images" / "train"
            lab_dst = out_root / "labels" / "train"
        else:
            img_dst = out_root / "images" / "val"
            lab_dst = out_root / "labels" / "val"

        # ----------------------------------------------------------
        # Copy image & write (possibly empty) label file
        # ----------------------------------------------------------
        copy_image(img_path, img_dst)
        write_yolo_txt(lab_dst / (img_path.stem + ".txt"), weapon_rows)

        if weapon_rows:
            weapon_img_cnt += 1

    # --------------------------------------------------------------
    # 10️⃣  Write the YOLO‑v5 dataset descriptor (weapon.yaml)
    # --------------------------------------------------------------
    yaml_content = f"""\
train: {out_root / "images" / "train"}
val:   {out_root / "images" / "val"}

nc: 1
names: [ 'weapon' ]
"""
    (out_root / "weapon.yaml").write_text(yaml_content)

    # --------------------------------------------------------------
    # 11️⃣  Summary
    # --------------------------------------------------------------
    total_imgs = len(all_imgs)
    print("\n✅  Dataset creation finished")
    print(f"   Total images               : {total_imgs}")
    print(f"   Images containing a weapon: {weapon_img_cnt}")
    print(f"   Background‑only images      : {total_imgs - weapon_img_cnt}")
    print(f"   Train images               : {len(train_idx)}")
    print(f"   Validation images          : {len(val_idx)}")
    print(f"   Descriptor written to       : {out_root / 'weapon.yaml'}\n")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create a **binary** weapon / background dataset for YOLO‑v5. "
            "Only the class(es) you specify (default = ['Gun']) are kept and "
            "remapped to a single new class 0 = weapon. All other objects are "
            "discarded, which results in an empty label file (background)."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="weapon_data",
        help="Root folder where the processed dataset will be written (default: ./weapon_data)",
    )
    parser.add_argument(
        "-k",
        "--knife",
        type=str,
        default=None,
        help="Path to a YOLO‑v5 formatted knife dataset (optional).",
    )
    parser.add_argument(
        "-t",
        "--target-classes",
        nargs="+",
        default=["Gun"],
        help=(
            "One or more class names (exactly as they appear in the original "
            "classes.txt) that should be treated as a weapon. Case‑insensitive. "
            "Default = ['Gun']."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split (default 42).",
    )
    args = parser.parse_args()
    main(
        out_root=Path(args.output),
        knife_src=args.knife,
        target_names=args.target_classes,
        seed=args.seed,
    )
