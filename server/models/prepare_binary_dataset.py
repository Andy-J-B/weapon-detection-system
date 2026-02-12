# --------------------------------------------------------------
#  prepare_dataset_binary.py
#   • Downloads the Kaggle gun‑detection set (if not already present)
#   • Optionally adds a public knife dataset
#   • Collapses Knife (id 0) and Handgun (id 1) into a single class 0 = weapon
#   • Discards every other label (background / human / …)
#   • Writes a YOLO‑v5 “weapon.yaml” with nc = 1
# --------------------------------------------------------------
import argparse
import random
import shutil
from pathlib import Path
import kagglehub
import tqdm


# ------------------------------------------------------------------
# Helper I/O
# ------------------------------------------------------------------
def read_yolo_txt(txt_path: Path):
    with txt_path.open("r") as f:
        lines = f.read().strip().splitlines()
    return [tuple(l.split()) for l in lines if l]


def write_yolo_txt(txt_path: Path, rows):
    with txt_path.open("w") as f:
        for r in rows:
            f.write(" ".join(map(str, r)) + "\n")


def copy_image(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)


# ------------------------------------------------------------------
# Main conversion routine
# ------------------------------------------------------------------
def main(out_root: Path, knife_src: Path = None, seed: int = 42):
    random.seed(seed)

    # ---------- 1️⃣  Download gun‑detection set -----------------------
    print("🔽 Downloading gun‑detection dataset from Kaggle …")
    gun_root = Path(kagglehub.dataset_download("ugorjiir/gun-detection"))
    gun_imgs = list((gun_root / "All" / "images").glob("*.jpg"))
    gun_labs = [(gun_root / "All" / "labels" / (p.stem + ".txt")) for p in gun_imgs]

    # ---------- 2️⃣  Optional knife source ---------------------------
    knife_imgs = []
    knife_labs = []
    if knife_src:
        knife_src = Path(knife_src)
        knife_imgs = list((knife_src / "images").glob("*.jpg"))
        knife_labs = [(knife_src / "labels" / (p.stem + ".txt")) for p in knife_imgs]

    # ---------- 3️⃣  Merge the two lists ----------------------------
    all_imgs = gun_imgs + knife_imgs
    all_labs = gun_labs + knife_labs

    # ---------- 4️⃣  Train/val split (80/20) ------------------------
    indices = list(range(len(all_imgs)))
    random.shuffle(indices)
    n_val = int(0.20 * len(indices))
    val_idx = set(indices[:n_val])
    train_idx = set(indices[n_val:])

    # ---------- 5️⃣  Destination hierarchy -------------------------
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # ---------- 6️⃣  Process every image ----------------------------
    for i, (img_path, lab_path) in enumerate(
        tqdm.tqdm(zip(all_imgs, all_labs), total=len(all_imgs), desc="Re‑labelling")
    ):
        rows = read_yolo_txt(lab_path)

        # Keep *only* knives (old id=0) and handguns (old id=1)
        # Everything else → discard (background)
        weapon_rows = []
        for r in rows:
            old_id = int(r[0])
            if old_id in (0, 1):  # 0=knife, 1=handgun in our raw data
                weapon_rows.append((0, *r[1:]))  # new class 0 = weapon
            # else: ignore → treated as background

        # Choose split
        if i in train_idx:
            img_dst = out_root / "images" / "train"
            lab_dst = out_root / "labels" / "train"
        else:
            img_dst = out_root / "images" / "val"
            lab_dst = out_root / "labels" / "val"

        copy_image(img_path, img_dst)
        write_yolo_txt(lab_dst / (img_path.stem + ".txt"), weapon_rows)

    # ---------- 7️⃣  Write YOLO‑v5 yaml descriptor -----------------
    yaml_content = f"""\
train: {out_root / "images" / "train"}
val:   {out_root / "images" / "val"}

nc: 1
names: [ 'weapon' ]
"""
    (out_root / "weapon.yaml").write_text(yaml_content)
    print("\n✅  Dataset ready →", out_root / "weapon.yaml")
    print(f"   Train images: {len(train_idx)}   Val images: {len(val_idx)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Create a binary weapon / background dataset for YOLO‑v5."
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="weapon_data",
        help="Root folder for the processed dataset (default: ./weapon_data)",
    )
    p.add_argument(
        "-k",
        "--knife",
        type=str,
        default=None,
        help="Path to a YOLO‑v5 formatted knife dataset (optional).",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = p.parse_args()
    main(Path(args.output), knife_src=args.knife, seed=args.seed)
