"""
Step 1 of 2: prep OCM draft masks for CVAT.

Input:  ROOT/.../<crop folder>/image.tif + an OCM mask
        (mask_draft.tif or mask_ocm_assisted.tif, whichever the folder has)

Output:
  CROPS/<same path as in ROOT>/image.png   -> PNG copy of image.tif (same pixels)
                                              CROPS defaults to ROOT (i.e. next to image.tif)
  OUT/[<group>/]images/<name>.png          -> renamed copies, drag ALL into one CVAT task
  OUT/[<group>/]cvat_masks.zip             -> Actions > Upload annotations > "Segmentation mask 1.1"
  OUT/[<group>/]manifest.csv               -> which upload name belongs to which crop folder
                                              (apply_cvat_masks.py uses it to put masks back)

Mask classes in:  0 clear, 1 thick, 2 thin, 3 shadow, 4 smoke, 5 unsure
CVAT labels out:  cloud  = thick + thin   (our rule: thin cloud counts as cloud)
                  smoke  = smoke          (kept separate; smoke is NOT cloud)
                  unsure = unsure
                  background = clear + shadow
In CVAT, create a project with labels named exactly: cloud, smoke, unsure

Optional --groups CSV (columns: crop, group) splits the output into one CVAT upload set
per group, e.g. group "1_high_priority/test". crop = the crop folder's name.

Colab usage:
    from google.colab import drive; drive.mount('/content/drive')
    !pip install -q tifffile imagecodecs
    !python prep_for_cvat.py "/content/drive/MyDrive/<your OCM folder>" /content/cvat_upload
"""
import argparse
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

MASK_NAMES = ("mask_draft.tif", "mask_ocm_assisted.tif")
LABELS = [  # (name, color, OCM classes)
    ("background", (0, 0, 0), ()),
    ("cloud", (255, 0, 0), (1, 2)),
    ("smoke", (255, 106, 0), (4,)),
    ("unsure", (0, 255, 60), (5,)),
]


def load_rgb(path):
    img = tifffile.imread(path)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)  # bands-first -> bands-last
    img = img[..., :3]
    if img.dtype != np.uint8:  # stretch 16-bit to 8-bit
        lo, hi = np.percentile(img, (1, 99))
        img = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return img


def mask_to_rgb(m):
    rgb = np.zeros((*m.shape, 3), np.uint8)
    for _, color, classes in LABELS:
        if classes:
            rgb[np.isin(m, classes)] = color
    return rgb


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="folder containing the crop folders")
    ap.add_argument("out", type=Path, help="where the CVAT upload files go")
    ap.add_argument("--crops", type=Path, help="where image.png copies go (default: next to image.tif)")
    ap.add_argument("--groups", type=Path, help="CSV with columns crop,group")
    args = ap.parse_args()
    crops_root = args.crops or args.root

    group_of = {}
    if args.groups:
        with open(args.groups, newline="") as f:
            group_of = {r["crop"]: r["group"] for r in csv.DictReader(f)}

    by_group = defaultdict(list)  # group -> [(name, crop_dir, img_path, mask_path)]
    for img_path in sorted(args.root.rglob("image.tif")):
        crop_dir = img_path.parent
        mask_path = next((crop_dir / n for n in MASK_NAMES if (crop_dir / n).exists()), None)
        if mask_path is None:
            print("skip (no OCM mask):", crop_dir)
            continue
        group = group_of.get(crop_dir.name, "") if group_of else ""
        if group_of and not group:
            print("skip (not in groups CSV):", crop_dir.name)
            continue
        name = f"{crop_dir.parent.name}__{crop_dir.name}"
        by_group[group].append((name, crop_dir, img_path, mask_path))

    labelmap = "# label:color_rgb:parts:actions\n" + "".join(
        f"{n}:{','.join(map(str, c))}::\n" for n, c, _ in LABELS)
    total = 0
    for group, items in sorted(by_group.items()):
        gout = args.out / group
        (gout / "images").mkdir(parents=True, exist_ok=True)
        seen, rows = set(), []
        with zipfile.ZipFile(gout / "cvat_masks.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("labelmap.txt", labelmap)
            for name, crop_dir, img_path, mask_path in items:
                rel = crop_dir.relative_to(args.root)
                rows.append((name, rel.as_posix()))
                img = load_rgb(img_path)
                dst = crops_root / rel
                dst.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(dst / "image.png")
                if name in seen:  # duplicate copy of a crop already in this group
                    continue
                seen.add(name)
                Image.fromarray(img).save(gout / "images" / f"{name}.png")
                z.writestr(f"SegmentationClass/{name}.png", png_bytes(mask_to_rgb(tifffile.imread(mask_path))))
            z.writestr("ImageSets/Segmentation/default.txt", "\n".join(sorted(seen)) + "\n")
        with open(gout / "manifest.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cvat_name", "crop_dir"])
            w.writerows(rows)
        total += len(seen)
        print(f"{group or '.'}: {len(seen)} crops -> {gout}")

    print(f"done: {total} crops; image.png copies under {crops_root}")


if __name__ == "__main__":
    main()
