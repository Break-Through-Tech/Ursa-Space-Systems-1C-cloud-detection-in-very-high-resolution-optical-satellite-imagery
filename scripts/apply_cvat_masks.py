"""
Step 2 of 2: put the fixed CVAT masks back into each crop folder.

After fixing the masks in CVAT, export the task with
Actions > Export task dataset > "Segmentation mask 1.1" (no images needed).

Input:
  CVAT_EXPORT.zip   the zip you exported from CVAT
  ROOT              the same OCM folder you gave prep_for_cvat.py
  MANIFEST          manifest.csv made by prep_for_cvat.py

Output:
  ROOT/<scene>/crop_XX/mask_gt.png   single-band mask, 0 = not cloud, 255 = cloud
                                     (same 2048x2048 grid as image.tif; use mask > 0 for 0/1)

Colab usage:
    !python apply_cvat_masks.py /content/cvat_export.zip "/content/drive/MyDrive/<your OCM folder>" /content/cvat_upload/manifest.csv
"""
import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def cloud_color(z):
    """Read the cloud label's color from the export's labelmap.txt (default red)."""
    try:
        for line in z.read("labelmap.txt").decode().splitlines():
            if line.startswith("cloud:"):
                return tuple(int(v) for v in line.split(":")[1].split(","))
    except KeyError:
        pass
    return (255, 0, 0)


def main():
    export, root, manifest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

    dirs = defaultdict(list)
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            dirs[row["cvat_name"]].append(root / row["crop_dir"])

    written, missing = 0, []
    with zipfile.ZipFile(export) as z:
        color = np.array(cloud_color(z), np.uint8)
        masks = {Path(n).stem: n for n in z.namelist()
                 if n.startswith("SegmentationClass/") and n.endswith(".png")}

        for name, crop_dirs in dirs.items():
            if name not in masks:
                missing.append(name)
                continue
            rgb = np.array(Image.open(io.BytesIO(z.read(masks[name]))).convert("RGB"))
            gt = np.where(np.all(rgb == color, axis=-1), 255, 0).astype(np.uint8)

            for d in crop_dirs:
                h, w = tifffile.imread(d / "image.tif").shape[:2]
                if gt.shape != (h, w):
                    print(f"SIZE MISMATCH, skipped: {d} mask {gt.shape} vs image {(h, w)}")
                    continue
                Image.fromarray(gt).save(d / "mask_gt.png")
                written += 1

    print(f"wrote mask_gt.png into {written} crop folders")
    if missing:
        print("no CVAT mask found for:", ", ".join(missing))


if __name__ == "__main__":
    main()
