"""
Step 2 of 2: put the fixed CVAT masks back into each crop folder.

After fixing the masks in CVAT, export the task with
Actions > Export task dataset > "Segmentation mask 1.1" (no images needed).
If prep_for_cvat.py split the crops into groups, run this once per group,
with that group's export zip and manifest.csv.

Input:
  CVAT_EXPORT.zip   the zip you exported from CVAT
  CROPS             the folder holding the crop folders (the one with image.tif or image.png)
  MANIFEST          manifest.csv made by prep_for_cvat.py

Output:
  CROPS/<...>/crop_XX/mask_gt.png   single-band mask on the same 2048x2048 grid as image.tif:
                                      0 = not cloud, 255 = cloud, 64 = smoke, 128 = unsure
                                    For training, cloud = (mask == 255).

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

OUT_VALUES = {"cloud": 255, "smoke": 64, "unsure": 128}
DEFAULT_COLORS = {"cloud": (255, 0, 0), "smoke": (255, 106, 0), "unsure": (0, 255, 60)}


def label_colors(z):
    """Read each label's color from the export's labelmap.txt."""
    colors = dict(DEFAULT_COLORS)
    try:
        for line in z.read("labelmap.txt").decode().splitlines():
            if line and not line.startswith("#"):
                name, color = line.split(":")[:2]
                if name in OUT_VALUES and color:
                    colors[name] = tuple(int(v) for v in color.split(","))
    except KeyError:
        pass
    return colors


def image_size(d):
    ref = d / "image.tif"
    if ref.exists():
        return tifffile.imread(ref).shape[:2]
    w, h = Image.open(d / "image.png").size  # e.g. a local copy that only has image.png
    return h, w


def main():
    export, root, manifest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

    dirs = defaultdict(list)
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            dirs[row["cvat_name"]].append(root / row["crop_dir"])

    written, missing = 0, []
    with zipfile.ZipFile(export) as z:
        colors = label_colors(z)
        masks = {Path(n).stem: n for n in z.namelist()
                 if n.startswith("SegmentationClass/") and n.endswith(".png")}

        for name, crop_dirs in dirs.items():
            if name not in masks:
                missing.append(name)
                continue
            rgb = np.array(Image.open(io.BytesIO(z.read(masks[name]))).convert("RGB"))
            gt = np.zeros(rgb.shape[:2], np.uint8)
            for label, value in OUT_VALUES.items():
                gt[np.all(rgb == np.array(colors[label], np.uint8), axis=-1)] = value

            for d in crop_dirs:
                if gt.shape != tuple(image_size(d)):
                    print(f"SIZE MISMATCH, skipped: {d} mask {gt.shape} vs image {image_size(d)}")
                    continue
                Image.fromarray(gt).save(d / "mask_gt.png")
                written += 1

    print(f"wrote mask_gt.png into {written} crop folders")
    if missing:
        print("no CVAT mask found for:", ", ".join(missing))


if __name__ == "__main__":
    main()
