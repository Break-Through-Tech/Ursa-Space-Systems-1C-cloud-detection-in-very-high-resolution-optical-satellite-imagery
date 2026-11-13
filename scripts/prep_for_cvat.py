"""
Step 1 of 2: prep OCM draft masks for CVAT.

Input  (your Drive layout):  ROOT/<scene>/crop_XX/image.tif + mask_draft.tif

Output:
  ROOT/<scene>/crop_XX/image.png      -> saved next to image.tif (same pixels, PNG copy)
  OUT/images/<scene>__crop_XX.png     -> renamed copies, drag ALL of these into one CVAT task
  OUT/cvat_masks.zip                  -> Actions > Upload annotations > "Segmentation mask 1.1"
  OUT/manifest.csv                    -> which upload name belongs to which crop folder
                                         (apply_cvat_masks.py uses it to put masks back)

OUT/images is only for uploading to CVAT; you can delete it after the upload.

OCM classes: 0 clear, 1 thick, 2 thin, 3 shadow.
Our rule: thin cloud counts as cloud, so thick+thin -> cloud; clear+shadow -> background.

Colab usage:
    from google.colab import drive; drive.mount('/content/drive')
    !pip install -q tifffile imagecodecs
    !python prep_for_cvat.py "/content/drive/MyDrive/<your OCM folder>" /content/cvat_upload
"""
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def load_rgb(path):
    img = tifffile.imread(path)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)  # bands-first -> bands-last
    img = img[..., :3]
    if img.dtype != np.uint8:  # stretch 16-bit to 8-bit
        lo, hi = np.percentile(img, (1, 99))
        img = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return img


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "images").mkdir(parents=True, exist_ok=True)

    uploaded = []  # unique names sent to CVAT
    rows = []      # (name, crop folder relative to root)
    zpath = out / "cvat_masks.zip"

    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("labelmap.txt",
                   "# label:color_rgb:parts:actions\n"
                   "background:0,0,0::\n"
                   "cloud:255,0,0::\n")

        for img_path in sorted(root.rglob("image.tif")):
            crop_dir = img_path.parent
            mask_path = crop_dir / "mask_draft.tif"
            if not mask_path.exists():
                print("skip (no mask_draft.tif):", crop_dir)
                continue

            name = f"{crop_dir.parent.name}__{crop_dir.name}"
            rows.append((name, crop_dir.relative_to(root).as_posix()))

            # PNG copy next to the original, in every crop folder (duplicates included)
            img = load_rgb(img_path)
            Image.fromarray(img).save(crop_dir / "image.png")

            if name in uploaded:  # duplicate copy of a crop already sent to CVAT
                continue
            uploaded.append(name)

            Image.fromarray(img).save(out / "images" / f"{name}.png")

            m = tifffile.imread(mask_path)
            rgb = np.zeros((*m.shape, 3), np.uint8)
            rgb[np.isin(m, (1, 2))] = (255, 0, 0)
            tmp = out / "_tmp.png"
            Image.fromarray(rgb).save(tmp)
            z.write(tmp, f"SegmentationClass/{name}.png")
            tmp.unlink()

        z.writestr("ImageSets/Segmentation/default.txt", "\n".join(uploaded) + "\n")

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cvat_name", "crop_dir"])
        w.writerows(rows)

    print(f"{len(rows)} crop folders got image.png")
    print(f"{len(uploaded)} unique crops -> {out / 'images'}  +  {zpath}")
    print(f"manifest -> {out / 'manifest.csv'} (keep this for apply_cvat_masks.py)")


if __name__ == "__main__":
    main()
