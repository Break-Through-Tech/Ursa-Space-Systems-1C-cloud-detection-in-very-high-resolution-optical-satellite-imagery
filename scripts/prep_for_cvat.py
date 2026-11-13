"""
Prep OCM draft masks for CVAT.

Input  (your Drive layout):  ROOT/<scene>/crop_XX/image.tif + mask_draft.tif
Output: OUT/images/<scene>__crop_XX.png   -> drag these into a new CVAT task
        OUT/cvat_masks.zip                -> Actions > Upload annotations > "Segmentation mask 1.1"

OCM classes: 0 clear, 1 thick, 2 thin, 3 shadow.
Our rule: thin cloud counts as cloud, so thick+thin -> cloud; clear+shadow -> background.

Colab usage:
    from google.colab import drive; drive.mount('/content/drive')
    !pip install -q tifffile imagecodecs
    !python prep_for_cvat.py "/content/drive/MyDrive/<your OCM folder>" /content/cvat_upload
"""
import sys, zipfile
from pathlib import Path
import numpy as np, tifffile
from PIL import Image

root, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "images").mkdir(parents=True, exist_ok=True)

names = []
zpath = out / "cvat_masks.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("labelmap.txt",
               "# label:color_rgb:parts:actions\n"
               "background:0,0,0::\n"
               "cloud:255,0,0::\n")
    for img_path in sorted(root.rglob("image.tif")):
        mask_path = img_path.with_name("mask_draft.tif")
        if not mask_path.exists():
            print("skip (no mask):", img_path); continue
        name = f"{img_path.parent.parent.name}__{img_path.parent.name}"
        if name in names:  # duplicate copy of the same crop
            continue
        names.append(name)

        img = tifffile.imread(img_path)
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)          # bands-first -> bands-last
        img = img[..., :3]
        if img.dtype != np.uint8:                   # stretch 16-bit to 8-bit
            lo, hi = np.percentile(img, (1, 99))
            img = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(out / "images" / f"{name}.png")

        m = tifffile.imread(mask_path)
        cloud = np.isin(m, (1, 2))
        rgb = np.zeros((*m.shape, 3), np.uint8)
        rgb[cloud] = (255, 0, 0)
        tmp = out / "_tmp.png"
        Image.fromarray(rgb).save(tmp)
        z.write(tmp, f"SegmentationClass/{name}.png")
        tmp.unlink()

    z.writestr("ImageSets/Segmentation/default.txt", "\n".join(names) + "\n")

print(f"{len(names)} crops -> {out/'images'}  +  {zpath}")
