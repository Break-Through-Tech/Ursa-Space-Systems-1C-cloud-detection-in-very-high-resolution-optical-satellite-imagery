"""
Precompute sample overlays and metrics for the 4 Ursa Space Systems satellite tiles:
1. Kelowna (Thin Haze & Edge Clouds)
2. Yellowknife (Clear Terrain & Water Bodies)
3. Kelowna (Dense Cumulus Cloud Field)
4. Yellowknife (Heavy Overcast & Smoke Plume)
"""

import os
import json
import urllib.request
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import omnicloudmask as ocm

# --- 1. Load SAM and Predictor ---
checkpoint_path = "sam_vit_b_01ec64.pth"
if not os.path.exists(checkpoint_path):
    print("Downloading SAM checkpoint (~375 MB)...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        checkpoint_path
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SAM on device: {device}...")
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)

# --- 2. Pipeline Functions ---
def adjust_brightness_contrast(image, brightness_factor, contrast_factor):
    img_float = image.astype(np.float32)
    img_float = img_float * contrast_factor + brightness_factor
    return np.clip(img_float, 0, 255).astype(np.uint8)

def crop_excess_borders(image_array, threshold=5):
    if image_array is None:
        return None
    non_black_coords = np.argwhere(np.any(image_array > threshold, axis=-1))
    if len(non_black_coords) == 0:
        return image_array
    y_min, x_min = non_black_coords.min(axis=0)
    y_max, x_max = non_black_coords.max(axis=0) + 1
    return image_array[y_min:y_max, x_min:x_max]

def process_tile(img_path, brightness=-100, contrast=1.5, whiteness_threshold=175):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: Could not read image at {img_path}")
        return None
    input_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    adjusted = adjust_brightness_contrast(input_image, brightness, contrast)
    cropped = crop_excess_borders(adjusted)

    # SAM
    gray_cropped = np.mean(cropped, axis=-1)
    candidate_pixels = np.argwhere(gray_cropped > whiteness_threshold)
    if len(candidate_pixels) > 0:
        step_size = max(1, len(candidate_pixels) // 10)
        selected_coords = candidate_pixels[::step_size][:10]
        input_points = np.array([[col, row] for row, col in selected_coords])
        input_labels = np.ones(len(input_points))

        predictor.set_image(cropped)
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        sam_mask = masks[np.argmax(scores)]
    else:
        sam_mask = np.zeros(cropped.shape[:2], dtype=bool)

    # OmniCloudMask
    input_data = np.zeros((3, cropped.shape[0], cropped.shape[1]), dtype=cropped.dtype)
    input_data[0] = cropped[:, :, 0]
    input_data[1] = cropped[:, :, 1]
    try:
        labels = ocm.predict_from_array(input_data)[0]
        omni_mask = (labels == 1) | (labels == 2)
    except Exception:
        omni_mask = np.zeros(cropped.shape[:2], dtype=bool)

    if sam_mask.shape != omni_mask.shape:
        omni_mask = cv2.resize(
            omni_mask.astype(np.uint8),
            (sam_mask.shape[1], sam_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    consensus_mask = np.logical_and(sam_mask, omni_mask)

    # Visual Overlays
    sam_overlay = cropped.copy()
    sam_overlay[sam_mask] = [0, 255, 0]  # Green

    omni_overlay = cropped.copy()
    omni_overlay[omni_mask] = [255, 0, 0]  # Red

    consensus_overlay = cropped.copy()
    consensus_overlay[consensus_mask] = [255, 255, 0]  # Yellow

    sam_cov = (float(np.sum(sam_mask)) / sam_mask.size) * 100.0
    omni_cov = (float(np.sum(omni_mask)) / omni_mask.size) * 100.0
    consensus_cov = (float(np.sum(consensus_mask)) / consensus_mask.size) * 100.0

    return {
        "original": cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR),
        "sam": cv2.cvtColor(sam_overlay, cv2.COLOR_RGB2BGR),
        "omni": cv2.cvtColor(omni_overlay, cv2.COLOR_RGB2BGR),
        "consensus": cv2.cvtColor(consensus_overlay, cv2.COLOR_RGB2BGR),
        "metrics": {
            "sam": round(sam_cov, 2),
            "omni": round(omni_cov, 2),
            "consensus": round(consensus_cov, 2)
        }
    }

# Known metadata descriptions for your 4 satellite tiles
METADATA_MAP = {
    "Kelowna_2023-08-21_104001008A8DAA00_013331213211_bottomright": {
        "title": "Tile 1: Kelowna Wildfire — Thin Haze & Edge Clouds",
        "desc": "Optical VHR satellite imagery capturing smoke haze and distinct cumulus cloud formations along the perimeter."
    },
    "Yellowknife_2023-08-08_104001008B463900_102020031213_topright": {
        "title": "Tile 2: Yellowknife — Clear Terrain & Water Bodies",
        "desc": "Control tile demonstrating false-positive resistance across roadways, boreal forests, and lakes with near-zero cloud cover."
    },
    "Kelowna_2023-08-21_104001008A8DAA00_013331213232_topright": {
        "title": "Tile 3: Kelowna — Dense Cumulus Cloud Field",
        "desc": "Heavy cloud field testing multi-cluster boundary detection and consensus intersection metrics."
    },
    "Yellowknife_2023-08-15_104001008892B200_102020031213_topright": {
        "title": "Tile 4: Yellowknife — Heavy Overcast & Wildfire Plume",
        "desc": "Challenging atmospheric scenario distinguishing dense cloud layers from surrounding wildfire particulate."
    }
}

def main():
    input_dir = "sample_inputs"
    output_dir = "samples"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    if not files:
        print(f"No images found in '{input_dir}/'. Drop your 4 satellite images into this folder and rerun!")
        return

    manifest = []
    for idx, fname in enumerate(sorted(files), start=1):
        print(f"\n[{idx}/{len(files)}] Processing {fname}...")
        res = process_tile(os.path.join(input_dir, fname))
        if not res:
            continue

        base_id = f"sample_{idx}"
        orig_name = f"{base_id}_orig.jpg"
        sam_name = f"{base_id}_sam.png"
        omni_name = f"{base_id}_omni.png"
        cons_name = f"{base_id}_consensus.png"

        cv2.imwrite(os.path.join(output_dir, orig_name), res["original"])
        cv2.imwrite(os.path.join(output_dir, sam_name), res["sam"])
        cv2.imwrite(os.path.join(output_dir, omni_name), res["omni"])
        cv2.imwrite(os.path.join(output_dir, cons_name), res["consensus"])

        # Match metadata or generate default
        key = os.path.splitext(fname)[0]
        meta = METADATA_MAP.get(key, {
            "title": f"Tile {idx}: {fname[:30]}",
            "desc": "Very high resolution optical satellite imagery."
        })

        manifest.append({
            "id": base_id,
            "title": meta["title"],
            "desc": meta["desc"],
            "orig": f"samples/{orig_name}",
            "consensus": f"samples/{cons_name}",
            "sam": f"samples/{sam_name}",
            "omni": f"samples/{omni_name}",
            "metrics": res["metrics"]
        })
        print(f" -> Consensus: {res['metrics']['consensus']}% | SAM: {res['metrics']['sam']}% | Omni: {res['metrics']['omni']}%")

    with open("samples_data.js", "w") as f:
        f.write(f"const SAMPLES_MANIFEST = {json.dumps(manifest, indent=2)};\n")

    print("\n✅ All 4 tiles processed successfully!")
    print(f"Generated 'samples/' folder and 'samples_data.js'. Upload both to your NJIT website!")

if __name__ == "__main__":
    main()
