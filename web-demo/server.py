import os
import io
import base64
import urllib.request
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from segment_anything import sam_model_registry, SamPredictor
import omnicloudmask as ocm

app = FastAPI(title="Cloud Detection Backend API")

# Enable CORS so your NJIT website (web.njit.edu) can send requests to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Load SAM and Predictor ---
checkpoint_path = "sam_vit_b_01ec64.pth"
if not os.path.exists(checkpoint_path):
    print("Downloading SAM model checkpoint (~375 MB)...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        checkpoint_path
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SAM model on device: {device}...")
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)
print("SAM model loaded successfully!")

# --- 2. Helper Functions ---
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

def numpy_to_base64_png(image_array):
    # Convert RGB to BGR for OpenCV encoding
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    success, encoded_img = cv2.imencode(".png", image_bgr)
    if not success:
        return ""
    base64_str = base64.b64encode(encoded_img).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

# --- 3. Prediction Endpoint ---
@app.post("/predict")
async def predict_clouds(
    file: UploadFile = File(...),
    brightness: float = Form(-100.0),
    contrast: float = Form(1.5),
    whiteness_threshold: float = Form(175.0)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"error": "Invalid image file uploaded."}
    
    input_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess image
    adjusted = adjust_brightness_contrast(input_image, brightness, contrast)
    cropped = crop_excess_borders(adjusted)

    # Run SAM Pipeline
    gray_cropped = np.mean(cropped, axis=-1)
    candidate_pixels = np.argwhere(gray_cropped > whiteness_threshold)

    if len(candidate_pixels) > 0:
        step_size = max(1, len(candidate_pixels) // 10)
        selected_coords = candidate_pixels[::step_size][:10]
        input_points = np.array([[col, row] for row, col in selected_coords])
        input_labels = np.ones(len(input_points))

        predictor.set_image(cropped)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        sam_mask = masks[np.argmax(scores)]
    else:
        sam_mask = np.zeros(cropped.shape[:2], dtype=bool)

    # Run OmniCloudMask
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

    # Consensus
    consensus_mask = np.logical_and(sam_mask, omni_mask)

    # Visual overlays (RGB)
    sam_overlay = cropped.copy()
    sam_overlay[sam_mask] = [0, 255, 0]  # Green

    omni_overlay = cropped.copy()
    omni_overlay[omni_mask] = [255, 0, 0]  # Red

    consensus_overlay = cropped.copy()
    consensus_overlay[consensus_mask] = [255, 255, 0]  # Yellow

    # Metrics
    sam_cov = (float(np.sum(sam_mask)) / sam_mask.size) * 100.0
    omni_cov = (float(np.sum(omni_mask)) / omni_mask.size) * 100.0
    consensus_cov = (float(np.sum(consensus_mask)) / consensus_mask.size) * 100.0

    return {
        "status": "success",
        "metrics": {
            "sam_coverage": round(sam_cov, 2),
            "omni_coverage": round(omni_cov, 2),
            "consensus_coverage": round(consensus_cov, 2),
        },
        "images": {
            "consensus": numpy_to_base64_png(consensus_overlay),
            "sam": numpy_to_base64_png(sam_overlay),
            "omni": numpy_to_base64_png(omni_overlay),
        }
    }

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Cloud segmentation backend is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)