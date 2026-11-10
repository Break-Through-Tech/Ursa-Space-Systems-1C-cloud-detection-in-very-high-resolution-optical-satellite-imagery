⛈️ Cloud Detection & Preprocessing Ensemble — Web Demo

Break Through Tech AI • Ursa Space Systems Challenge

Cloud Detection in Very High Resolution Optical Satellite Imagery

This directory contains the interactive web demonstration and backend inference API for the cloud segmentation ensemble pipeline. The application combines Meta's Segment Anything Model (SAM ViT-B) with OmniCloudMask to produce consensus cloud masks alongside coverage metrics.

🏛️ Architecture Overview

The application is structured into a decoupled client-server architecture:

Visitor Browser (NJIT / Local HTML)
       │
       └── (AJAX POST /predict with image & preprocessing sliders)
       │
       ▼
FastAPI Backend (server.py)
       ├── Image Preprocessing (Brightness, Contrast, Crop)
       ├── SAM Point-Prompt Inference (sam_vit_b_01ec64.pth)
       ├── OmniCloudMask Segmentation
       └── Ensemble Intersection & Coverage Metrics Calculation


Frontend (cloud-segmentation-app.html): Lightweight, standalone HTML5/CSS3/JavaScript interface. Can be hosted statically (e.g. on university web servers like NJIT Highlander AFS) or opened locally in any browser.

Backend (server.py): High-performance FastAPI server exposing a REST API with CORS enabled, returning Base64-encoded visual overlays and metric summaries.

✨ Key Features

Interactive Preprocessing Controls:

Brightness Offset: -150 to +150

Contrast Multiplier: 0.5 to 3.0

Whiteness Prompt Threshold: 100 to 250 for automated SAM candidate point sampling

Multi-Model Mask Overlays:

🟡 Consensus Overlay (Yellow): High-confidence intersection of SAM and OmniCloudMask.

🟢 SAM Overlay (Green): Prompt-guided segmentation from Segment Anything.

🔴 OmniCloudMask Overlay (Red): Specialized cloud feature detection.

Quantitative Metrics: Real-time computation of cloud cover percentages for each individual mask and the consensus union/intersection.

Flexible Networking: Toggle between local development (http://localhost:8000) and live public tunnels (via Cloudflare Tunnel) without editing source code.

🚀 Setup & Installation

1. Prerequisites

Python 3.10, 3.11, or 3.12

[Optional] NVIDIA GPU with CUDA for faster inference (defaults to CPU automatically if unavailable)

2. Install Dependencies

From within this directory, install the required packages:

pip install -r requirements.txt


Or manually:

pip install fastapi uvicorn python-multipart torch opencv-python numpy segment-anything omnicloudmask


Note on Model Weights: On first launch, server.py automatically checks for and downloads Meta's official SAM ViT-B checkpoint (sam_vit_b_01ec64.pth, ~375 MB) if it is not already present in the folder.

💻 Running the App Locally

Step 1: Start the Backend API

Run the server using Python:

python server.py


(On Windows using the Python Launcher: py server.py)

The server will initialize the model and listen on port 8000:

Loading SAM model on device: cpu (or cuda)...
SAM model loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)


Step 2: Open the Web Interface

Double-click cloud-segmentation-app.html to open it in your browser, or visit your hosted deployment (e.g. https://web.njit.edu/~cwo3/breakthroughtechApp/cloud-segmentation-app.html).

Verify the status indicator at the top reads Backend online (green dot). Upload a satellite image and click Segment Clouds.

🌐 Running for Remote / Stakeholder Demos

To allow team members, mentors, or reviewers to test the app from their own devices without installing Python:

Keep server.py running in your first terminal.

In a second terminal, start a free Cloudflare Tunnel:

cloudflared tunnel --url http://localhost:8000


Copy the generated public URL:

https://your-tunnel-name.trycloudflare.com


Paste this URL into the Backend API Endpoint field on the web interface. Any visitor to the webpage can now run the segmentation pipeline through your live tunnel.

📡 API Reference

GET /

Health check endpoint.

Response: {"status": "ok", "message": "Cloud segmentation backend is running."}

POST /predict

Performs preprocessing and segmentation on an uploaded satellite image.

Parameters (Form Data):

file: Satellite image file (PNG, JPG, TIFF)

brightness: Float (default: -100.0)

contrast: Float (default: 1.5)

whiteness_threshold: Float (default: 175.0)

Response (JSON):

{
  "status": "success",
  "metrics": {
    "sam_coverage": 32.45,
    "omni_coverage": 28.10,
    "consensus_coverage": 24.85
  },
  "images": {
    "consensus": "data:image/png;base64,...",
    "sam": "data:image/png;base64,...",
    "omni": "data:image/png;base64,..."
  }
}


👥 Contributors

Ursa Space Systems 1C Challenge Team • Break Through Tech AI

Deployed & Hosted on NJIT Highlander AFS (web.njit.edu)