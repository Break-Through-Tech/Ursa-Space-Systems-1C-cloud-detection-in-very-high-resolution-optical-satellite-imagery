# ⛈️ Cloud Detection & Preprocessing Ensemble — Dual-Mode Web Demo

<p align="center">
  <img src="https://img.shields.io/badge/Break%20Through%20Tech-AI%20Studio-blueviolet?style=for-the-badge" alt="Break Through Tech AI">
  <img src="https://img.shields.io/badge/Partner-Ursa%20Space%20Systems-0052cc?style=for-the-badge" alt="Ursa Space Systems">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Deployment-NJIT%20Highlander%20AFS-c90000?style=for-the-badge" alt="NJIT">
</p>

---

## 📖 Project Overview

This repository subfolder provides an **interactive dual-mode web demonstration** and **REST inference API** for our optical satellite cloud detection pipeline developed in partnership with **Ursa Space Systems** and **Break Through Tech AI**.

The ensemble pipeline integrates **Meta's Segment Anything Model (SAM ViT-B)** with **OmniCloudMask** to perform automated point-prompt sampling across optical satellite tiles, outputting consensus cloud masks alongside quantitative cloud coverage percentages.

> [!NOTE]
> **Challenge Goal:** Automated cloud detection in Very High Resolution (VHR) optical satellite imagery across challenging high-albedo terrain, wildfire smoke haze, and diverse cloud formations.

---

## 🌓 Dual-Mode Application Architecture

The application is engineered with a **progressive enhancement architecture** featuring two operational modes:

    +------------------------------------------------------------------------+
    |                        DUAL-MODE WEB INTERFACE                         |
    |               File: web-demo/cloud-segmentation-app.html               |
    |               Hosted: web.njit.edu (NJIT Highlander AFS)               |
    +-----------------------------------┬------------------------------------+
                                        |
             +--------------------------┴--------------------------+
             |                                                     |
             v                                                     v
    +------------------------------------+   +------------------------------------+
    |    ⚡ MODE 1: INSTANT SHOWCASE     |   |    🔬 MODE 2: LIVE TESTING API     |
    |     (24/7 Always Online, 0 RAM)    |   |      (Custom Uploads & Sliders)    |
    +------------------------------------+   +------------------------------------+
    | • Zero backend dependency          |   | • Connects to FastAPI backend      |
    | • Zero wait time / instant masks   |   | • Real-time SAM + Omni inference   |
    | • 4 precomputed benchmark tiles    |   | • Custom brightness & contrast     |
    | • Data: samples/ & samples_data.js |   | • Endpoint: localhost / Cloudflare |
    +------------------------------------+   +------------------------------------+

1. **⚡ Mode 1: Instant Showcase (Default — Recruiter-Ready 24/7):**
   * Operates completely client-side with zero server dependencies.
   * Allows recruiters, mentors, and teammates to explore 4 benchmark satellite scenarios with instant mask toggling (0.01s latency) and exact precomputed coverage metrics even when the local backend is powered off.
2. **🔬 Mode 2: Live Interactive Testing (Custom Images & Real-Time Tuning):**
   * Upload custom satellite imagery and adjust preprocessing parameters (Brightness, Contrast, Whiteness Threshold) in real-time.
   * Sends multipart requests to the **FastAPI (`server.py`)** engine running locally or via a secure **Cloudflare Tunnel**.

---

## 🗂️ File Structure

    web-demo/
    ├── cloud-segmentation-app.html   # Main dual-mode web interface
    ├── server.py                     # FastAPI backend inference engine
    ├── generate_samples.py           # Precomputation script to process benchmark tiles
    ├── samples_data.js               # Manifest and precalculated metrics for tiles
    ├── samples/                      # Benchmark satellite tiles and overlay masks
    │   ├── sample_1_orig.jpg
    │   ├── sample_1_consensus.png
    │   ├── sample_1_sam.png
    │   └── sample_1_omni.png ...
    ├── requirements.txt              # Python package dependencies
    └── README.md                     # Documentation

---

## 🛰️ Precomputed Benchmark Tiles (Mode 1)

| Benchmark Tile | Location & Event | Scenario Highlight | Key Metric |
| :--- | :--- | :--- | :---: |
| **Tile 1** | **Kelowna Wildfire** (Aug 2023) | Subtle cumulus cloud formation amidst thin wildfire smoke haze | ~24.8% Consensus |
| **Tile 2** | **Yellowknife Control** (Aug 2023) | Negative control over boreal forests, roads, and lakes demonstrating near-zero false alarms | ~0.0% Consensus |
| **Tile 3** | **Kelowna Cloud Field** (Aug 2023) | Dense multi-cluster cumulus cloud field testing complex perimeter segmentation | ~45.2% Consensus |
| **Tile 4** | **Yellowknife Wildfire** (Aug 2023) | Heavy atmospheric smoke plume challenging spectral cloud/smoke separation | ~38.6% Consensus |

---

## 🎯 Mask Overlays Explained

* 🟨 **Consensus Mask (Yellow):** High-confidence logical intersection where **both** SAM and OmniCloudMask agree. Minimizes false positives over high-albedo terrain.
* 🟩 **SAM Mask (Green):** Zero-shot prompt-guided segmentation from Meta's Segment Anything Model. Excels at boundary detail, cloud edges, and wispy cirrus structures.
* 🟥 **OmniCloudMask (Red):** Specialized machine learning model trained specifically on satellite cloud feature detection.

---

## ⚡ Quickstart: Local Setup

### 1. Clone & Navigate
    git clone https://github.com/Break-Through-Tech/Ursa-Space-Systems-1C-cloud-detection-in-very-high-resolution-optical-satellite-imagery.git
    cd Ursa-Space-Systems-1C-cloud-detection-in-very-high-resolution-optical-satellite-imagery/web-demo

### 2. Install Dependencies
    pip install -r requirements.txt

*(On Windows using the Python Launcher: `py -m pip install -r requirements.txt`)*

> [!TIP]
> **Automatic Weight Retrieval:** On first run, `server.py` automatically checks for and downloads Meta's SAM checkpoint (`sam_vit_b_01ec64.pth`, ~375 MB).

### 3. Launch the Backend Server (For Live Mode)
    python server.py

*(Windows: `py server.py`)*

Once loaded, Uvicorn will confirm the server is running on port 8000:

    Loading SAM model on device: cpu (or cuda)...
    SAM model loaded successfully!
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

### 4. Open the Web Application
Double-click `cloud-segmentation-app.html` to open it in your browser, or visit our live NJIT hosted page:
**https://web.njit.edu/~cwo3/breakthroughtechApp/cloud-segmentation-app.html**

---

## 🌐 Public Stakeholder Demos (Cloudflare Tunnel)

To allow teammates, professors, or partners at Ursa Space Systems to test custom imagery from their own devices without installing Python:

1. Start the backend in terminal 1:
    py server.py

2. In a second terminal, open a free Cloudflare Tunnel:
    cloudflared tunnel --url http://localhost:8000

> [!IMPORTANT]
> Copy the output tunnel URL (e.g. `https://xxxx.trycloudflare.com`) and paste it into the **Backend API Endpoint** box under *Live Testing* mode. The status indicator will turn **green**, enabling live cloud segmentation over the public internet.

---

## 🔄 Re-Generating Benchmark Samples

To update or add new satellite tiles to the 24/7 instant showcase:
1. Place raw satellite image tiles into a `sample_inputs/` folder.
2. Run the precomputation script:
    py generate_samples.py
3. The script will re-generate `samples_data.js` and all overlay masks in `samples/`.

---

## 📡 REST API Reference

### `POST /predict`
Upload an image and run the live ensemble pipeline.

#### Request (Multipart Form-Data)
* `file`: Satellite image file (`.png`, `.jpg`, `.jpeg`, `.tif`)
* `brightness`: Float (default: `-100.0`)
* `contrast`: Float (default: `1.5`)
* `whiteness_threshold`: Float (default: `175.0`)

#### Response (JSON)
    {
      "status": "success",
      "metrics": {
        "sam_coverage": 32.45,
        "omni_coverage": 28.10,
        "consensus_coverage": 24.85
      },
      "images": {
        "consensus": "data:image/png;base64,iVBORw0KGgoAAAANSU...",
        "sam": "data:image/png;base64,iVBORw0KGgoAAAANSU...",
        "omni": "data:image/png;base64,iVBORw0KGgoAAAANSU..."
      }
    }

---

<p align="center">
  <sub>Developed for the <strong>Break Through Tech AI Studio</strong> in collaboration with <strong>Ursa Space Systems</strong>.<br>Deployed on NJIT Highlander AFS &bull; Fall 2026</sub>
</p>
