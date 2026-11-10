# ⛈️ Cloud Detection & Preprocessing Ensemble — Web Demo

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

This subfolder contains the interactive web demo and REST inference API for our optical satellite cloud detection pipeline.

Combining **Meta's Segment Anything Model (SAM ViT-B)** with **OmniCloudMask**, the ensemble performs point-prompt sampling across preprocessed satellite tiles to generate multi-model masks, consensus intersections, and coverage metrics.

> [!NOTE]
> **Challenge Objective:** Cloud Detection in Very High Resolution (VHR) Optical Satellite Imagery in partnership with **Ursa Space Systems** and **Break Through Tech AI**.

---

## 🏛️ System Architecture

* **Frontend (`cloud-segmentation-app.html`):** Lightweight HTML5/CSS/JavaScript interface hosted on NJIT Highlander AFS (`web.njit.edu`).
* **Backend (`server.py`):** FastAPI Python server running SAM and OmniCloudMask, returning Base64 overlays and metric summaries.

---

## 🎯 Mask Comparison

| Mask Layer | Color | Description | Optimal Use Case |
| :--- | :---: | :--- | :--- |
| **Consensus Mask** | 🟨 **Yellow** | Logical intersection where **both** models agree | High-precision cloud masking with minimal false positives |
| **SAM Mask** | 🟩 **Green** | Automated point prompt segmentation via Meta SAM ViT-B | Fine structural boundaries, cirrus clouds, and edges |
| **OmniCloudMask** | 🟥 **Red** | Specialized deep learning model for cloud detection | Reliable macro cloud segmentation across terrain |

---

## 🎛️ Preprocessing Controls

| Parameter | Slider Range | Default | Function |
| :--- | :---: | :---: | :--- |
| **Brightness Offset** | `-150` to `+150` | `-100` | Suppresses ground albedo and darkens surface terrain |
| **Contrast Multiplier** | `0.5` to `3.0` | `1.5` | Sharpens cloud texture gradients against backgrounds |
| **Whiteness Threshold** | `100` to `250` | `175` | Sets pixel threshold for candidate SAM point generation |

---

## ⚡ Quickstart: Local Setup

### 1. Install Dependencies
In your terminal, run:
`py -m pip install fastapi uvicorn python-multipart torch opencv-python numpy segment-anything omnicloudmask`

> [!TIP]
> On first run, `server.py` will automatically download Meta's SAM checkpoint (`sam_vit_b_01ec64.pth`, ~375 MB).

### 2. Launch the Backend Server
In your terminal, run:
`py server.py`

### 3. Open the Web App
Open `cloud-segmentation-app.html` in your browser, or visit the live NJIT hosted page:
**https://web.njit.edu/~cwo3/breakthroughtechApp/cloud-segmentation-app.html**

---

## 🌐 Public Demo Setup (Cloudflare Tunnel)

To allow teammates or mentors to test from their own phones or computers without installing Python:

1. Run `py server.py` in your first terminal.
2. In a second terminal, run: `cloudflared tunnel --url http://localhost:8000`
3. Copy the output URL (`https://xxxx.trycloudflare.com`) and paste it into the **Backend API Endpoint** box on the website.
Current live URL: `https://sat-modify-receive-marriage.trycloudflare.com`

---

<p align="center">
  <sub>Developed for the <strong>Break Through Tech AI Studio</strong> in collaboration with <strong>Ursa Space Systems</strong>.<br>Hosted on NJIT Highlander AFS</sub>
</p>
