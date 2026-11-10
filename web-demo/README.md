⛈️ Cloud Detection & Preprocessing Ensemble — Web Demo<p align="center">
<img src="https://img.shields.io/badge/Break%20Through%20Tech-AI%20Studio-blueviolet?style=for-the-badge" alt="Break Through Tech AI">
<img src="https://img.shields.io/badge/Partner-Ursa%20Space%20Systems-0052cc?style=for-the-badge" alt="Ursa Space Systems">
<img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/Deployment-NJIT%20Highlander%20AFS-c90000?style=for-the-badge" alt="NJIT Deployment">
</p>📖 Project OverviewThis repository subfolder provides an interactive, end-to-end web demonstration and REST inference API for our optical satellite cloud detection pipeline.By combining Meta's Segment Anything Model (SAM ViT-B) with OmniCloudMask, the ensemble performs automated point-prompt sampling across preprocessed satellite tiles to generate multi-model segmentation masks, consensus intersections, and accurate cloud coverage metrics.[!NOTE]Challenge Objective: Cloud Detection in Very High Resolution (VHR) Optical Satellite Imagery in partnership with Ursa Space Systems and Break Through Tech AI.🏛️ System ArchitectureThe demo uses a decoupled client-server architecture to deliver full deep learning inference while remaining accessible from any web browser without server memory caps:┌────────────────────────────────────────────────────────────────────────┐
│                          CLIENT FRONTEND                               │
│      Hosted on NJIT Highlander AFS (web.njit.edu) or Localhost         │
│             File: web-demo/cloud-segmentation-app.html                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ HTTP POST (Multipart Form-Data)
                                    │ Image + Preprocessing Sliders
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND API                             │
│                  File: web-demo/server.py                              │
│                                                                        │
│   1. Preprocessing Pipeline (Brightness, Contrast, Border Crop)        │
│   2. SAM Automated Point Prompting (sam_vit_b_01ec64.pth)              │
│   3. OmniCloudMask Specialized Inference                               │
│   4. Consensus Intersection (Logical AND)                              │
│   5. Surface Cloud Coverage Metrics Calculation                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ JSON Response
                                    │ Metrics + Base64 PNG Overlays
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     INTERACTIVE RESULTS DASHBOARD                      │
│   • Consensus Mask (Yellow)   • SAM Mask (Green)   • Omni Mask (Red)   │
└────────────────────────────────────────────────────────────────────────┘
🎯 Mask ComparisonMask LayerVisual OverlayDescriptionOptimal Use CaseConsensus Mask🟨 YellowLogical intersection where both SAM and OmniCloudMask agreeHigh-precision cloud masking with minimum false positivesSAM Mask🟩 GreenAutomated point prompt segmentation via Meta SAM ViT-BCapturing fine structural boundaries, cirrus clouds, and edgesOmniCloudMask🟥 RedSpecialized deep learning model for cloud feature detectionReliable macro cloud segmentation across high-albedo terrain🎛️ Preprocessing ControlsParameterSlider RangeDefaultFunctionBrightness Offset-150 to +150-100Suppresses ground albedo and darkens surface terrainContrast Multiplier0.5 to 3.01.5Sharpens cloud texture gradients against ground backgroundsWhiteness Threshold100 to 250175Sets grayscale pixel threshold for candidate SAM point generation⚡ Quickstart: Local Setup1. Clone & Navigategit clone https://github.com/Break-Through-Tech/Ursa-Space-Systems-1C-cloud-detection-in-very-high-resolution-optical-satellite-imagery.git
cd Ursa-Space-Systems-1C-cloud-detection-in-very-high-resolution-optical-satellite-imagery/web-demo
2. Install Dependenciespip install -r requirements.txt
(On Windows using the Python Launcher: py -m pip install -r requirements.txt)[!TIP]
Automatic Model Retrieval: On its first run, server.py will automatically download Meta's SAM checkpoint (sam_vit_b_01ec64.pth, ~375 MB). No manual download required.3. Launch the Backend Serverpython server.py
(Windows: py server.py)Once loaded, Uvicorn will confirm the server is running:Loading SAM model on device: cpu (or cuda)...
SAM model loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
4. Launch the FrontendDouble-click cloud-segmentation-app.html to open it in your browser, or visit our live NJIT hosted page:
https://web.njit.edu/~cwo3/breakthroughtechApp/cloud-segmentation-app.html🌐 Stakeholder & Remote Demos (Cloudflare Tunnel)To allow teammates, professors, or partners at Ursa Space Systems to test the AI live from their own devices without installing Python:# 1. Start the backend in terminal 1
py server.py

# 2. In a second terminal, open a free Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8000
[!IMPORTANT]
Copy the output tunnel URL (e.g. https://xxxx.trycloudflare.com) and paste it into the Backend API Endpoint box at the top of the web application. The status indicator will turn green, and anyone in the world can test the segmentation live!📡 REST API ReferencePOST /predictUpload an image and run the full ensemble pipeline.Request (Multipart Form-Data)file: Satellite image file (.png, .jpg, .jpeg, .tif)brightness: Float (e.g. -100.0)contrast: Float (e.g. 1.5)whiteness_threshold: Float (e.g. 175.0)Response (JSON){
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
<p align="center">
<sub>Developed for the <strong>Break Through Tech AI Studio</strong> in collaboration with <strong>Ursa Space Systems</strong>.<br>Hosted on NJIT Highlander AFS &bull; Fall 2026</sub>
</p>