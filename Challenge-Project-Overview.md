# Cloud Detection in Very-High-Resolution Optical Satellite Imagery: Classical Signal Processing vs. Pretrained Deep Learning Models

**Company / Org:** Ursa Space Systems  
**Challenge Advisor:** John Savage  
**AI Studio Coach:** Julio Contreras, Julio.Contreras@breakthroughtech.org   
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Ursa Space Systems
Ursa Space Systems is a global leader in satellite intelligence, leveraging a virtual constellation of SAR and optical sensors to provide high-fidelity insights for commercial and government sectors. The company specializes in transforming massive volumes of geospatial data into actionable intelligence, helping clients monitor economic activity and environmental changes worldwide.

---

## 🎯 The Challenge
### Project Summary
In this project, you will use very-high-resolution (~30–50cm) optical satellite imagery from the Maxar/Vantor Open Data Program and classical image signal-processing techniques (e.g., brightness/saturation thresholding, the dark-channel prior, visible-band haze indices) compared against off-the-shelf pretrained computer-vision segmentation models to build and benchmark methods that automatically identify and mask cloud-covered regions in satellite scenes. This will help our company address the problem of cloud contamination in commercial VHR optical imagery, where clouds obscure ground features and degrade every downstream analytics and modeling product we build.

### Success Criteria
Success is a clear, reproducible comparison rather than a single accuracy number, since there are no ground-truth masks. Concretely: (1) at least two classical methods and one pretrained model implemented and runnable end-to-end in free Colab; (2) evaluation against a small student-curated reference set using standard segmentation metrics — Intersection-over-Union (IoU), precision/recall, F1 — on cloud vs. non-cloud pixels; (3) a documented qualitative error analysis of where each method breaks (thin cloud, smoke, bright non-cloud surfaces). A successful December outcome is a clean notebook and writeup that says, with evidence, which approaches work best on VHR RGB imagery and why — something genuinely useful as an internal reference.

### Stretch Goals
(1) Test cross-scene/cross-event generalization — does a method tuned on a hurricane event hold up on a wildfire-smoke scene?
(2) Bring in the multispectral product (adding NIR) and test whether an extra band improves the classical baselines. 
(3) Fine-tune a small segmentation model (e.g., a compact U-Net) on a student-labeled subset and compare to the zero-shot and classical approaches. 
(4) Build a simple cloud-cover percentage estimator and a "usable imagery" filter as a practical mini-product. 
(5) Extend cloud detection to cloud-shadow detection, which is a related and harder problem.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | Foundations & data | Students get comfortable in Colab and with geospatial Python (rasterio/leafmap), learn to stream and window-read cloud-optimized GeoTIFFs from the Maxar Open Data catalog, and assemble a small working set of VHR scenes deliberately filtered toward higher cloud cover. **Deliverable:** a notebook that loads, displays, and tiles VHR imagery, plus a short written characterization of what clouds, smoke, and cloud-like surfaces (snow, bright rooftops, sand) look like in these scenes. |
| October | Classical baseline | Implement two or three signal-processing cloud-detection baselines in the RGB/visible domain (e.g., a brightness/saturation threshold, the dark-channel prior, a HOT-style index). Build a small hand-labeled or visually-assessed evaluation set so methods can be compared. **Deliverable:** a working classical cloud-mask pipeline plus a qualitative and quantitative comparison of the baselines. |
| November | Pretrained models & comparison | Apply one or more off-the-shelf pretrained segmentation models (e.g., SAM in a zero-shot setup, and/or an existing RGB cloud-segmentation checkpoint) to the same scenes, then run a head-to-head comparison against the classical baselines on the same evaluation set, with attention to failure modes (smoke vs. cloud, bright surfaces misclassified as cloud). **Deliverable:** a final benchmark notebook, results writeup, and a short presentation of findings. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** Maxar Open Data Program / Vantor  
**Format:** Cloud-optimized GeoTIFFs  
**Size:** over 10gb  
**Location:** https://radiantearth.github.io/stac-browser/#/external/maxar-opendata.s3.amazonaws.com/events/catalog.json

### Key Details
- This catalog contains optical data from a number of natural disasters.
- You will need to pick one (or more) disasters to pull imagery from
- Within each disaster's catalog, there are composite images made up of smaller images (click around)
- After clicking on the tiles (smaller images), you should see a preview of the satellite image.

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification,Computer Vision,Deep Learning / Neural Networks,Transfer Learning / Pre-trained Models 

**Recommended Libraries:**
- rasterio, leafmap, pystac-client, fsspec, rio-tiler, geopandas, shapely, numpy, scikit-image, scipy, torch, torchvision, transformers, scikit-learn, matplotlib, lightning

**Evaluation Metrics:**
- Intersection over Union (IoU), precision, recall, and F1, computed per-pixel on cloud vs. non-cloud
- Qualitative error analysis of failure modes: thin cloud, smoke, bright non-cloud surfaces

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [Cloud detection techniques high-level summary](https://www.nature.com/nature-index/topics/l4/cloud-detection-techniques-in-remote-sensing). Look at references for long-form papers

**Technical Tutorials:**
- [Hugging Face cloud detection models](https://huggingface.co/search/full-text?q=cloud+detection&type=model)

**Starter Code for Data Access:**

```
import geopandas as gpd
from pystac_client import Client

# Connect to the Maxar/Vantor Open Data STAC root
catalog_url = "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"
client = Client.open(catalog_url)

# Print all available disaster event collection IDs
print("Available disaster events:")
for collection in client.get_all_collections():
    print(f"- {collection.id}")

# Target a specific event (e.g., Morocco Earthquake or a specific flood event)
# Replacing 'event-id' with an active collection name string listed above
collection_id = "morocco-earthquake-sep-2023" 
collection = client.get_collection(collection_id)
```

**Other:**
- [Helpful video on U-Net architecture](https://www.youtube.com/watch?v=NhdzGfB1q74). Talks about image generation, but relevant.

*Feel free to explore beyond these*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* Reach out to me on Discord (or by email)
  * Feel free to message individually as well 
* Note: I will try to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.

**Recommended free coding / collaboration tools**
* [Google Colab](https://colab.research.google.com/)
* Github (you are here)

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

Looking forwards to meeting you all and working this semester!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
