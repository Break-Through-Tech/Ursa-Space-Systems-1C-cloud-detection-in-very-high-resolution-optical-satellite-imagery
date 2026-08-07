---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff and CAs only — remove before sharing with students)*

### Technical Vetting
| Check | Status | Notes |
| :--- | :--- | :--- |
| Python Compatibility | 🟢 | The project explicitly mentions 'Geospatial Python' libraries (rasterio, leafmap) and 'computer-vision segmentation models', which are well-supported within the Python ecosystem. Google Colab provides a Python environment conducive to these libraries. |
| Data Readiness | 🟡 | While the data is described as 'cloud-optimized GeoTIFFs,' very high-resolution satellite imagery often requires substantial preprocessing. Tasks like mosaicking, resampling, projection handling, and potentially atmospheric correction could be time-consuming. The size estimate also indicates potential challenges in download and processing. |
| Resource Check | 🟢 | The project explicitly states 'Google Colab free tier' and no specialized hardware. The specified models (SAM, compact U-Net) are generally accessible and runnable within this environment, though performance may vary. |

### Internal Scores
- **Student Fit Score:** 7/10
- **Technical Depth Score:** 8/10
- **Overall Recommendation:** REVISE

### Advisor Feedback Draft
This project presents a compelling opportunity to apply classical signal processing and modern CV models to a real-world problem in satellite imagery. 

Technical Adjustments:
1. Data Ingestion & Preprocessing Focus: To mitigate potential data volume issues, consider providing fellows with pre-selected, smaller subsets of VHR imagery (e.g., 5-10 representative scenes) that are already organized and tiled. This shifts the focus from data wrangling to model benchmarking. 
2. Model Scope Refinement: Explicitly define the 'off-the-shelf' pretrained models to be used. For instance, specify one or two well-documented segmentation models (e.g., a standard U-Net implementation, or a lightweight SAM variant) to prevent fellows from spending excessive time on model selection or environment setup.

Call to Action: We recommend proceeding with this project, contingent on refining the data access strategy and clearly scoping the specific pretrained models to ensure feasibility within the 12-week timeframe.

---

# Cloud Detection in Very-High-Resolution Optical Satellite Imagery: Classical Signal Processing vs. Pretrained Deep Learning Models

**Company / Org:** Ursa Space Systems  
**Challenge Advisor:** John Savage, [Email address]  
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
- [Brief description of what's in the data]
- [Any known limitations or preprocessing needed]
- [Link to data dictionary or documentation, if available]

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification,Computer Vision,Deep Learning / Neural Networks,Transfer Learning / Pre-trained Models 

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- [e.g., Accuracy, Precision/Recall, RMSE, BLEU score]

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
