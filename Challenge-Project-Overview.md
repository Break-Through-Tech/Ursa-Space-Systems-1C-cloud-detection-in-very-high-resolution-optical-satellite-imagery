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
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Ursa Space Systems
Ursa Space Systems is a global leader in satellite intelligence, leveraging a virtual constellation of SAR and optical sensors to provide high-fidelity insights for commercial and government sectors. The company specializes in transforming massive volumes of geospatial data into actionable intelligence, helping clients monitor economic activity and environmental changes worldwide.

---

## 🎯 The Challenge
### Project Summary
This challenge focuses on developing robust cloud-detection masks for very-high-resolution (VHR) satellite imagery to minimize data degradation in downstream analytics. Students will compare classical signal-processing techniques against modern computer-vision segmentation models to identify and mask cloud-covered pixels. This work is critical to improving the quality of satellite imagery products provided by Ursa Space Systems to its global customer base.

### Success Criteria
Clear, reproducible comparison using standard segmentation metrics — Intersection-over-Union (IoU), precision/recall, F1 — on cloud vs. non-cloud pixels; a documented qualitative error analysis of where each method breaks; a clean notebook and writeup that identifies which approaches work best on VHR RGB imagery.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Exploration & Preprocessing | Load and tile VHR imagery into manageable chunks, while performing initial visualization and descriptive statistics of cloud vs. non-cloud surface signatures. |
| **October** | Feature Engineering & Baseline Modeling | Implement classical signal-processing thresholds (e.g., brightness, haze indices) and establish initial performance baselines for the segmentation models. |
| **November** | Model Optimization & Evaluation | Refine model hyperparameters, execute comparative benchmarking runs across test scenes, and validate segmentation accuracy against ground-truth assumptions. |
| **December** | Insights, Deliverables & Presentation | Synthesize results into a technical report, package the final cloud-masking codebase, and prepare a presentation summarizing findings for stakeholders. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** Maxar Open Data Program / Vantor  
**Format:** Cloud-optimized GeoTIFFs  
**Size:** over 10gb  
**Location:** Accessible via Maxar Open Data Program portal or provided cloud buckets.  

### Key Details
- Very-high-resolution (~30–50cm) optical satellite imagery (cloud-optimized GeoTIFFs) from the Maxar Open Data Program.
- Teams must implement preprocessing routines for handling coordinate reference systems (CRS) and normalizing image tiling to ensure model compatibility.

---

## 🛠️ Suggested Approach
**ML Problem Type:** Computer Vision / Image Segmentation  
**Recommended Libraries:**
- Geospatial Python (rasterio, leafmap)
- Google Colab
- Computer-vision segmentation models (e.g., SAM, compact U-Net)
**Evaluation Metrics:** Intersection-over-Union (IoU), Precision, Recall, and F1-Score for binary mask accuracy.

---

## 📚 Resources to Get Started
The following resources will help your team understand the problem space and potential technical approaches for this project:
**Background Reading:**
- [Maxar Open Data Program Documentation](https://www.maxar.com/open-data)
**Technical Tutorials:**
- [Rasterio Documentation for Geospatial Image Processing](https://rasterio.readthedocs.io/)
**Code Examples:**
- [Segment Anything Model (SAM) GitHub Repository](https://github.com/facebookresearch/segment-anything)

---

## 🤝 How We'll Work Together
**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Via Slack and GitHub Discussions.  
**Response time:** 24–48 hours for non-urgent inquiries.  
**Recommended Tools:**
- **Coding:** Google Colab Free Tier  
- **Collaboration:** GitHub, Notion  
- **Virtual Meetings:** Zoom, Google Meet  

---

## 🚀 Getting Started
1. **Review this overview document** and note any questions for our first meeting.
2. **Begin reviewing the dataset** using the link provided in the Dataset section.
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects).

I'm excited to work with you!

---

## ❓ Questions?
Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).
