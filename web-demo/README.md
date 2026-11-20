# Cloud Detection Benchmark: web demo

A small website for our Break Through Tech AI Studio project with Ursa Space Systems (team 1C, Fall 2026). It explains the project, describes the cloud-detection methods we compare, and lets you flip between the original imagery, our hand-checked ground-truth masks and each model's prediction.

The site is plain HTML, CSS and JavaScript, so the main pages need no server or install. An optional Python backend (`server.py`) adds a "Try your own image" mode that runs SAM and OmniCloudMask on an uploaded image.

## Pages

| Page | File | What's on it |
|---|---|---|
| Home | `cloud-segmentation-app.html` | The research question and three example crops that show where cloud detection gets hard (smoke, thin cloud, bright ground) |
| Methods | `methods.html` | The four classical methods and OmniCloudMask, how the ground truth was made, and how the comparison is kept fair |
| Explore crops | `explore.html` | The image viewer: pick a crop and switch between mask layers |
| Team & credits | `about.html` | Who did what, plus data, model and people credits |

The home page keeps the original file name, `cloud-segmentation-app.html`, so existing links to the demo still work.

## Run it locally

You only need Python, which is used here as a simple file server:

    cd web-demo
    python -m http.server 8000

Then open <http://localhost:8000/cloud-segmentation-app.html>.

You can also double-click `cloud-segmentation-app.html`, but some browsers block parts of a page opened straight from disk, so the server is more reliable.

The page fonts load from Google Fonts. Offline, the site falls back to standard system fonts and still works.

## File structure

    web-demo/
    ├── cloud-segmentation-app.html   # Home page
    ├── methods.html                  # Methods page
    ├── explore.html                  # Image viewer (sample gallery + "Try your own image")
    ├── about.html                    # Team & credits
    ├── styles.css                    # Shared styles for all pages
    ├── site.js                       # Shared script (fade-in on scroll)
    ├── samples_data.js               # SAM + OCM ensemble tiles: list and cloud-cover numbers
    ├── gt_samples_data.js            # Ground-truth crops: list and cloud-cover numbers
    ├── generate_samples.py           # Builds the SAM + OCM ensemble tiles
    ├── generate_gt_samples.py        # Builds the ground-truth crops
    ├── server.py                     # Optional FastAPI backend for "Try your own image"
    ├── requirements.txt              # Python packages for the backend
    └── samples/
        ├── sample_<n>_{orig,sam,omni,consensus}.*    # SAM + OCM ensemble tiles
        ├── gt_<crop>_{orig,gt,omni}.jpg              # Ground-truth crops
        └── hero_val_39_c1.jpg                        # Home page banner (crop val_39_c1)

When you host the site, upload everything except the Python files.

## The image viewer

The **Explore crops** page has two sets of samples.

### Hand-checked ground truth vs. OmniCloudMask

Six crops whose masks were drafted with OmniCloudMask and then fixed by hand in CVAT (see the main [README](../README.md#ground-truth-masks)). They were picked to show a range of cases.

| Crop | What it shows | Ground truth cloud | OCM cloud |
|---|---|---:|---:|
| `test_03_c1` | Broken cloud over terrain | 76.2% | 92.9% |
| `val_39_c1` | Half cloud, half ground | 51.2% | 48.8% |
| `test_01_c1` | Wildfire smoke (89.6% smoke in the ground truth) | 0.8% | 90.4% |
| `val_17_c1` | Cloud and smoke together (58.1% smoke) | 27.6% | 85.6% |
| `test_05_c2` | Clear ground that OCM calls thin cloud | 0% | 100% |
| `test_12_c1` | Clear ground | 0% | 0% |

Layers: **Original**; **Ground truth** (cloud in blue, smoke in orange, unsure in grey); and **OmniCloudMask** (red, with thick and thin cloud merged). SAM wasn't run on these crops, so the SAM and Consensus tabs are greyed out.

To change which crops are shown, edit the `CHOSEN` list in `generate_gt_samples.py` and run:

    python generate_gt_samples.py --crops <path to ursaspace1C_data/crops> --ocm <path to _ocm_predictions>

It rewrites `gt_samples_data.js` and the `samples/gt_*.jpg` images.

### SAM + OmniCloudMask ensemble

Four tiles run through Chidera's ensemble: SAM (Segment Anything, ViT-B) is prompted from bright, white pixels, and its mask is combined with OmniCloudMask's.

| Sample | Title on the site | SAM | OCM | Consensus |
|---|---|---:|---:|---:|
| `sample_1` | Tile 1: Kelowna Wildfire, Thin Haze & Edge Clouds | 12.57% | 16.96% | 10.34% |
| `sample_2` | Tile 3: Kelowna, Dense Cumulus Cloud Field | 99.94% | 89.47% | 89.41% |
| `sample_3` | Tile 2: Yellowknife, Clear Terrain & Water Bodies | 84.52% | 0% | 0% |
| `sample_4` | Tile 4: Yellowknife, Heavy Overcast & Wildfire Plume | 99.28% | 95.84% | 95.14% |

Layers: **Consensus** (yellow, where SAM and OCM agree), **SAM** (green) and **OmniCloudMask** (red). There are no hand-checked masks for these tiles, so the Ground truth tab is greyed out. Numbers are the share of the tile marked as cloud.

To rebuild these tiles, put the source images in a `sample_inputs/` folder and run `python generate_samples.py`. It rewrites `samples_data.js` and the overlays in `samples/`.

## Optional: "Try your own image" backend

This mode sends an uploaded image to `server.py`, which runs SAM and OmniCloudMask and returns the masks. It isn't needed for any other part of the site.

1. Install the packages (Python 3.10–3.12):

        pip install -r requirements.txt

   On Windows with the Python launcher: `py -m pip install -r requirements.txt`.

2. Start the server:

        python server.py

   On first run it downloads the SAM checkpoint (`sam_vit_b_01ec64.pth`, about 375 MB). When it's ready it prints `Uvicorn running on http://0.0.0.0:8000`. Without a GPU, each image takes about 30 seconds.

3. On the **Explore crops** page, choose **Try your own image**. The backend address box defaults to `http://localhost:8000`.

### Letting others try it (Cloudflare Tunnel)

To let someone use your running backend from their own device:

1. Start `server.py` as above.
2. In a second terminal: `cloudflared tunnel --url http://localhost:8000`
3. Paste the `https://….trycloudflare.com` address it prints into the backend address box.

This only works while your computer and the server are running.

### API

`POST /predict` takes multipart form data:

- `file`: the image (`.png`, `.jpg`, `.jpeg`, `.tif`)
- `brightness`: float, default `-100.0`
- `contrast`: float, default `1.5`
- `whiteness_threshold`: float, default `175.0`

It returns:

    {
      "status": "success",
      "metrics": { "sam_coverage": 32.45, "omni_coverage": 28.10, "consensus_coverage": 24.85 },
      "images": {
        "consensus": "data:image/png;base64,...",
        "sam": "data:image/png;base64,...",
        "omni": "data:image/png;base64,..."
      }
    }

## Hosting

A live copy is on Chidera's school web space: <https://web.njit.edu/~cwo3/breakthroughtechApp/cloud-segmentation-app.html>. It may be behind this branch until the new files are uploaded there.

## Known issues

- On the ensemble tiles, the "Tile" numbers in the titles don't match the sample order (`sample_2` is titled Tile 3, and `sample_3` is titled Tile 2).
- `sample_3` is described as clear terrain, but SAM marks 84.5% of it as cloud. That's a real SAM failure, but the description doesn't mention it.
- `sample_3_omni.png` and `sample_3_consensus.png` are the same size to the byte, so they may be duplicates.
- The ensemble tile PNGs are 4–6 MB each, which slows the page down.
- No method scores (IoU, precision, recall, F1) are shown yet. They'll be added once the benchmark has been run.

## Credits

Imagery: [Maxar Open Data Program](https://radiantearth.github.io/stac-browser/#/external/maxar-opendata.s3.amazonaws.com/events/catalog.json) (CC BY-NC 4.0). Models: [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) (Wright et al., 2025, MIT license) and [Segment Anything](https://github.com/facebookresearch/segment-anything) (Meta AI, Apache 2.0). Website and SAM + OCM ensemble by Chidera Okafor; ground-truth crops, Methods page and redesign by Nurbu Lama. See the [main README](../README.md) for the full team.
