const SAMPLES_MANIFEST = [
  {
    "id": "sample_1",
    "title": "Tile 1: Kelowna Wildfire \u2014 Thin Haze & Edge Clouds",
    "desc": "Optical VHR satellite imagery capturing smoke haze and distinct cumulus cloud formations along the perimeter.",
    "orig": "samples/sample_1_orig.jpg",
    "consensus": "samples/sample_1_consensus.png",
    "sam": "samples/sample_1_sam.png",
    "omni": "samples/sample_1_omni.png",
    "metrics": {
      "sam": 12.57,
      "omni": 16.96,
      "consensus": 10.34
    }
  },
  {
    "id": "sample_2",
    "title": "Tile 3: Kelowna \u2014 Dense Cumulus Cloud Field",
    "desc": "Heavy cloud field testing multi-cluster boundary detection and consensus intersection metrics.",
    "orig": "samples/sample_2_orig.jpg",
    "consensus": "samples/sample_2_consensus.png",
    "sam": "samples/sample_2_sam.png",
    "omni": "samples/sample_2_omni.png",
    "metrics": {
      "sam": 99.94,
      "omni": 89.47,
      "consensus": 89.41
    }
  },
  {
    "id": "sample_3",
    "title": "Tile 2: Yellowknife \u2014 Clear Terrain & Water Bodies",
    "desc": "Control tile demonstrating false-positive resistance across roadways, boreal forests, and lakes with near-zero cloud cover.",
    "orig": "samples/sample_3_orig.jpg",
    "consensus": "samples/sample_3_consensus.png",
    "sam": "samples/sample_3_sam.png",
    "omni": "samples/sample_3_omni.png",
    "metrics": {
      "sam": 84.52,
      "omni": 0.0,
      "consensus": 0.0
    }
  },
  {
    "id": "sample_4",
    "title": "Tile 4: Yellowknife \u2014 Heavy Overcast & Wildfire Plume",
    "desc": "Challenging atmospheric scenario distinguishing dense cloud layers from surrounding wildfire particulate.",
    "orig": "samples/sample_4_orig.jpg",
    "consensus": "samples/sample_4_consensus.png",
    "sam": "samples/sample_4_sam.png",
    "omni": "samples/sample_4_omni.png",
    "metrics": {
      "sam": 99.28,
      "omni": 95.84,
      "consensus": 95.14
    }
  }
];
