# Sentinel-1 Land Use and Land Cover Segmentation

A U-Net that classifies every 10 m pixel of a **Sentinel-1 radar** image into one of six land cover classes, and a Streamlit prototype that maps a location over **12 consecutive months** and reports how the class shares change.

Only Sentinel-1 data (VV and VH backscatter) is used as model input. ESA WorldCover 2021 is used only as training labels.

---

## Project structure

```
s1-lulc/
├── config.py                  # class scheme, regions, compositing rules (shared everywhere)
├── s1_data.py                 # Earth Engine access: monthly S1 composites, WorldCover, grids, features
├── lulc_model.py              # U-Net, checkpoint I/O, inference, smoothing, area statistics
├── train_utils.py             # datasets, loss, metrics, train and eval loops
├── app.py                     # Streamlit prototype
├── notebooks/
│   ├── 01_build_dataset.ipynb # download S1 + WorldCover, QA plots, normalisation stats
│   ├── 02_train.ipynb         # training with curves (loss, mIoU, accuracy, per-class IoU)
│   └── 03_evaluate.ipynb      # test metrics, confusion matrix, per-month accuracy, stability
├── data/                      # created by notebook 01 (not in git, see dataset link)
├── models/                    # best_model.pth, training history, figures
├── requirements.txt
└── README.md
```

---

## Required accounts

| Account | Why | Cost |
|---|---|---|
| **Google account + Google Cloud project registered for Earth Engine** | Download Sentinel-1 and WorldCover (dataset building **and** the app) | Free for noncommercial and academic use |
| Google Colab or Kaggle (optional) | GPU for training | Free tier is enough |

No API keys are needed. Authentication is done once through the browser.

### Earth Engine setup (once)
1. Go to <https://code.earthengine.google.com/register>, choose **noncommercial / academic use**, and create or select a Google Cloud project. Note its **project ID**.
2. Install the requirements (below), then run:
   ```bash
   earthengine authenticate
   ```
3. Use the project ID wherever `EE_PROJECT` is asked for (notebook 01 and the app sidebar), or set it once:
   ```bash
   export EE_PROJECT=your-project-id        # macOS or Linux
   set EE_PROJECT=your-project-id           # Windows cmd
   ```

---

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training install the CUDA build of PyTorch from <https://pytorch.org> (Colab and Kaggle already have it).

---

## Reproducing the model

### 1 · Build the dataset (`notebooks/01_build_dataset.ipynb`)
- 53 regions across Europe, each 7.68 km × 7.68 km (768 × 768 px at 10 m)
- Monthly median Sentinel-1 composites for all 12 months of 2021 (IW, VV + VH, both orbit directions, swath edges masked)
- Labels: ESA WorldCover v200 (2021), merged to 6 classes
- Split **by region**: 40 train, 7 validation, 6 test (Mannheim is in test, so the demo city is unseen)
- Output size ≈ 1.5 GB. Download time depends on Earth Engine load; the cell is resumable.

**Dataset download:** `<add your Google Drive or Zenodo link here>`

### 2 · Train (`notebooks/02_train.ipynb`)
U-Net with ResNet-34 encoder (ImageNet-pretrained), inputs VV, VH and VV − VH (dB, standardised). Loss: class-weighted cross-entropy + 0.5 × Dice. Random 256 × 256 crops with rotations and flips, AdamW with OneCycle learning rate, mixed precision on GPU. The checkpoint with the best validation mIoU is saved to `models/best_model.pth`.

On Colab or Kaggle: upload the project folder with `data/`, uncomment the first cell, run all.
On CPU the notebook switches to a tiny smoke-test configuration automatically.

**Trained weights:** `<add link to best_model.pth here>` → place in `models/`

### 3 · Evaluate (`notebooks/03_evaluate.ipynb`)
Predicts the full test regions for all 12 months and reports overall accuracy, mIoU, per-class IoU, precision, recall, F1, the confusion matrix, accuracy by month and temporal stability (share of pixels that flip class between months although the reference label is constant).

---

## Running the prototype

```bash
streamlit run app.py
```

1. Enter the Earth Engine project ID in the sidebar (or set `EE_PROJECT`).
2. Enter latitude, longitude, start year and start month. The 12-month window must end by the last complete month.
3. Press **Run analysis**.

The app then
1. downloads 12 monthly Sentinel-1 median composites for a square area around the point (2.56, 5.12 or 7.68 km side),
2. runs the U-Net on each month → 12 land cover maps,
3. computes for every month the **share of the area per class (%)**, the **month-to-month change** and the **total change from the first to the last month**, both in **percentage points (pp)**,
4. shows the 12 maps in a grid, a share-over-time chart, the radar inputs, and optionally WorldCover 2021 for comparison.

Maps (PNG) and statistics (CSV) can be downloaded.

*Percentage points:* if cropland covers 40 % in January and 43 % in February, the change is +3 pp (not +7.5 %).

**Example for the screen recording:** Mannheim, latitude 49.4875, longitude 8.4660, start May 2023.

---

## Class scheme

| Class | WorldCover classes merged |
|---|---|
| Tree cover | Tree cover, Mangroves |
| Shrub & grassland | Shrubland, Grassland, Herbaceous wetland, Moss and lichen |
| Cropland | Cropland |
| Built-up | Built-up |
| Bare / sparse | Bare or sparse vegetation |
| Water | Permanent water bodies |
| *(ignored)* | Snow and ice, no data |

Classes that radar cannot reliably distinguish (for example shrubland and grassland) were merged to keep the task well-posed.

---

## Known limitations

- **Seasonal effects look like change.** Radar responds to crop growth, harvest, soil moisture, leaf-off forest and wet snow. Because the labels are a single annual map, part of the month-to-month change the app reports is model response to these seasonal signals, not real land cover change. Notebook 03 quantifies this noise level; the sidebar option *3-month majority smoothing* reduces it.
- **Cropland vs grassland** is the hardest pair for radar alone.
- **No terrain flattening.** `COPERNICUS/S1_GRD` is ortho-corrected but not radiometrically terrain-flattened, so steep slopes can be misclassified.
- **Coverage changes over time.** Sentinel-1B failed in December 2021, so 2022 to 2024 composites contain fewer scenes than 2021 (Sentinel-1C has been operational since 2025). The app warns when a month has incomplete coverage.
- **Labels are model-derived.** WorldCover itself has errors; the metrics measure agreement with WorldCover, not with ground truth.
- Trained on European regions; performance elsewhere (tropics, deserts) is not validated.

---

## Data sources and licences
- Sentinel-1 GRD: Copernicus Sentinel data 2015 onwards, processed by ESA; accessed through Google Earth Engine.
- ESA WorldCover 10 m 2021 v200 (CC BY 4.0), Zanaga et al. 2022, doi:10.5281/zenodo.7254221.
