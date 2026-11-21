# Public Dataset Evidence Lab

This local, offline page demonstrates outputs from the repository's MERL sample and selected public datasets. The data files live under ignored `data/external/`; they are not committed. Generate the example summaries and figures with:

```powershell
.\venv\Scripts\python.exe scripts/run_public_dataset_examples.py
```

Then open the Streamlit app at `http://localhost:8501/?view=public` or select **Public Dataset Evidence Lab** in the sidebar. The page reads local files only.

## What the examples show

| Source | Venue/context | Evidence shown | What it cannot establish |
| --- | --- | --- | --- |
| MERL project clip | Supermarket | Existing detector/tracker, configured homography, zone, dwell, transition, and density pipeline outputs | Representative spatial coverage; its existing run is sparse and boundary-truncated, and floor calibration is unverified |
| EgoCart sequence 0 | Supermarket, cart-mounted camera | Supplied cart poses and aisle-section labels; counts by labeled section and a fixed 0.5 m highest-count cell within the top section | Person trajectories, shopper identity, shelf-row polygons, product interest, dwell, demand, or placement effectiveness |
| PRCV Site03 scenario 006 | Anonymized retail store | One supplied stereo video and source-provided track annotations | Evaluation of this repository's tracker; shelf-row locations or recommendation outcomes |
| CAVIAR WalkByShop1 | Shopping-centre corridor | Repository YOLO11-n ONNX detection and ByteTrack output on sampled frames | Merchandise-aisle analysis. The configured homography check residual is large, so floor mapping and heatmap are withheld |
| Mall Dataset | Shopping-mall webcam | Supplied image-coordinate head points and aggregate density for 2,000 frames | Person identities, trajectories, dwell, metric floor positions, shelf rows, or product placement |

These are five dataset/video sources when the existing MERL clip is included, but they do **not** represent five distinct shop types. The available contexts are supermarkets, an anonymized retail store, a shopping-centre corridor, and a shopping-mall crowd view. Avoid presenting these as five-store validation.

## Exploratory aisle and position candidate

For EgoCart sequence 0, frames are sampled at 5 FPS from its 15 FPS encoded sequence. Across labeled aisle sections 1–9, the page ranks sampled annotated cart-pose frames, breaking ties by section ID. It chooses the highest-count aisle section and then the highest-count cell on a fixed 0.5 m grid within that section, breaking ties by grid order.

In the current cached run, section 5 has 44 sampled pose frames. The highest-count cell within it is centered at approximately `(-7.45 m, 3.75 m)` and contains 5 sampled poses. These are cart-camera poses in the dataset's coordinate system. They do not mean five shoppers stopped there, and this is not a product-placement recommendation. The dataset does not provide shelf-row polygons or product outcomes.

## CAVIAR calibration gate

The CAVIAR detector/tracker preview is an image-space engineering example. A configured homography using the source's published correspondences yielded a mean check-point residual of approximately 6.78 m and maximum of 11.42 m. The check references are not an independent physical survey, and this large error makes the spatial output unsuitable for interpretation. The page therefore shows no CAVIAR floor heatmap or zone/dwell recommendation.

## Data sources and terms

See [`data/external/DATASET_SOURCES.json`](../data/external/DATASET_SOURCES.json) for source URLs, files, and recorded license notes. In brief:

- [Voxel51 EgoCart](https://huggingface.co/datasets/Voxel51/egocart_videos): research use per dataset card.
- [PRCV Labs multiview retail tracking](https://huggingface.co/datasets/prcvlabs/multiview-retail-tracking): CC BY-NC 4.0 per source card.
- [CAVIAR Shopping Centre](https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/): source states CC BY-SA.
- [CUHK Mall Dataset](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html): source restricts commercial use.

The RetailAction dataset was reviewed as a possible convenience-store source but not fetched: its source describes a roughly 31.9 GB dataset and the available license terms were not sufficiently clear for a local project copy. No source data were altered. Check source terms before redistribution or use beyond the permitted research/demo context.

## Evidence boundary

All displayed counts are sample-specific operational or supplied-annotation statistics. The YOLO11-n ONNX model is pretrained and was not trained by this project. No independent store-floor survey, shelf-row map, controlled product-placement trial, or product outcome labels are available. Layout optimization effectiveness and actionable product-placement advice remain unvalidated and withheld.
