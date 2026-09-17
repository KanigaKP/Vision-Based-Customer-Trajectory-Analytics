# Vision-Based Customer Trajectory Analytics for Retail Layout Optimization

A lightweight, explainable, single-camera computer-vision research prototype that converts customer movement trajectories into zone-level retail analytics and layout insights.

> **Status:** Phase 0 — Project Foundation & Environment Validation

## Overview

This project implements an end-to-end pipeline for analysing customer movement in retail environments using recorded video footage:

```
Recorded Video
→ Video Quality Check
→ YOLO11 Person Detection
→ ByteTrack Multi-Object Tracking
→ Track Quality Monitoring
→ Trajectory Extraction
→ Homography / Floor Mapping
→ Manual Zone Definition
→ Footfall / Dwell / Zone Transitions
→ Spatial Heatmaps
→ Zone Classification
→ Explainable Rule-Based Insights
→ SQLite
→ Streamlit Dashboard
→ Evaluation
```

## Scope

This project is a **research prototype** designed for academic evaluation. It does **not** claim:

- Sales/revenue improvement or causal sales impact
- Automatic mathematical layout optimisation
- Real-time CCTV deployment readiness
- Facial recognition or biometric identification
- Cross-session customer identification
- Multi-camera tracking
- Production deployment
- A novel detection or tracking algorithm

## Datasets

| Dataset | Purpose |
|---|---|
| **MOT17** | External tracking benchmark — MOTA / IDF1 evaluation |
| **MERL Shopping Dataset** | Retail-context demonstration (no continuous tracking GT) |
| **Controlled student-recorded footage** | Ground-truth validation for footfall, dwell, zones, transitions |
| **PRISM** | Optional / stretch only |

## Project Structure

```
Vision-Based-Customer-Trajectory-Analytics/
├── data/               # Dataset storage (git-ignored)
│   ├── mot17/
│   ├── merl/
│   └── controlled/
├── src/                # Source code modules
│   ├── detection/
│   ├── tracking/
│   ├── trajectory/
│   ├── spatial/
│   ├── analytics/
│   ├── recommendations/
│   └── database/
├── dashboard/          # Streamlit dashboard
├── tests/              # Unit and integration tests
├── outputs/            # Generated outputs (git-ignored)
│   ├── videos/
│   ├── trajectories/
│   ├── heatmaps/
│   └── reports/
├── docs/               # Documentation and phase reports
├── requirements.txt
├── config.yaml
├── .gitignore
└── README.md
```

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd Vision-Based-Customer-Trajectory-Analytics

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, numpy, pandas, matplotlib; print('All imports OK')"
```

## Documentation

- [Phase 0 Report](docs/PHASE_00_REPORT.md) — Environment setup and validation
- [Decision Log](docs/DECISIONS.md) — Architectural and technical decisions

## License

This project is developed as an academic research prototype for MCA final-year coursework.
