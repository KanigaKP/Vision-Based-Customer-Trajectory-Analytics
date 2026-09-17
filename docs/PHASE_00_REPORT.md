# Phase 0 Report — Project Foundation & Environment Validation

**Date:** 2026-09-17  
**Status:** ✅ PASS

---

## 1. Phase Objective

Establish the project foundation: verify the development environment, create the project directory structure, set up a Python virtual environment, install and validate foundational dependencies, initialise documentation, and create a reproducible baseline for subsequent phases.

---

## 2. Environment Information

| Item | Value |
|---|---|
| **OS** | Microsoft Windows 11 Home Single Language |
| **OS Version** | 10.0.26200, Build 26200 |
| **System Type** | x64-based PC |
| **Python Version** | 3.14.6 (tags/v3.14.6:c63aec6, Jun 10 2026, 10:26:10) [MSC v.1944 64 bit (AMD64)] |
| **Python Executable** | `C:\Users\sarca\AppData\Local\Programs\Python\Python314\python.exe` |
| **pip Version** | 26.2.1 (inside venv) |
| **Git Version** | 2.53.0.windows.2 |
| **Virtual Environment** | `venv/` (Python built-in `venv` module) |

---

## 3. Project Structure

```
Vision-Based-Customer-Trajectory-Analytics/
├── .gitignore
├── config.yaml
├── README.md
├── requirements.txt
│
├── data/
│   ├── mot17/.gitkeep
│   ├── merl/.gitkeep
│   └── controlled/.gitkeep
│
├── src/
│   ├── __init__.py
│   ├── detection/__init__.py
│   ├── tracking/__init__.py
│   ├── trajectory/__init__.py
│   ├── spatial/__init__.py
│   ├── analytics/__init__.py
│   ├── recommendations/__init__.py
│   └── database/__init__.py
│
├── dashboard/
│   └── __init__.py
│
├── tests/
│   └── __init__.py
│
├── outputs/
│   ├── videos/.gitkeep
│   ├── trajectories/.gitkeep
│   ├── heatmaps/.gitkeep
│   └── reports/.gitkeep
│
├── docs/
│   ├── PHASE_00_REPORT.md    (this file)
│   └── DECISIONS.md
│
└── venv/                      (git-ignored)
```

The structure matches the specification exactly. No modifications were required.

---

## 4. Virtual Environment Status

| Item | Value |
|---|---|
| **Tool** | `python -m venv venv` |
| **Location** | `venv/` (project root) |
| **Python in venv** | 3.14.6 |
| **pip in venv** | 26.2.1 |
| **Activation (PowerShell)** | `.\venv\Scripts\Activate.ps1` |
| **Activation (cmd)** | `.\venv\Scripts\activate.bat` |
| **Status** | ✅ Created and verified |

---

## 5. Dependencies Installed

### Direct dependencies (requirements.txt)

| Package | Installed Version |
|---|---|
| numpy | 2.5.3 |
| opencv-python | 5.0.0.93 |
| pandas | 3.0.5 |
| matplotlib | 3.11.2 |
| pyyaml | 6.0.3 |

### Transitive dependencies (auto-installed)

| Package | Version |
|---|---|
| contourpy | 1.4.0 |
| cycler | 0.12.1 |
| fonttools | 4.65.0 |
| kiwisolver | 1.5.1 |
| packaging | 26.3 |
| pillow | 12.3.0 |
| pyparsing | 3.3.2 |
| python-dateutil | 2.9.0.post0 |
| six | 1.17.0 |
| tzdata | 2026.4 |

### Deferred dependencies (future phases)

| Package | Phase | Purpose |
|---|---|---|
| ultralytics | Phase 1 | YOLO11 person detection |
| streamlit | Phase 5+ | Dashboard |
| scipy | As needed | Spatial analysis |

---

## 6. Import / Installation Verification

All verification was performed using the venv Python (`.\venv\Scripts\python.exe`).

```
  [OK]   numpy                -> numpy 2.5.3
  [OK]   opencv-python        -> cv2 5.0.0
  [OK]   pandas               -> pandas 3.0.5
  [OK]   matplotlib           -> matplotlib 3.11.2
  [OK]   pyyaml               -> yaml 6.0.3
  [OK]   sqlite3 (stdlib)     -> sqlite3 stdlib

=== ALL IMPORTS PASSED ===
```

**Result: ✅ ALL IMPORTS PASSED**

---

## 7. Files Created

| File | Purpose |
|---|---|
| `README.md` | Project overview and quick-start guide |
| `requirements.txt` | Phase 0 Python dependencies |
| `config.yaml` | Project configuration stub |
| `.gitignore` | Git exclusion rules |
| `docs/PHASE_00_REPORT.md` | This report |
| `docs/DECISIONS.md` | Project decision log |
| `src/__init__.py` | Package marker |
| `src/detection/__init__.py` | Package marker |
| `src/tracking/__init__.py` | Package marker |
| `src/trajectory/__init__.py` | Package marker |
| `src/spatial/__init__.py` | Package marker |
| `src/analytics/__init__.py` | Package marker |
| `src/recommendations/__init__.py` | Package marker |
| `src/database/__init__.py` | Package marker |
| `dashboard/__init__.py` | Package marker |
| `tests/__init__.py` | Package marker |
| `data/mot17/.gitkeep` | Empty directory placeholder |
| `data/merl/.gitkeep` | Empty directory placeholder |
| `data/controlled/.gitkeep` | Empty directory placeholder |
| `outputs/videos/.gitkeep` | Empty directory placeholder |
| `outputs/trajectories/.gitkeep` | Empty directory placeholder |
| `outputs/heatmaps/.gitkeep` | Empty directory placeholder |
| `outputs/reports/.gitkeep` | Empty directory placeholder |

---

## 8. Decisions Made

All decisions are recorded in [`docs/DECISIONS.md`](DECISIONS.md). Summary:

| ID | Decision |
|---|---|
| DEC-001 | Use Python 3.14.6 (system-installed version) |
| DEC-002 | Use built-in `venv` for virtual environment |
| DEC-003 | Install only foundational packages in Phase 0 |
| DEC-004 | No hard version pins in Phase 0; record actuals |
| DEC-005 | Adopt proposed project structure without modification |
| DEC-006 | Use SQLite (stdlib) as the database |
| DEC-007 | No Docker/cloud/enterprise infrastructure |

---

## 9. Problems Encountered

| # | Problem | Resolution |
|---|---|---|
| 1 | `pip` was not installed in the system Python 3.14 | Bootstrapped via `python -m ensurepip --upgrade` |
| 2 | `pip` command not on PATH | Used `python -m pip` instead of bare `pip` |
| 3 | Unicode arrow character (`→`) caused `UnicodeEncodeError` on Windows cp1252 console | Replaced with ASCII `->` in verification script |

No blocking issues were encountered.

---

## 10. Current Limitations

1. **Python 3.14 compatibility** — Python 3.14 is very recent. Some future dependencies (particularly `ultralytics` for YOLO11) may not yet have pre-built wheels for 3.14. This must be verified at the start of Phase 1. Fallback: install Python 3.12 alongside.
2. **No GPU verification** — GPU/CUDA availability for YOLO11 inference has not been checked. This is deferred to Phase 1.
3. **No datasets downloaded** — Dataset directories are created but empty. Downloads are scoped to Phase 1+.
4. **requirements.txt is unpinned** — Exact versions are recorded in this report. A `pip freeze` lock file can be generated if needed.

---

## 11. Phase 0 Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | Workspace inspected | ✅ |
| 2 | OS and Python version confirmed | ✅ |
| 3 | Project folder correctly opened | ✅ |
| 4 | Clean project structure created | ✅ |
| 5 | Python virtual environment created | ✅ |
| 6 | Initial dependency/config files created | ✅ |
| 7 | Python environment verified | ✅ |
| 8 | Basic project dependencies import successfully | ✅ |
| 9 | Initial project documentation created | ✅ |
| 10 | Reproducible baseline (Git) established | ✅ |

**Overall: ✅ ALL CRITERIA MET**

---

## 12. Phase 1 Prerequisites

Before starting Phase 1 (YOLO11 Person Detection), the following should be addressed:

1. **Verify `ultralytics` compatibility with Python 3.14** — Check if YOLO11 / Ultralytics has wheels for Python 3.14. If not, install Python 3.12 and recreate the venv.
2. **Check GPU availability** — Run `nvidia-smi` and check CUDA toolkit availability. Decide CPU vs. GPU inference.
3. **Download MOT17 dataset** — At minimum, a subset for initial detection testing.
4. **Download or record controlled test video** — A short clip for pipeline development.
5. **Decide video input format** — Resolution, FPS, codec expectations.
6. **Install ultralytics** — Add to requirements.txt and verify YOLO11 model loading.

---

## Commands Executed (Chronological)

```powershell
# 1. Environment inspection
systeminfo | Select-String "OS Name|OS Version|System Type"
python --version
python -c "import sys; print(f'Path: {sys.executable}'); ..."
git status

# 2. pip bootstrap
python -m ensurepip --upgrade
python -m pip --version

# 3. Virtual environment
python -m venv venv
.\venv\Scripts\python.exe --version
.\venv\Scripts\python.exe -m pip --version
.\venv\Scripts\python.exe -m pip install --upgrade pip

# 4. Directory structure
# Created via PowerShell New-Item for all data/, src/, dashboard/, tests/, outputs/, docs/

# 5. Dependency installation
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# 6. Verification
.\venv\Scripts\python.exe -c "import numpy, cv2, pandas, matplotlib, yaml, sqlite3; ..."
.\venv\Scripts\python.exe -m pip freeze

# 7. Git
git init
git add .
git status
```
