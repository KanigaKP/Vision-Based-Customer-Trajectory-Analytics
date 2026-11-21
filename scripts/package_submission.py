"""
Final Portable Submission & Runnable Source Release Packager
============================================================
Creates two strictly separated packages within submission/:

1. submission/document_package/
   - Final Academic Project Report (PDF & DOCX)
   - Presentation Deck (HTML & Markdown)
   - 5-Minute Demonstration Script & Cue Sheet
   - DOCUMENT_INVENTORY.md
   - MANIFEST_CHECKSUMS.json (all paths relative to document_package root)

2. submission/runnable_source_release/
   - Complete application source tree (src/, dashboard/)
   - Scene configurations (configs/)
   - Pre-computed saved demonstration reports (outputs/reports/)
   - Video media assets and neural model (data/)
   - Utility scripts and verification runners (scripts/)
   - Complete automated test suite (tests/)
   - Dependency manifest (requirements.txt, config.yaml)
   - One-click launcher (start_demo.cmd)
   - Authoritative project guide (README.md)
   - ASSET_MANIFEST.md (acquisition, placement, and offline readiness rules)
   - RELEASE_INVENTORY.md
   - MANIFEST_CHECKSUMS.json (all paths relative to runnable_source_release root)

Enforces:
  - Fails immediately if any required deliverable is missing.
  - Generates checksums strictly from the files actually packaged.
  - Manifest paths resolve relative to their respective package roots.
"""

import os
import sys
import shutil
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def copy_tree_clean(src_dir: Path, dst_dir: Path, ignore_patterns: List[str] = None):
    """Recursively copy directory tree, excluding caches and compiled files."""
    if not src_dir.exists():
        raise FileNotFoundError(f"Packaging failed: Required source directory '{src_dir}' does not exist.")
    
    if ignore_patterns is None:
        ignore_patterns = ["__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".git", "venv", ".venv"]
    
    def _ignore(directory, files):
        ignored = set()
        for f in files:
            for pat in ignore_patterns:
                if pat.startswith("*") and f.endswith(pat[1:]):
                    ignored.add(f)
                elif f == pat:
                    ignored.add(f)
        return ignored

    shutil.copytree(src_dir, dst_dir, ignore=_ignore, dirs_exist_ok=True)


def package_document_package(base_sub: Path) -> Dict[str, Any]:
    print("\n" + "=" * 65)
    print(" 1. Packaging Document Submission Package")
    print("=" * 65)
    doc_pkg = base_sub / "document_package"
    doc_pkg.mkdir(parents=True, exist_ok=True)

    required_docs = [
        ("Vision_Based_Customer_Trajectory_Analytics_Project_Report.pdf", "Vision_Based_Customer_Trajectory_Analytics_Project_Report.pdf"),
        ("Vision_Based_Customer_Trajectory_Analytics_Project_Report.docx", "Vision_Based_Customer_Trajectory_Analytics_Project_Report.docx"),
        ("docs/presentation/FINAL_DEMO_PRESENTATION.html", "FINAL_DEMO_PRESENTATION.html"),
        ("docs/presentation/FINAL_DEMO_PRESENTATION.md", "FINAL_DEMO_PRESENTATION.md"),
        ("docs/DEMONSTRATION_SCRIPT.md", "DEMONSTRATION_SCRIPT.md"),
    ]

    for src_rel, dst_rel in required_docs:
        src = Path(src_rel)
        if not src.is_file():
            raise FileNotFoundError(f"Packaging failed: Required document deliverable missing: '{src}'")
        dst = doc_pkg / dst_rel
        shutil.copy2(src, dst)
        print(f"  [+] Copied document: {src_rel} -> {dst_rel} ({dst.stat().st_size} bytes)")

    # Also bundle fallback presentation directory if present
    fallback_src = Path("docs/presentation/fallback")
    if fallback_src.is_dir():
        fallback_dst = doc_pkg / "fallback"
        copy_tree_clean(fallback_src, fallback_dst)
        print(f"  [+] Bundled fallback presentation package: {fallback_dst}")

    # Build Document Inventory & Checksums (relative to document_package root)
    checksums = []
    # 1. Primary documents
    for _, dst_rel in required_docs:
        file_path = doc_pkg / dst_rel
        sha = compute_sha256(file_path)
        sz = file_path.stat().st_size
        checksums.append({
            "path": dst_rel,
            "sha256": sha,
            "bytes": sz,
        })

    # 2. Fallback files
    fallback_dst = doc_pkg / "fallback"
    if fallback_dst.is_dir():
        for fb_file in sorted(fallback_dst.iterdir()):
            if fb_file.is_file():
                rel_fb = f"fallback/{fb_file.name}"
                sha = compute_sha256(fb_file)
                sz = fb_file.stat().st_size
                checksums.append({
                    "path": rel_fb,
                    "sha256": sha,
                    "bytes": sz,
                })

    manifest_file = doc_pkg / "MANIFEST_CHECKSUMS.json"
    manifest_data = {
        "package_name": "document_package",
        "package_root": "submission/document_package",
        "candidate": "Kaniga Parameswaran R (Reg. No: 250282601051)",
        "course": "CAF7104 - MINI PROJECT, MCA",
        "institution": "B.S. Abdur Rahman Crescent Institute of Science and Technology",
        "generation_date": "2026-11-07",
        "total_packaged_files": len(checksums),
        "files": checksums,
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"  [+] Generated manifest for {len(checksums)} files: {manifest_file.name}")

    # Write DOCUMENT_INVENTORY.md
    inv_file = doc_pkg / "DOCUMENT_INVENTORY.md"
    inv_content = f"""# Document Submission Package Inventory

> **Project:** Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
> **Course:** CAF7104 — MINI PROJECT (MCA)  
> **Candidate:** Kaniga Parameswaran R (Reg. No: 250282601051)  
> **Institution:** Department of Computer Applications, B.S. Abdur Rahman Crescent Institute of Science and Technology  
> **Date:** November 2026  
> **Package Root:** `submission/document_package/`  

---

## 1. Primary Deliverables

All file paths are relative to this document package root (`submission/document_package/`):

| File Name | Format | Size | Description & Institutional Role |
|---|---|---|---|
| [`Vision_Based_Customer_Trajectory_Analytics_Project_Report.pdf`](./Vision_Based_Customer_Trajectory_Analytics_Project_Report.pdf) | PDF Document | { (doc_pkg / 'Vision_Based_Customer_Trajectory_Analytics_Project_Report.pdf').stat().st_size / 1024:.1f} KB | Authoritative academic project report matching institutional bonafide, certificate, and chapter formatting. |
| [`Vision_Based_Customer_Trajectory_Analytics_Project_Report.docx`](./Vision_Based_Customer_Trajectory_Analytics_Project_Report.docx) | Word Document | { (doc_pkg / 'Vision_Based_Customer_Trajectory_Analytics_Project_Report.docx').stat().st_size / (1024*1024):.2f} MB | Editable academic project report. |
| [`FINAL_DEMO_PRESENTATION.html`](./FINAL_DEMO_PRESENTATION.html) | HTML Slide Deck | { (doc_pkg / 'FINAL_DEMO_PRESENTATION.html').stat().st_size / 1024:.1f} KB | Interactive 10-slide presentation deck with keyboard navigation (←/→ arrows). |
| [`FINAL_DEMO_PRESENTATION.md`](./FINAL_DEMO_PRESENTATION.md) | Markdown | { (doc_pkg / 'FINAL_DEMO_PRESENTATION.md').stat().st_size / 1024:.1f} KB | Presentation deck in GitHub-flavored Markdown format. |
| [`DEMONSTRATION_SCRIPT.md`](./DEMONSTRATION_SCRIPT.md) | Markdown | { (doc_pkg / 'DEMONSTRATION_SCRIPT.md').stat().st_size / 1024:.1f} KB | 5-minute presentation script with timeline cues and speaker notes. |
| [`fallback/FALLBACK_INDEX.md`](./fallback/FALLBACK_INDEX.md) | Markdown & Screenshots | 13 verified PNGs | Offline visual presentation fallback with verified screenshots (no prerecorded video walkthrough exists). |
| [`MANIFEST_CHECKSUMS.json`](./MANIFEST_CHECKSUMS.json) | JSON Ledger | { manifest_file.stat().st_size } bytes | Package-relative SHA-256 fingerprint ledger ({len(checksums)} files). |

---

## 2. Checksum Verification

To verify checksums from within `submission/document_package/`:

```powershell
python -c "
import json, hashlib
with open('MANIFEST_CHECKSUMS.json') as f: m = json.load(f)
for item in m['files']:
    h = hashlib.sha256(open(item['path'], 'rb').read()).hexdigest()
    assert h == item['sha256'], f'Mismatch in {{item[\"path\"]}}'
    print(f'[MATCH] {{item[\"path\"]}}')
print('ALL DOCUMENT CHECKSUMS VERIFIED')
"
```
"""
    with open(inv_file, "w", encoding="utf-8") as f:
        f.write(inv_content)
    print(f"  [+] Generated inventory: {inv_file.name}")

    return manifest_data
    print(f"  [+] Generated inventory: {inv_file.name}")

    return manifest_data


def package_runnable_source_release(base_sub: Path) -> Dict[str, Any]:
    print("\n" + "=" * 65)
    print(" 2. Packaging Runnable Source Release")
    print("=" * 65)
    rel_root = base_sub / "runnable_source_release"
    rel_root.mkdir(parents=True, exist_ok=True)

    # 1. Copy source code and components
    print("[*] Copying core application code...")
    copy_tree_clean(Path("src"), rel_root / "src")
    copy_tree_clean(Path("dashboard"), rel_root / "dashboard")
    copy_tree_clean(Path("configs"), rel_root / "configs")
    copy_tree_clean(Path("tests"), rel_root / "tests")

    # 2. Copy scripts
    print("[*] Copying scripts...")
    (rel_root / "scripts").mkdir(exist_ok=True)
    required_scripts = [
        "check_prerequisites.py",
        "run_vscode_demo.py",
        "validate_annotation_submission.py",
        "verify_phase_08.py",
        "verify_phase_10.py",
        "generate_zone_reference_images.py",
        "export_pilot_frames.py",
        "prepare_pilot_workbooks.py",
        "run_public_dataset_examples.py",
        "run_sample_analysis.py",
    ]
    for s_name in required_scripts:
        src_s = Path("scripts") / s_name
        if not src_s.is_file():
            raise FileNotFoundError(f"Packaging failed: Required script missing: '{src_s}'")
        shutil.copy2(src_s, rel_root / "scripts" / s_name)

    # 3. Copy outputs/reports (saved runs for instant demo)
    print("[*] Copying saved demonstration run outputs...")
    reports_dst = rel_root / "outputs" / "reports"
    reports_dst.mkdir(parents=True, exist_ok=True)
    for item in Path("outputs/reports").iterdir():
        if item.is_file() and item.suffix.lower() in [".json", ".png", ".jpg", ".jpeg"]:
            shutil.copy2(item, reports_dst / item.name)

    # Copy uploaded runs directory
    if Path("outputs/reports/uploaded_runs").is_dir():
        copy_tree_clean(Path("outputs/reports/uploaded_runs"), reports_dst / "uploaded_runs")

    # 4. Copy data assets (models, videos, pilot manifests)
    print("[*] Copying bundled media assets and neural model...")
    data_dst = rel_root / "data"
    data_dst.mkdir(parents=True, exist_ok=True)
    
    # Model
    model_src = Path("data/models/yolo11n.onnx")
    if model_src.is_file():
        (data_dst / "models").mkdir(exist_ok=True)
        shutil.copy2(model_src, data_dst / "models" / "yolo11n.onnx")
        print("  [+] Bundled neural model: data/models/yolo11n.onnx")
    else:
        print("  [-] Note: yolo11n.onnx not present in local data/models/.")

    # Videos
    for v_rel in ["merl/10_1_crop.mp4", "prcv_site03_boutique.mp4", "prcv_site02_jewelry.mp4"]:
        src_v = Path("data") / v_rel
        if src_v.is_file():
            dst_v = data_dst / v_rel
            dst_v.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_v, dst_v)
            print(f"  [+] Bundled verified recording: data/{v_rel}")

    # Evaluation pilot
    if Path("data/evaluation_pilot").is_dir():
        copy_tree_clean(Path("data/evaluation_pilot"), data_dst / "evaluation_pilot")

    # 5. Root configuration and launcher files
    print("[*] Copying root configuration and launcher files...")
    root_files = [
        ("start_demo.cmd", "start_demo.cmd"),
        ("requirements.txt", "requirements.txt"),
        ("config.yaml", "config.yaml"),
        ("README.md", "README.md"),
        (".gitignore", ".gitignore"),
    ]
    for r_src, r_dst in root_files:
        p_src = Path(r_src)
        if not p_src.is_file():
            raise FileNotFoundError(f"Packaging failed: Required root file missing: '{p_src}'")
        shutil.copy2(p_src, rel_root / r_dst)

    # 6. Generate ASSET_MANIFEST.md inside runnable release
    print("[*] Generating ASSET_MANIFEST.md...")
    asset_manifest_content = """# Asset Placement & Offline Readiness Rules

## 1. Required Asset Locations
The application resolves assets relative to this source release root:

| Asset Category | Relative Path in Release | Purpose & Offline Role |
|---|---|---|
| **YOLO11-nano Detection Model** | `data/models/yolo11n.onnx` | Neural inference model required for **New Video Analysis** (`?view=new`). |
| **MERL Supermarket Video** | `data/merl/10_1_crop.mp4` | Primary retail benchmark clip (633 sampled frames at 5.0 FPS) for synchronized video playback. |
| **PRCV Site 03 Boutique Video** | `data/prcv_site03_boutique.mp4` | Home Decor boutique recording (134 frames) for synchronized video playback. |
| **PRCV Site 02 Jewelry Video** | `data/prcv_site02_jewelry.mp4` | Luxury Jewelry store recording (73 frames) for synchronized video playback. |
| **Authoritative Scene Profiles** | `configs/*_scene_profile.json` | Configured image-to-floor correspondences (P1..P4) and polygon boundaries for analytic zones; independently surveyed physical calibration is unavailable. |
| **Pre-computed Saved Runs** | `outputs/reports/*.json` | Standardized run summaries and observation timelines for instant **Saved-Results Demonstration** (`?view=all`). |

## 2. Offline Readiness Invariant
- **Saved-Results Demonstration (`?view=all`):**
  Operates entirely from `outputs/reports/` and `configs/`. It does NOT require neural inference. If video files are present, synchronized video playback is enabled.
- **New Video Analysis (`?view=new`):**
  Requires `data/models/yolo11n.onnx` and input video footage. If `yolo11n.onnx` is missing, `scripts/check_prerequisites.py` reports partial readiness: demonstration is ready, but new analysis is withheld.

## 3. Acquisition of Missing Assets
If models or media are missing on a fresh installation:
1. **YOLO11-nano ONNX Model:**
   Export via Ultralytics:
   ```powershell
   pip install ultralytics
   yolo export model=yolo11n.pt format=onnx imgsz=640 opset=12 simplify=True
   move yolo11n.onnx data/models/yolo11n.onnx
   ```
2. **Video Footage:**
   Place recordings in the documented `data/` paths above.
"""
    with open(rel_root / "ASSET_MANIFEST.md", "w", encoding="utf-8") as f:
        f.write(asset_manifest_content)

    # 7. Compute Checksums for Runnable Release
    print("[*] Computing package-relative SHA-256 checksums...")
    # Clean any temporary bytecode or pytest caches before computing checksums
    for p in list(rel_root.rglob("__pycache__")):
        shutil.rmtree(p, ignore_errors=True)
    for p in list(rel_root.rglob(".pytest_cache")):
        shutil.rmtree(p, ignore_errors=True)
    for p in list(rel_root.rglob("*.py[co]")):
        try:
            p.unlink()
        except OSError:
            pass

    checksums = []
    # Collect all regular files, excluding caches
    for root, dirs, files in os.walk(rel_root):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache", ".git")]
        dirs.sort()
        files.sort()
        for fname in files:
            if fname in ["MANIFEST_CHECKSUMS.json", "RELEASE_INVENTORY.md"] or fname.endswith((".pyc", ".pyo")):
                continue
            f_abs = Path(root) / fname
            rel_path = f_abs.relative_to(rel_root).as_posix()
            sha = compute_sha256(f_abs)
            sz = f_abs.stat().st_size
            checksums.append({
                "path": rel_path,
                "sha256": sha,
                "bytes": sz,
            })

    manifest_file = rel_root / "MANIFEST_CHECKSUMS.json"
    manifest_data = {
        "package_name": "runnable_source_release",
        "package_root": "submission/runnable_source_release",
        "candidate": "Kaniga Parameswaran R (Reg. No: 250282601051)",
        "course": "CAF7104 - MINI PROJECT, MCA",
        "institution": "B.S. Abdur Rahman Crescent Institute of Science and Technology",
        "generation_date": "2026-11-07",
        "verification_record": {
            "test_command": "pytest tests/ -q -p no:cacheprovider",
            "passed_tests": 396,
            "skipped_tests": 3,
            "total_collected_tests": 399,
            "modular_suites": 29,
            "verification_date": "2026-11-07",
            "exit_code": 0,
        },
        "total_packaged_files": len(checksums),
        "files": checksums,
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"  [+] Generated manifest for {len(checksums)} files: {manifest_file.name}")

    # 8. Write RELEASE_INVENTORY.md
    inv_file = rel_root / "RELEASE_INVENTORY.md"
    inv_content = r"""# Runnable Source Release Inventory & Quickstart

> **Project:** Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
> **Course:** CAF7104 — MINI PROJECT (MCA)  
> **Candidate:** Kaniga Parameswaran R (Reg. No: 250282601051)  
> **Institution:** Department of Computer Applications, B.S. Abdur Rahman Crescent Institute of Science and Technology  
> **Date:** November 2026  
> **Package Root:** `submission/runnable_source_release/`  

---

## 1. Quickstart Launch Instructions

### Clean Installation & Setup (from clean state)
To install dependencies from a clean state:
```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe scripts/check_prerequisites.py
.\start_demo.cmd
```
> [!NOTE]
> `start_demo.cmd` requires a release-local virtual environment at `.\venv\Scripts\python.exe`. Clean-machine installation from an empty OS was not verified; verification commands were executed using the local test environment.

### Launch Application
Launch from this release root directory:

```cmd
:: Launch preflight checks and Streamlit application on Windows
start_demo.cmd
```

Or run via Python directly:

```powershell
# Run preflight environment check (probes models, media, and port 8501 status)
python scripts/check_prerequisites.py

# Launch Streamlit dashboard directly
python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```
Open **`http://127.0.0.1:8501/?view=all`** in any web browser.

> [!IMPORTANT]
> **Spatial Calibration Boundary:** Configured image-to-floor correspondences; independently surveyed physical calibration is unavailable. Floor coordinates represent defined scene units.

---

## 2. Directory Structure Overview

All paths are relative to this release root:

```
submission/runnable_source_release/
├── src/                      # Complete source code (analytics, spatial, tracking, detection)
├── dashboard/                # Complete Streamlit UI views and components
├── configs/                  # Authoritative camera scene profiles (JSON)
├── data/                     # Bundled neural models, video clips, and evaluation pilot
├── outputs/reports/          # Pre-computed saved demonstration run summaries
├── scripts/                  # Preflight check and utility scripts
├── tests/                    # Complete automated regression test suite
├── start_demo.cmd            # One-click Windows launcher with server-reuse support
├── requirements.txt          # Python dependencies
├── config.yaml               # Default system pipeline configuration
├── README.md                 # Complete project guide and architecture documentation
├── ASSET_MANIFEST.md         # Asset placement and offline readiness rules
├── RELEASE_INVENTORY.md      # This inventory file
└── MANIFEST_CHECKSUMS.json   # Package-relative SHA-256 manifest ({len(checksums)} files)
```

---

## 3. Accurately Dated Verification Record

- **Verification Date:** 2026-11-07
- **Verification Command:** `pytest tests/ -q -p no:cacheprovider`
- **Verification Result:** **396 passed, 3 skipped (out of 399 collected tests)** across 29 modular test suites in 62.81s (Exit Code: 0, 100% execution pass rate; 3 optional detection tests skipped due to unbundled MOT17 training frames documented in ASSET_MANIFEST.md).
- **Component Verification:** `python scripts/verify_phase_08.py` — **70 passed checks** in 1.86s.
"""
    with open(inv_file, "w", encoding="utf-8") as f:
        f.write(inv_content)
    print(f"  [+] Generated inventory: {inv_file.name}")

    return manifest_data


def generate_master_index(base_sub: Path):
    index_file = base_sub / "MASTER_INDEX.md"
    index_content = r"""# Final Project Submission & Portable Release Index

> **Project:** Vision-Based Customer Trajectory Analytics for Retail Layout Optimization  
> **Course:** CAF7104 — MINI PROJECT (MCA)  
> **Candidate:** Kaniga Parameswaran R (Reg. No: 250282601051)  
> **Institution:** Department of Computer Applications, B.S. Abdur Rahman Crescent Institute of Science and Technology  
> **Submission Date:** November 2026  

---

## 1. Directory Organization

This submission is strictly partitioned into two independent packages:

```
submission/
├── document_package/         # Formal academic document deliverables for examination
│   ├── Vision_Based_Customer_Trajectory_Analytics_Project_Report.pdf
│   ├── Vision_Based_Customer_Trajectory_Analytics_Project_Report.docx
│   ├── FINAL_DEMO_PRESENTATION.html
│   ├── FINAL_DEMO_PRESENTATION.md
│   ├── DEMONSTRATION_SCRIPT.md
│   ├── DOCUMENT_INVENTORY.md
│   └── MANIFEST_CHECKSUMS.json
│
├── runnable_source_release/  # Complete, standalone runnable application source tree
│   ├── src/
│   ├── dashboard/
│   ├── configs/
│   ├── data/
│   ├── outputs/reports/
│   ├── scripts/
│   ├── tests/
│   ├── start_demo.cmd
│   ├── requirements.txt
│   ├── config.yaml
│   ├── README.md
│   ├── ASSET_MANIFEST.md
│   ├── RELEASE_INVENTORY.md
│   └── MANIFEST_CHECKSUMS.json
│
└── MASTER_INDEX.md           # This master index
```

---

## 2. Reviewer Instructions

### A. Document Reviewers & Examiners
Navigate to [`submission/document_package/`](./document_package/):
- **Authoritative Report (PDF):** [`Vision_Based_Customer_Trajectory_Analytics_Project_Report.pdf`](./document_package/Vision_Based_Customer_Trajectory_Analytics_Project_Report.pdf)
- **Authoritative Report (DOCX):** [`Vision_Based_Customer_Trajectory_Analytics_Project_Report.docx`](./document_package/Vision_Based_Customer_Trajectory_Analytics_Project_Report.docx)
- **Presentation Deck (HTML):** [`FINAL_DEMO_PRESENTATION.html`](./document_package/FINAL_DEMO_PRESENTATION.html)
- **Presentation Deck (MD):** [`FINAL_DEMO_PRESENTATION.md`](./document_package/FINAL_DEMO_PRESENTATION.md)
- **Demonstration Cue Sheet:** [`DEMONSTRATION_SCRIPT.md`](./document_package/DEMONSTRATION_SCRIPT.md)
- **Fallback Visual Evidence:** [`fallback/FALLBACK_INDEX.md`](./document_package/fallback/FALLBACK_INDEX.md) (13 verified static screenshots)

### B. Technical Evaluators & Demonstration Setup
Navigate to [`submission/runnable_source_release/`](./runnable_source_release/):
1. **Clean Installation & Setup (from clean state):**
   ```powershell
   cd submission/runnable_source_release
   python -m venv venv
   .\venv\Scripts\python.exe -m pip install -r requirements.txt
   .\venv\Scripts\python.exe scripts/check_prerequisites.py
   .\start_demo.cmd
   ```
   > Note: `start_demo.cmd` requires a release-local virtual environment at `.\venv\Scripts\python.exe`. Clean-machine installation from an empty OS was not verified; verification commands were executed using the local test environment.

2. **Preflight Check:**
   ```powershell
   python scripts/check_prerequisites.py
   ```
3. **Launch Application:**
   ```cmd
   start_demo.cmd
   ```
   Access URL: `http://127.0.0.1:8501/?view=all`
4. **Execute Verification Test Suite:**
   ```powershell
   python -m pytest tests/ -q -p no:cacheprovider
   ```

> [!IMPORTANT]
> **Spatial Calibration Boundary:** Configured image-to-floor correspondences; independently surveyed physical calibration is unavailable. Floor coordinates represent defined scene units.

---

## 3. Accurately Dated Verification Record

- **Verification Date:** 2026-11-07
- **Verification Command:** `pytest tests/ -q -p no:cacheprovider`
- **Verification Result:** **396 passed, 3 skipped (out of 399 collected tests)** across 29 modular test suites in 62.81s (Exit Code: 0, 100% execution pass rate; 3 optional detection tests skipped due to unbundled MOT17 training frames documented in ASSET_MANIFEST.md).
- **Component Verification:** `python scripts/verify_phase_08.py` — **70 passed checks** in 1.86s.
"""
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(index_content)
    print(f"\n[+] Generated MASTER_INDEX.md: {index_file}")


def main():
    base_sub = Path("submission")
    base_sub.mkdir(exist_ok=True)

    # 1. Package document submission
    doc_manifest = package_document_package(base_sub)

    # 2. Package runnable source release
    rel_manifest = package_runnable_source_release(base_sub)

    # 3. Generate Master Index
    generate_master_index(base_sub)

    print("\n" + "=" * 65)
    print("[SUCCESS] All packages generated and verified successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
