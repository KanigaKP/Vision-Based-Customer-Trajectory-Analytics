"""
Preflight Prerequisite and Environment Checker
===============================================
Verifies prerequisites for running the Vision-Based Customer Trajectory Analytics
prototype and Streamlit dashboard.

Explicitly distinguishes two execution modes:
  1. Saved-Results Demonstration Mode (?view=all):
     - Requires: Python, core dependencies, scene profiles, saved reports, video playback assets.
     - Does NOT require neural model inference (all analytics are pre-computed).
  2. New-Analysis Execution Mode (?view=new):
     - Requires: Detection model (yolo11n.onnx), video assets, scene profiles, and runtime libraries.
     - Missing model or media specifically invalidates new-analysis readiness.

Server Status & Port Management (Port 8501):
  - Free Port: Returns code 0 (ready for fresh launch).
  - Healthy Project Server: Returns code 2 (reuse running server without launching another).
  - Occupied Foreign / Unresponsive Port: Returns code 3 (explicit conflict detection).
  - Core Failure: Returns code 1 (missing dependencies or malformed configs).

Usage:
  python scripts/check_prerequisites.py [--port 8501] [--launcher]
"""

import sys
import os
import json
import socket
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


def check_python_version() -> bool:
    print(f"[*] Python Version: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 9):
        print("    [FAIL] Python 3.9 or higher is required.")
        return False
    print("    [PASS] Python version compatible.")
    return True


def check_dependencies() -> bool:
    required_packages = [
        ("streamlit", "streamlit"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("yaml", "pyyaml"),
        ("onnxruntime", "onnxruntime"),
    ]
    all_ok = True
    print("[*] Checking Core Python Dependencies:")
    for mod_name, pkg_name in required_packages:
        try:
            __import__(mod_name)
            print(f"    [PASS] {pkg_name} installed.")
        except ImportError:
            print(f"    [FAIL] Missing package: {pkg_name} (run: pip install {pkg_name})")
            all_ok = False
    return all_ok


def check_models() -> Tuple[bool, Optional[Path]]:
    print("[*] Checking Neural Detection Models (Required for New Analysis):")
    candidate_paths = [
        Path("data/models/yolo11n.onnx"),
        Path("models/yolo11n.onnx"),
    ]
    for model_path in candidate_paths:
        if model_path.is_file():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"    [PASS] Model found: {model_path} ({size_mb:.2f} MB).")
            return True, model_path

    print("    [WARN] Detection model 'yolo11n.onnx' not found.")
    print("           New video analysis cannot execute inference without this model.")
    return False, None


def check_media() -> Tuple[bool, int, int]:
    print("[*] Checking Video Assets (Required for Video Playback):")
    videos = [
        Path("data/merl/10_1_crop.mp4"),
        Path("data/prcv_site03_boutique.mp4"),
        Path("data/prcv_site02_jewelry.mp4"),
    ]
    found = 0
    for v in videos:
        if v.is_file():
            size_mb = v.stat().st_size / (1024 * 1024)
            print(f"    [PASS] {v} ({size_mb:.2f} MB)")
            found += 1
        else:
            print(f"    [WARN] {v} missing (synchronized playback disabled for this clip)")
    all_present = (found == len(videos))
    return all_present, found, len(videos)


def check_configs() -> bool:
    print("[*] Checking Authoritative Scene Profiles:")
    profiles = [
        ("MERL Supermarket", Path("configs/merl_scene_profile.json")),
        ("PRCV Site 03 Boutique", Path("configs/prcv_site03_scene_profile.json")),
        ("PRCV Site 02 Jewelry", Path("configs/prcv_site02_scene_profile.json")),
    ]
    all_ok = True
    for label, path in profiles:
        if not path.is_file():
            print(f"    [FAIL] {label} profile missing at {path}")
            all_ok = False
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            zones = [z.get("zone_id") for z in data.get("zones", [])]
            bounds = data.get("bounds", data.get("floor_bounds", []))
            print(f"    [PASS] {label} ({path.name}): {len(zones)} zones, bounds {bounds}")
        except Exception as exc:
            print(f"    [FAIL] {label} profile invalid: {exc}")
            all_ok = False
    return all_ok


def check_saved_reports() -> bool:
    print("[*] Checking Saved Demonstration Reports:")
    reports = [
        ("MERL Full Demo", Path("outputs/reports/demo_real_summary.json")),
        ("MERL Upload Run", Path("outputs/reports/uploaded_runs/upload_ca30f3e8/run_summary.json")),
        ("Site 03 Boutique Run", Path("outputs/reports/uploaded_runs/upload_6f969f54/run_summary.json")),
        ("Site 02 Jewelry Run", Path("outputs/reports/uploaded_runs/upload_site02_jewelry/run_summary.json")),
    ]
    found = 0
    for label, path in reports:
        if path.is_file():
            print(f"    [PASS] {label}: {path}")
            found += 1
        else:
            print(f"    [WARN] {label} missing at {path}")
    if found > 0:
        print(f"    [PASS] {found}/{len(reports)} saved run reports available for immediate demonstration.")
        return True
    else:
        print("    [FAIL] No saved run reports found. Saved-results demonstration cannot render.")
        return False


def probe_port_status(port: int = 8501, host: str = "127.0.0.1") -> str:
    """Probe port state. Returns 'free', 'healthy_server', or 'occupied_unknown'."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    result = s.connect_ex((host, port))
    s.close()

    if result != 0:
        return "free"

    # Port is open; test if it is a responsive Streamlit server
    health_url = f"http://{host}:{port}/_stcore/health"
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "PreflightChecker"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.getcode() == 200:
                body = resp.read().decode("utf-8", errors="ignore").strip().lower()
                if "ok" in body or resp.getcode() == 200:
                    return "healthy_server"
    except Exception:
        pass

    return "occupied_unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prerequisite checker for trajectory analytics prototype.")
    parser.add_argument("--port", type=int, default=8501, help="Port to probe (default: 8501)")
    parser.add_argument("--launcher", action="store_true", help="Launcher mode: return specific exit codes for batch file")
    args = parser.parse_args()

    print("=" * 68)
    print(" Vision-Based Trajectory Analytics: Startup Preflight Verification")
    print("=" * 68)

    v_ok = check_python_version()
    d_ok = check_dependencies()
    c_ok = check_configs()
    r_ok = check_saved_reports()
    all_media, found_media, total_media = check_media()
    m_ok, _ = check_models()

    print("-" * 68)
    print("[*] Probing Port & Server Status:")
    port_state = probe_port_status(args.port)
    if port_state == "healthy_server":
        print(f"    [INFO] Streamlit server is ALREADY RUNNING and healthy on port {args.port}!")
        print(f"    -> Access URL: http://127.0.0.1:{args.port}/?view=all")
        print("    -> REUSE READY: Do NOT launch another server; use existing instance.")
    elif port_state == "occupied_unknown":
        print(f"    [WARN] Port {args.port} is OCCUPIED by an unresponsive or non-Streamlit service.")
        print(f"    -> Cannot bind port {args.port}. Free port {args.port} or choose another port.")
    else:
        print(f"    [INFO] Port {args.port} is FREE. Ready for fresh server launch.")

    print("=" * 68)
    print("[*] Operational Readiness Assessment:")

    # 1. Saved-Results Demonstration Readiness
    # Requires Python, core dependencies, scene profiles, saved reports, and video files
    demo_ready = v_ok and d_ok and c_ok and r_ok and (found_media > 0)
    if demo_ready:
        print("  [PASS] Saved-Results Demonstration: READY")
        print("         (View precomputed MERL, Site 03, and Site 02 analytics at ?view=all)")
    else:
        print("  [FAIL] Saved-Results Demonstration: NOT READY (missing core deps, configs, or reports)")

    # 2. New-Analysis Execution Readiness
    # Requires Python, core dependencies, scene profiles, ONNX model, and video files
    new_analysis_ready = v_ok and d_ok and c_ok and m_ok and (found_media > 0)
    if new_analysis_ready:
        print("  [PASS] New-Analysis Pipeline: READY")
        print("         (Offline YOLO11-nano ONNX detection and ByteTrack tracking at ?view=new)")
    else:
        reasons = []
        if not m_ok:
            reasons.append("model 'yolo11n.onnx' missing")
        if found_media == 0:
            reasons.append("no video media in data/")
        if not (v_ok and d_ok and c_ok):
            reasons.append("core dependencies or profiles missing")
        print(f"  [WARN] New-Analysis Pipeline: NOT READY ({', '.join(reasons)})")

    # Overall Summary - strictly distinguish full pass from partial readiness
    print("-" * 68)
    if demo_ready and new_analysis_ready:
        print("[SUCCESS] All system prerequisites verified (Both Demonstration & New Analysis ready).")
    elif demo_ready:
        print("[PARTIAL SUCCESS] Saved-Results Demonstration is fully functional.")
        print("                  New video analysis is withheld until yolo11n.onnx is placed in data/models/.")
    else:
        print("[FAILURE] Essential runtime prerequisites are missing. See errors above.")

    print("=" * 68)

    # Exit code determination
    if not (v_ok and d_ok and c_ok):
        return 1

    if args.launcher:
        if port_state == "healthy_server":
            return 2  # Reusable healthy server
        elif port_state == "occupied_unknown":
            return 3  # Occupied port conflict
        else:
            return 0 if demo_ready else 1

    # In standard mode: return 0 if demo is ready and port not foreign-occupied
    if port_state == "occupied_unknown":
        return 1
    return 0 if demo_ready else 1


if __name__ == "__main__":
    sys.exit(main())
