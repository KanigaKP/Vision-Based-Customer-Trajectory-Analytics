# Phase 1 — Environment Report

**Date:** 2026-09-17

## System Hardware

| Component | Details |
|---|---|
| **CPU** | AMD Ryzen 5 7520U with Radeon Graphics |
| **Cores / Threads** | 4 cores / 8 logical processors |
| **Max Clock** | 2801 MHz |
| **RAM** | 15.28 GB total |
| **GPU** | AMD Radeon(TM) Graphics (integrated, 512 MB VRAM) |
| **NVIDIA GPU** | None |
| **CUDA** | Not available |

## Software

| Component | Version |
|---|---|
| **OS** | Windows 11 Home Single Language, Build 26200 |
| **Python** | 3.14.6 (MSC v.1944, 64-bit) |
| **pip** | 26.2.1 |
| **Git** | 2.53.0.windows.2 |
| **Virtual Environment** | `venv/` (stdlib venv) |

## GPU / CUDA Assessment

- **No NVIDIA GPU** is present. `nvidia-smi` is not available.
- Only an AMD Radeon integrated GPU with 512 MB VRAM exists.
- **CUDA is not available.** PyTorch and YOLO11 will run in **CPU-only mode**.
- CPU-only inference is viable for this academic prototype since we process recorded video (not real-time).

## Security Policy

- **Windows Smart App Control:** Enabled (enforcement mode)
- **Impact:** Blocks loading of unsigned DLLs including `torch.dll`
- **Resolution required:** User must disable Smart App Control
- See Phase 1 Report for details
