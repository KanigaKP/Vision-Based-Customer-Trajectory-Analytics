# Decision Log

This document records key architectural and technical decisions made throughout the project.

---

## DEC-001 — Python Version

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Use Python 3.14.6 (system-installed version) |
| **Rationale** | This is the version available on the development machine. No compatibility issues have been identified with the Phase 0 dependency stack. Compatibility with Ultralytics/YOLO11 and other future dependencies will be verified in Phase 1 before proceeding. |
| **Alternatives** | Install Python 3.11 or 3.12 for broader library compatibility. Will revisit if Ultralytics or ByteTrack requires it. |

---

## DEC-002 — Virtual Environment Tool

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Use Python's built-in `venv` module |
| **Rationale** | Lightweight, no extra tooling required, sufficient for a single-project academic prototype. |
| **Alternatives** | conda, poetry, pipenv — unnecessary complexity for this project. |

---

## DEC-003 — Phase 0 Dependency Scope

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Install only foundational scientific/utility packages: numpy, opencv-python, pandas, matplotlib, pyyaml |
| **Rationale** | Phase 0 is environment validation only. Heavy dependencies (ultralytics, streamlit, etc.) are deferred to their respective implementation phases to avoid premature breakage and keep the environment minimal. |
| **Alternatives** | Install the full stack upfront — rejected to reduce Phase 0 risk and keep the environment clean. |

---

## DEC-004 — No Version Pinning in Phase 0

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Install latest compatible versions without hard pins in Phase 0 |
| **Rationale** | Python 3.14 is recent; arbitrary pins from older projects are likely to cause install failures. Actual installed versions are recorded in PHASE_00_REPORT.md and can be frozen via `pip freeze` once validated. |
| **Alternatives** | Pin exact versions — will be done after Phase 0 validation succeeds (via `pip freeze > requirements-lock.txt`). |

---

## DEC-005 — Project Structure

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Adopt the structure proposed in the project specification with no modifications |
| **Rationale** | The proposed structure cleanly separates concerns (data, source, dashboard, tests, outputs, docs) and matches the pipeline stages. No technical reason to deviate. |
| **Alternatives** | Add `notebooks/` directory — deferred; will add if needed. |

---

## DEC-006 — Database Choice

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Use SQLite (stdlib `sqlite3`) — no external database server |
| **Rationale** | Zero-install, file-based, sufficient for a single-user research prototype. Keeps deployment trivial for academic evaluation. |
| **Alternatives** | PostgreSQL, MongoDB — unnecessary for this project scale. |

---

## DEC-007 — No Docker/Cloud Infrastructure

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Phase** | 0 |
| **Decision** | Do not use Docker, cloud services, Redis, Kafka, or any enterprise infrastructure |
| **Rationale** | This is an academic research prototype. The environment is a single Windows machine. Adding infrastructure would add complexity without benefit. |
| **Alternatives** | N/A |
