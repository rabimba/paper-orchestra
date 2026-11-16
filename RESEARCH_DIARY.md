# Research Diary: PaperOrchestra Offloaded Compute

> **Project Objective**: Multi-agent framework for automated AI research paper writing, offloading computation and model inference to `puffingbird`.
> **Host**: `puffingbird` (`172.23.79.37`) | **Hardware**: 2x NVIDIA H100 NVL (94GB each, PCIe NVLink), 64 vCPUs, 1TB DDR5 RAM
> **Storage Root**: `/data/rabimba/projects/paper-orchestra/` (8TB Enterprise SSD)
> **GitHub Repository**: `https://github.com/rabimba/paper-orchestra`

---

## Experiment Index

| Exp ID | Date & Time | Git Commit | Focus / Hypothesis | Status | Primary Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `SETUP-001` | 2026-11-16 03:30 | `origin/main` | Environment scaffold, remote dual-H100 setup, offloading runner | Completed | Local & remote venv verified |

---

## Detailed Experiment Logs

### [SETUP-001] Infrastructure & Offloading Pipeline Scaffold

- **Timestamp**: `2026-11-16 03:30:00 UTC`
- **Git State**:
  - Branch: `main`
  - Remotes: `origin: rabimba/paper-orchestra`, `upstream: google-research/paper-orchestra`
- **Hardware & Environment**:
  - Local: Ubuntu 24.04 (`.venv` Python 3.12, 82 packages installed)
  - Remote: `puffingbird` 2x NVIDIA H100 NVL, 1TB RAM (`/data/rabimba/envs/paper-orchestra`)
  - Storage: Strictly on `/data/rabimba/` 8TB Enterprise SSD
- **Configuration**:
  - Open-source and frontier LLM offloading via `OPENAI_BASE_URL`
  - Automated job offloading via `offload_to_puffingbird.sh`
