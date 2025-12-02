# CASSIA Docker Demo

This zip contains a minimal working prototype of **CASSIA: Context-Aware Self-Supervised Intelligent Autoscheduler**.

## Files

- `cassia_demo.py`  — main Python script (self-contained demo)
- `requirements.txt` — Python dependencies
- `Dockerfile` — to build and run in Docker

## How to Use

1. Unzip files.
2. Open a terminal in the unzipped folder.

### Run with plain Python (optional)

```bash
pip install -r requirements.txt
python cassia_demo.py
```

### Run with Docker

```bash
docker build -t cassia-demo .
docker run --rm -v "$PWD:/app" cassia-demo
```

The script will:

- Print metrics for FIFO, Round Robin, CASSIA, and REINFORCE strategies.
- Save three plots in the working directory:
  - `latency_comparison.png`
  - `cold_starts_comparison.png`
  - `throughput_comparison.png`
