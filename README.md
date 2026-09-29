# SwapeDev: On-Demand Kaggle GPU Worker Orchestrator

SwapeDev is an automated orchestration system designed to provision on-demand Kaggle GPU compute workers, inject per-profile runtime secrets via private Kaggle datasets, establish secure reverse tunnels via Cloudflare, and manage asynchronous ML/inference job queues.

---

## Architecture Overview

```
                                      +-------------------------------+
                                      |    SwapeDev Orchestrator      |
                                      |   (FastAPI + WorkerManager)   |
                                      +---------------+---------------+
                                                      |
                    1. Create/Upload Secrets Dataset  | 2. Push & Run Kernel
                    (swapedev-secrets-{profile_id})   | (scripts/kaggle_kernel.py)
                                                      v
                                              +---------------+
                                              |  Kaggle API   |
                                              +-------+-------+
                                                      |
                                                      v
                                        +---------------------------+
                                        | Kaggle GPU Worker (T4/P100|
                                        +-------------+-------------+
                                                      |
                      +-------------------------------+-------------------------------+
                      |                                                               |
                      v                                                               v
         3. Mount Secrets Dataset                                       4. Publish Tunnel & Errors
       (/kaggle/input/swapedev-secrets-*)                                (swapedev:{profile}:*)
                      |                                                               |
                      v                                                               v
         [ Cloudflared Tunnel URL ]                                           [ Upstash Redis ]
                      |                                                               |
                      +-------------------------------+-------------------------------+
                                                      |
                                                      v
                                        5. Poll Tunnel / Status
                                        (Direct HTTP Proxy & Dispatch)
```

### Key Components

- **`swapedev_service/`**: Core orchestrator application.
  - `worker_manager.py`: Manages profile configs, dataset packaging, Kaggle kernel launch, Upstash tunnel discovery, idle watchdogs, and teardown.
  - `queue_manager.py`: Job queue and worker assignment logic.
  - `worker_routes.py` & `main.py`: FastAPI endpoints for worker lifecycle, job dispatch, and health checks.
  - `ui_app.py`: Streamlit management dashboard.
  - `tests/`: Comprehensive unit and live integration test suite.
- **`scripts/kaggle_kernel.py`**: Self-contained worker script executed on Kaggle GPU. Loads secrets from the dataset mount, initializes local models/FastAPI server, spawns a cloudflared tunnel, and publishes registration state to Upstash Redis.
- **`scripts/live_boot_verifier.py`**: Standalone verification script for live end-to-end boot tests.
- **`configs/kernel-metadata.json`**: Kaggle kernel metadata configuration template.

---

## Quick Start

### 1. Environment Configuration

Copy `.env.example` to `.env` and fill in credentials:

```bash
cp .env.example .env
```

Required variables:
- `KAGGLE_USERNAME`: Your Kaggle account username.
- `KAGGLE_KEY`: Your Kaggle API token.
- `UPSTASH_REDIS_REST_URL`: Upstash REST endpoint URL.
- `UPSTASH_REDIS_REST_TOKEN`: Upstash REST token.
- `SWAPEDEV_WEBHOOK_TOKEN`: Shared secret for worker-orchestrator webhook authentication.
- `AVIDO_PROFILES_DIR`: Directory containing browser/device profiles.

### 2. Running Tests

Run the test suite with pytest:

```bash
pytest
```

To run the live Kaggle GPU integration test:

```bash
pytest swapedev_service/tests/test_worker_real_integration.py -m integration -s
```

### 3. Launching the Orchestrator & UI

Launch the UI dashboard:

```bash
./run_ui.sh
```

Or start the FastAPI orchestrator directly:

```bash
uvicorn swapedev_service.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Security

- Secrets are never embedded in kernel code or logs.
- Profile-specific secrets are dynamically created as ephemeral private Kaggle datasets (`swapedev-secrets-{profile_id}`) and attached directly to worker kernels.
- Cloudflare quick tunnels provide HTTPS ingress with no exposed local ports.
- Sensitive environment variables (`.env`, `secrets.json`, credentials) are strictly ignored by `.gitignore`.

---

## License

MIT License.
