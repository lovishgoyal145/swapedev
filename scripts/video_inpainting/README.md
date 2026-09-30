# SwapeDev: Video Character Inpainting Pipeline on Kaggle GPU

This module implements an automated, headless video-to-video character replacement pipeline designed to run on a free Kaggle GPU notebook (T4 / P100, 15GB–16GB VRAM) and integrate with the SwapeDev orchestrator.

---

## Architecture Overview

```
                                    +-----------------------------------+
                                    |       Caller / SwapeDev UI        |
                                    |    (POST /process-shot Request)   |
                                    +-----------------+-----------------+
                                                      |
                                                      v
                                        [ Cloudflared / pyngrok ]
                                                      |
                                                      v
+-------------------------------------------------------------------------------------------------------+
| Kaggle GPU Worker (T4 / P100 - 15GB/16GB VRAM)                                                         |
|                                                                                                       |
|  [Stage 1: Video Ingestion]                                                                           |
|   - Load MP4 clip (sub-5s, 24fps) -> Resize to vertical mobile aspect (512x896 / 576x1024)            |
|   - Isolate original audio track via FFmpeg                                                           |
|                                                                                                       |
|  [Stage 2: Auto-Segmentation - SAM 2]                                                                 |
|   - Track central human actor across all frames -> Output mask tensor [N, H, W]                       |
|   - Apply morphological dilation (3-5px) & Gaussian feathering                                        |
|   - Unload SAM 2 from VRAM immediately                                                                |
|                                                                                                       |
|  [Stage 3: ControlNet Preprocessing]                                                                  |
|   - DWPose: Skeletal landmarks & pose dynamics extraction                                             |
|   - Depth Anything V2: Volumetric metric depth extraction                                             |
|   - Unload preprocessors from VRAM immediately                                                        |
|                                                                                                       |
|  [Stage 4: Diffusion Pass - SD 1.5 + AnimateDiff V3 + MultiControlNet]                                |
|   - Base Model: SD 1.5 (RealisticVision V6.0 or DreamShaper 8)                                        |
|   - Motion Module: AnimateDiff V3 with FreeNoise sliding context window (16 frames, overlap 4)        |
|   - Conditioning: DWPose (0.85 weight) + Depth (0.70 weight) + Character LoRA (e.g. Spider-Man 0.85)  |
|   - VRAM Memory Controls: CPU offloading, VAE slicing, VAE tiling, FreeNoise split inference          |
|                                                                                                       |
|  [Stage 5: Compositing & Video Assembly]                                                              |
|   - Alpha-composite character over original background: I_comp = I_inpaint * M + I_orig * (1 - M)     |
|   - Stitch frames and re-mux original audio into final MP4 using FFmpeg                               |
+-------------------------------------------------+-----------------------------------------------------+
                                                  |
                                                  v
                                     [ Artifact MP4 Export ]
```

---

## File Structure

```
scripts/video_inpainting/
├── setup_kaggle.sh          # Kaggle environment bootstrapper (apt packages, pip wheels, weights downloader)
├── kaggle_bootstrap.ipynb   # 1-click Jupyter notebook ready to import directly into Kaggle GPU
├── pipeline.py              # Standalone video processing pipeline with explicit VRAM management
├── server.py                # FastAPI HTTP service with pyngrok/cloudflared and SwapeDev webhook
├── sam2_hiera_b+.yaml       # Configuration file for SAM 2 Base Plus model
├── test_video_pipeline.py   # Complete unit and integration test suite
└── README.md                # Documentation and operational guide
```

---

## 1. Quick Start on Kaggle GPU

### Option A: Using the Jupyter Notebook (`kaggle_bootstrap.ipynb`)
1. Create a new notebook on [Kaggle](https://www.kaggle.com/code).
2. Set the accelerator to **GPU T4 x2** or **GPU P100** and enable **Internet**.
3. Upload `kaggle_bootstrap.ipynb` or copy-paste the cells into your notebook.
4. Run the setup cell (`!bash setup_kaggle.sh`).
5. Run the server cell (`!python3 server.py`).

### Option B: Running via Terminal / Bash
In the Kaggle notebook cell:
```bash
!bash /kaggle/working/scripts/video_inpainting/setup_kaggle.sh
!python3 /kaggle/working/scripts/video_inpainting/server.py
```

The script will automatically:
- Install FFmpeg, diffusers, controlnet_aux, SAM 2, FastAPI, and pyngrok.
- Download or mount pre-staged model weights from `/kaggle/input/swapedev-base-models/`.
- Establish a public reverse tunnel via Cloudflare Quick Tunnel or pyngrok.
- Output your public API endpoint URL (e.g., `https://random-name.trycloudflare.com`).

---

## 2. API Reference

### `POST /process-shot`
Submits a video clip for character replacement.

#### Request Body (JSON)
```json
{
  "video_url": "https://example.com/sample_dance.mp4",
  "character_lora": "spiderman",
  "prompt": "spiderman suit, highly detailed, cinematic lighting, marvel superhero, 8k",
  "negative_prompt": "deformed, blurry, bad anatomy, human face, glitch, boiling artifacts",
  "denoise": 0.8,
  "num_inference_steps": 25,
  "guidance_scale": 7.0,
  "lora_weight": 0.85,
  "width": 512,
  "height": 896,
  "return_format": "url"
}
```

#### Alternative: Direct Base64 Upload
```json
{
  "video_base64": "<base64_encoded_mp4_bytes>",
  "character_lora": "spiderman",
  "prompt": "spiderman suit, marvel comic style",
  "denoise": 0.8
}
```

#### Response Body (JSON)
```json
{
  "status": "success",
  "video_url": "https://xxx.trycloudflare.com/outputs/inpainted_1727670000000.mp4",
  "output_filename": "inpainted_1727670000000.mp4",
  "duration_seconds": 18.42,
  "width": 512,
  "height": 896,
  "error": null
}
```

### `GET /outputs/{filename}`
Streams the rendered artifact MP4 directly to the client.

### `GET /health`
Returns service readiness and compute availability:
```json
{
  "status": "healthy",
  "service": "SwapeDev Video Character Inpainting",
  "torch_available": true,
  "cuda_available": true,
  "tunnel_url": "https://xxx.trycloudflare.com"
}
```

### `GET /gpu-status`
Returns real-time CUDA VRAM memory telemetry:
```json
{
  "cuda_available": true,
  "device_name": "Tesla T4",
  "allocated_gb": 4.12,
  "reserved_gb": 6.85,
  "max_allocated_gb": 9.40,
  "total_memory_gb": 15.78
}
```

### `POST /vram/clear`
Manually flushes the PyTorch CUDA cache and triggers garbage collection.

---

## 3. CLI Standalone Usage

You can also run the pipeline directly from the command line:

```bash
python3 scripts/video_inpainting/pipeline.py \
    --input my_video.mp4 \
    --output output_character.mp4 \
    --prompt "spiderman suit, highly detailed, marvel superhero, 8k" \
    --character-lora "/kaggle/working/models/loras/spiderman_classic_sd15.safetensors" \
    --lora-weight 0.85 \
    --denoise 0.80 \
    --steps 25 \
    --cfg 7.0 \
    --width 512 \
    --height 896
```

---

## 4. SwapeDev Orchestrator Integration

Inside the SwapeDev service or client scripts:

```python
from swapedev_service.video_worker_client import VideoInpaintingWorkerClient

async def run():
    client = VideoInpaintingWorkerClient("https://xxx.trycloudflare.com")
    
    # 1. Verify health
    healthy = await client.is_healthy()
    print("Worker healthy:", healthy)
    
    # 2. Process shot
    result = await client.process_shot(
        video_url="https://example.com/actor_clip.mp4",
        prompt="spiderman suit, highly detailed, cinematic lighting",
        character_lora="spiderman",
        denoise=0.80,
    )
    print("Inpainted video URL:", result["video_url"])
    
    # 3. Download locally
    local_path = await client.download_output_video(
        result["video_url"],
        "/path/to/local/output.mp4"
    )
    print("Saved to:", local_path)
```

---

## 5. VRAM Budget & OOM Prevention

Running AnimateDiff with two ControlNets and SAM 2 on a single 15GB–16GB GPU requires strict staging:

| Stage | Memory Allocation | Actions |
| :--- | :--- | :--- |
| **Ingestion** | Minimal (CPU/RAM) | Conforms video to 24fps and 512x896; extracts audio. |
| **SAM 2 Tracking** | ~1.5 GB VRAM | Extracts binary actor masks; dilates & feathers edges; immediately unloaded via `del predictor` and `torch.cuda.empty_cache()`. |
| **DWPose + Depth** | ~1.2 GB VRAM | Extracts skeletal and volumetric depth conditioning; immediately unloaded prior to diffusion. |
| **Diffusion Pass** | ~7–9 GB VRAM | Loaded in `torch.float16` with `enable_model_cpu_offload()`, `vae.enable_slicing()`, `vae.enable_tiling()`, and FreeNoise split inference. |
| **Compositing** | Pure CPU (NumPy) | Blends character onto plate using inverse mask; re-muxes audio via FFmpeg. |
| **Post-Run Cleanup** | 0 GB VRAM | Automatic `finally` block evicts all cached tensors and runs `gc.collect()`. |

If a memory spike occurs, the server catches `torch.cuda.OutOfMemoryError`, purges VRAM, and returns an actionable HTTP 503 response.
