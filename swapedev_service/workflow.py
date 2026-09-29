"""
SwapeDev ComfyUI Workflow Generator
====================================
Builds hardened, multi-tenant ComfyUI execution graphs with:
- Strictly blank pipeline: ZERO default generic prompts, zero shared stylistic LoRAs.
- Only accepts node parameters passed down explicitly from the profile's execution context.
- Unique execution salting to defeat stale face embedding cache.
"""

import uuid
import time
from typing import Dict, Any, Optional
from swapedev_service.schemas import SwapJobParams


def build_face_swap_workflow(
    input_image_name: str,
    source_face_image_name: str,
    params: Optional[SwapJobParams] = None,
    output_prefix: str = "SwapeDev",
    is_video: bool = False,
) -> Dict[str, Any]:
    """
    Constructs a ComfyUI prompt workflow payload for ReActor face swapping.
    Enforces absolute tenancy:
    - No generic stylistic prompts or shared LoRAs are injected.
    - Operates as a blank pipeline parameterized exclusively by tenant execution context.
    - Employs randomized salt token to defeat node-level face embedding caches.
    """
    if params is None:
        params = SwapJobParams()

    unique_run_id = f"{uuid.uuid4().hex[:8]}_{int(time.time())}"
    safe_prefix = f"{output_prefix}_{unique_run_id}"

    # ReActor face swap parameters exclusively from tenant context
    face_restore_model = params.face_restore_model or "GFPGANv1.4"
    codeformer_weight = float(params.codeformer_weight)
    detect_gender_input = str(params.detect_gender_input or "no")
    detect_gender_source = str(params.detect_gender_source or "no")
    input_faces_index = str(params.input_faces_index)
    source_faces_index = str(params.source_faces_index)

    prompt_graph: Dict[str, Any] = {
        # 1. Target Input Media (where face is swapped into)
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": input_image_name,
            },
        },
        # 2. Source Face (from profile's Identity Vault)
        "2": {
            "class_type": "LoadImage",
            "inputs": {
                "image": source_face_image_name,
            },
        },
        # 3. ReActor Face Swap Core Node
        # Blank pipeline: no hardcoded LoRA or generic prompt injections
        "3": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["1", 0],
                "source_image": ["2", 0],
                "swap_model": "inswapper_128.onnx",
                "facedetection": "retinaface_resnet50",
                "face_restore_model": face_restore_model,
                "face_restore_visibility": 1.0,
                "codeformer_weight": codeformer_weight,
                "detect_gender_input": detect_gender_input,
                "detect_gender_source": detect_gender_source,
                "input_faces_index": input_faces_index,
                "source_faces_index": source_faces_index,
                "console_log_level": 1,
                # Dynamic tenant salting token to defeat ReActor in-memory face cache
                "_anti_cache_salt": unique_run_id,
            },
        },
        # 4. Save Rendered Output
        "4": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": safe_prefix,
                "images": ["3", 0],
            },
        },
    }

    return prompt_graph
