"""
Unit and Integration Tests: Kaggle Hot-Start Setup & Dataset Push Automation
=============================================================================
Tests:
1. setup_kaggle.sh Hot-Start path resolution (nested subdirectories)
2. setup_kaggle.sh Hot-Start path resolution (flat file layout)
3. setup_kaggle.sh Cold-Start fallback detection
4. push_kaggle_dataset.py profile directory resolution
5. push_kaggle_dataset.py metadata generation
6. push_kaggle_dataset.py credential persistence
7. push_kaggle_dataset.py dataset publishing flow
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import pytest

# Ensure scripts and service are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.automation.push_kaggle_dataset import (
    resolve_profile_user_data_dir,
    get_saved_profile_credentials,
    save_extracted_credentials,
    prepare_dataset_metadata,
    publish_kaggle_dataset,
)


@pytest.fixture
def temp_test_env():
    temp_dir = Path(tempfile.mkdtemp(prefix="test_hot_start_"))
    dataset_dir = temp_dir / "input" / "swapedev-base-models"
    workspace_dir = temp_dir / "working"
    dataset_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)

    yield {
        "root": temp_dir,
        "dataset_dir": dataset_dir,
        "workspace_dir": workspace_dir,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_hot_start_nested_directories(temp_test_env):
    """Verifies that nested subdirectories are symlinked and hot-start completes instantly."""
    dataset_dir = temp_test_env["dataset_dir"]
    workspace_dir = temp_test_env["workspace_dir"]

    # Populate dummy models in nested structure
    (dataset_dir / "sam2").mkdir(parents=True)
    (dataset_dir / "animatediff").mkdir(parents=True)
    (dataset_dir / "controlnet").mkdir(parents=True)
    (dataset_dir / "loras").mkdir(parents=True)

    (dataset_dir / "sam2" / "sam2_hiera_base_plus.pt").write_bytes(b"dummy_sam2")
    (dataset_dir / "sam2" / "sam2_hiera_b+.yaml").write_text("dummy_yaml: true")
    (dataset_dir / "animatediff" / "v3_sd15_mm.ckpt").write_bytes(b"dummy_animatediff")
    (dataset_dir / "controlnet" / "control_v11p_sd15_openpose.pth").write_bytes(b"dummy_pose")
    (dataset_dir / "controlnet" / "control_v11f1p_sd15_depth.pth").write_bytes(b"dummy_depth")
    (dataset_dir / "loras" / "spiderman_classic_sd15.safetensors").write_bytes(b"dummy_lora")

    env = os.environ.copy()
    env["DATASET_BASE"] = str(dataset_dir)
    env["WORKSPACE_DIR"] = str(workspace_dir)
    env["SKIP_PIP"] = "1"

    script_path = PROJECT_ROOT / "scripts" / "video_inpainting" / "setup_kaggle.sh"
    cmd = ["bash", "-c", f"source {script_path}"]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    assert result.returncode == 0
    assert "[HOT START] Mounted dataset detected. Bootstrapped in < 10s." in result.stdout

    # Verify symlinks
    models_dir = workspace_dir / "models"
    assert (models_dir / "sam2" / "sam2_hiera_base_plus.pt").is_symlink()
    assert (models_dir / "sam2" / "sam2_hiera_b+.yaml").is_symlink()
    assert (models_dir / "animatediff" / "v3_sd15_mm.ckpt").is_symlink()
    assert (models_dir / "controlnet" / "control_v11p_sd15_openpose.pth").is_symlink()
    assert (models_dir / "controlnet" / "control_v11f1p_sd15_depth.pth").is_symlink()
    assert (models_dir / "loras" / "spiderman_classic_sd15.safetensors").is_symlink()


def test_hot_start_flat_layout(temp_test_env):
    """Verifies that flat dataset structure is resolved and mapped to expected subdirectories."""
    dataset_dir = temp_test_env["dataset_dir"]
    workspace_dir = temp_test_env["workspace_dir"]

    # Populate dummy models in flat structure
    (dataset_dir / "sam2_hiera_base_plus.pt").write_bytes(b"flat_sam2")
    (dataset_dir / "sam2_hiera_b+.yaml").write_text("flat_yaml: true")
    (dataset_dir / "v3_sd15_mm.ckpt").write_bytes(b"flat_animatediff")
    (dataset_dir / "control_v11p_sd15_openpose.pth").write_bytes(b"flat_pose")
    (dataset_dir / "control_v11f1p_sd15_depth.pth").write_bytes(b"flat_depth")
    (dataset_dir / "spiderman_classic_sd15.safetensors").write_bytes(b"flat_lora")

    env = os.environ.copy()
    env["DATASET_BASE"] = str(dataset_dir)
    env["WORKSPACE_DIR"] = str(workspace_dir)
    env["SKIP_PIP"] = "1"

    script_path = PROJECT_ROOT / "scripts" / "video_inpainting" / "setup_kaggle.sh"
    cmd = ["bash", "-c", f"source {script_path}"]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    assert result.returncode == 0
    assert "[HOT START] Mounted dataset detected. Bootstrapped in < 10s." in result.stdout

    # Verify symlinks resolved into target subdirectories
    models_dir = workspace_dir / "models"
    assert (models_dir / "sam2" / "sam2_hiera_base_plus.pt").is_symlink()
    assert (models_dir / "animatediff" / "v3_sd15_mm.ckpt").is_symlink()
    assert (models_dir / "controlnet" / "control_v11p_sd15_openpose.pth").is_symlink()
    assert (models_dir / "controlnet" / "control_v11f1p_sd15_depth.pth").is_symlink()
    assert (models_dir / "loras" / "spiderman_classic_sd15.safetensors").is_symlink()


def test_cold_start_detection(temp_test_env):
    """Verifies that non-existent dataset triggers the cold-start branch."""
    workspace_dir = temp_test_env["workspace_dir"]
    fake_dataset = temp_test_env["root"] / "non_existent_dataset_path"

    env = os.environ.copy()
    env["DATASET_BASE"] = str(fake_dataset)
    env["WORKSPACE_DIR"] = str(workspace_dir)
    env["SKIP_PIP"] = "1"

    script_path = PROJECT_ROOT / "scripts" / "video_inpainting" / "setup_kaggle.sh"
    # Run with a short timeout to catch the cold start message before network download attempt
    proc = subprocess.Popen(
        ["bash", "-c", f"source {script_path}"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = ""
    try:
        # Read initial output lines
        for _ in range(30):
            line = proc.stdout.readline()
            if not line:
                break
            output += line
            if "[COLD START]" in output:
                break
    finally:
        proc.kill()
        proc.wait()

    assert "[COLD START] No mounted dataset detected" in output


def test_profile_user_data_dir_resolution(temp_test_env):
    """Tests resolution of profile user_data_dir with direct, prefixed, and default paths."""
    base_profiles = temp_test_env["root"] / "browser_profiles"
    base_profiles.mkdir(parents=True)

    # 1. Existing direct folder
    (base_profiles / "desi").mkdir()
    res1 = resolve_profile_user_data_dir("desi", str(base_profiles))
    assert res1 == (base_profiles / "desi").resolve()

    # 2. Existing prefixed folder
    (base_profiles / "profile_maxx").mkdir()
    res2 = resolve_profile_user_data_dir("maxx", str(base_profiles))
    assert res2 == (base_profiles / "profile_maxx").resolve()

    # 3. New folder creation
    res3 = resolve_profile_user_data_dir("new_profile", str(base_profiles))
    assert res3 == (base_profiles / "new_profile").resolve()
    assert res3.exists()


def test_metadata_generation(temp_test_env):
    """Verifies dataset-metadata.json content and schema."""
    models_dir = temp_test_env["workspace_dir"] / "models"
    meta_path = prepare_dataset_metadata(
        models_dir=models_dir,
        username="influencer_desi",
        dataset_slug="swapedev-base-models",
        dataset_title="SwapeDev Base Models",
    )

    assert meta_path.is_file()
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["id"] == "influencer_desi/swapedev-base-models"
    assert data["title"] == "SwapeDev Base Models"
    assert "licenses" in data


def test_credential_persistence_and_reading(temp_test_env):
    """Verifies credentials saving and reading logic in profile directory."""
    profile_dir = temp_test_env["root"] / "test_profile"
    profile_dir.mkdir(parents=True)

    save_extracted_credentials(
        user_data_dir=profile_dir,
        username="desi_creator",
        key="KGAT_mock1234567890abcdef",
    )

    creds = get_saved_profile_credentials(profile_dir)
    assert creds["username"] == "desi_creator"
    assert creds["key"] == "KGAT_mock1234567890abcdef"

    # Verify 0600 permissions
    k_file = profile_dir / "kaggle.json"
    assert k_file.exists()
    file_mode = oct(k_file.stat().st_mode & 0o777)
    assert file_mode == "0o600"


def test_publish_kaggle_dataset_mock_flow(temp_test_env, monkeypatch):
    """Verifies dataset publish pipeline under test mode."""
    monkeypatch.setenv("TEST_MOCK_KAGGLE_PUSH", "1")

    models_dir = temp_test_env["workspace_dir"] / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "dummy_weights.bin").write_bytes(b"12345")

    dataset_ref = publish_kaggle_dataset(
        models_dir=models_dir,
        username="desi_model_worker",
        key="KGAT_abcdef123456",
        dataset_slug="swapedev-base-models",
        dataset_title="SwapeDev Base Models",
    )

    assert dataset_ref == "desi_model_worker/swapedev-base-models"
    assert (models_dir / "dataset-metadata.json").is_file()
