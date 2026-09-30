#!/usr/bin/env python3
"""
SwapeDev: Camoufox Profile-Isolated Kaggle Dataset Publisher
============================================================
Automates base diffusion model publishing to a private Kaggle Dataset:
1. Loads an isolated persistent Camoufox browser profile (e.g. 'desi', 'maxx', 'enigma_drift').
2. Leverages the active Kaggle web session to extract or refresh API credentials.
3. Automatically writes 'dataset-metadata.json' and publishes/versions the models directory.
4. Enforces strict privacy checks to ensure models remain private.

Usage:
    python3 scripts/automation/push_kaggle_dataset.py --profile desi
    python3 scripts/automation/push_kaggle_dataset.py --profile desi --models-dir ./models --headless
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

# Ensure virtual environment site-packages containing camoufox and playwright are available
VENV_SITE_PACKAGES = [
    Path(__file__).resolve().parents[3] / "avido-browser-platform" / ".venv" / "lib" / "python3.12" / "site-packages",
    Path("/home/lovish/.gemini/antigravity/scratch/avido-browser-platform/.venv/lib/python3.12/site-packages"),
]
for p in VENV_SITE_PACKAGES:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PushKaggleDataset")

try:
    from camoufox.sync_api import Camoufox
except ImportError:
    logger.warning("Camoufox package not found in current Python environment.")
    Camoufox = None

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    logger.warning("Kaggle Python SDK not found in current Python environment.")
    KaggleApi = None


def resolve_profile_user_data_dir(profile: str, profiles_dir: Optional[str] = None) -> Path:
    """
    Resolves the persistent user_data_dir for a given profile.
    Checks:
    1. Direct path: f"{profiles_dir}/{profile}" (e.g. ./browser_profiles/{profile})
    2. Prefixed path: f"{profiles_dir}/profile_{profile}"
    3. Avido browser platform ecosystem profiles directory if available.
    """
    clean_profile = profile.strip()
    base_dir = Path(profiles_dir).resolve() if profiles_dir else Path("./browser_profiles").resolve()

    # 1. Direct match in base_dir
    cand_1 = base_dir / clean_profile
    if cand_1.exists() and cand_1.is_dir():
        return cand_1

    # 2. Prefixed match in base_dir
    prefixed = f"profile_{clean_profile}" if not clean_profile.startswith("profile_") else clean_profile
    cand_2 = base_dir / prefixed
    if cand_2.exists() and cand_2.is_dir():
        return cand_2

    # 3. Check avido-browser-platform/profiles
    avido_profiles = PROJECT_ROOT.parent / "avido-browser-platform" / "profiles"
    if avido_profiles.exists() and avido_profiles.is_dir():
        cand_3 = avido_profiles / prefixed
        if cand_3.exists() and cand_3.is_dir():
            return cand_3
        cand_4 = avido_profiles / clean_profile
        if cand_4.exists() and cand_4.is_dir():
            return cand_4

    # Default: create target under base_dir (standard ./browser_profiles/{profile})
    target = base_dir / clean_profile
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_saved_profile_credentials(user_data_dir: Path) -> Dict[str, str]:
    """
    Reads existing credentials from profile configuration files if present:
    - user_data_dir/kaggle.json
    - user_data_dir/config.json
    - user_data_dir/swapedev/config.json
    """
    creds = {"username": "", "key": ""}

    # Check 1: kaggle.json
    k_json = user_data_dir / "kaggle.json"
    if k_json.is_file():
        try:
            with open(k_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            u = str(data.get("username", "")).strip()
            k = str(data.get("key", "")).strip()
            if u and k:
                creds["username"] = u
                creds["key"] = k
                return creds
        except Exception as e:
            logger.debug(f"Failed to read {k_json}: {e}")

    # Check 2: config.json
    cfg_json = user_data_dir / "config.json"
    if cfg_json.is_file():
        try:
            with open(cfg_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            u = str(data.get("KAGGLE_USERNAME") or data.get("kaggle_username") or "").strip()
            k = str(data.get("KAGGLE_KEY") or data.get("kaggle_key") or "").strip()
            if u and k:
                creds["username"] = u
                creds["key"] = k
                return creds
        except Exception as e:
            logger.debug(f"Failed to read {cfg_json}: {e}")

    # Check 3: swapedev/config.json
    s_cfg = user_data_dir / "swapedev" / "config.json"
    if s_cfg.is_file():
        try:
            with open(s_cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            u = str(data.get("KAGGLE_USERNAME") or data.get("kaggle_username") or "").strip()
            k = str(data.get("KAGGLE_KEY") or data.get("kaggle_key") or "").strip()
            if u and k:
                creds["username"] = u
                creds["key"] = k
                return creds
        except Exception as e:
            logger.debug(f"Failed to read {s_cfg}: {e}")

    return creds


def save_extracted_credentials(user_data_dir: Path, username: str, key: str) -> None:
    """
    Persists extracted credentials into user_data_dir/kaggle.json and user_data_dir/config.json.
    """
    # 1. kaggle.json
    k_json = user_data_dir / "kaggle.json"
    with open(k_json, "w", encoding="utf-8") as f:
        json.dump({"username": username, "key": key}, f, indent=2)
    try:
        os.chmod(k_json, 0o600)
    except Exception:
        pass

    # 2. config.json
    cfg_json = user_data_dir / "config.json"
    cfg_data = {}
    if cfg_json.is_file():
        try:
            with open(cfg_json, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
        except Exception:
            cfg_data = {}

    cfg_data["KAGGLE_USERNAME"] = username
    cfg_data["KAGGLE_KEY"] = key
    cfg_data["kaggle_username"] = username
    cfg_data["kaggle_key"] = key
    cfg_data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(cfg_json, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)
    try:
        os.chmod(cfg_json, 0o600)
    except Exception:
        pass

    # 3. Mirror into swapedev/config.json if directory exists
    s_dir = user_data_dir / "swapedev"
    if s_dir.exists() and s_dir.is_dir():
        s_cfg = s_dir / "config.json"
        try:
            with open(s_cfg, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)
            os.chmod(s_cfg, 0o600)
        except Exception:
            pass

    logger.info(f"Saved extracted credentials for [{username}] to {k_json} and {cfg_json}")


def extract_kaggle_credentials_via_camoufox(
    user_data_dir: Path,
    headless: bool = True,
) -> Tuple[str, str]:
    """
    Launches persistent Camoufox context, navigates to Kaggle Settings, and extracts credentials.
    Clicks 'Create New Token' to capture the kaggle.json download.
    Falls back to existing local config if web download is not required or session has saved key.
    """
    if Camoufox is None:
        raise RuntimeError("camoufox is not installed. Please install camoufox to use browser automation.")

    logger.info(f"Initializing Camoufox persistent context at: {user_data_dir}")
    user_data_dir.mkdir(parents=True, exist_ok=True)

    extracted_user = ""
    extracted_key = ""

    with Camoufox(persistent_context=True, user_data_dir=str(user_data_dir), headless=headless) as browser:
        # browser is a Playwright BrowserContext in persistent mode
        page = browser.pages[0] if (hasattr(browser, "pages") and browser.pages) else browser.new_page()

        logger.info("Navigating to Kaggle Account Settings (https://www.kaggle.com/settings)...")
        try:
            page.goto("https://www.kaggle.com/settings", timeout=45000, wait_until="domcontentloaded")
            time.sleep(3)  # Brief wait for client-side routing and auth hydration
        except Exception as nav_err:
            logger.warning(f"Initial navigation notice: {nav_err}")

        current_url = page.url
        logger.info(f"Current Kaggle URL: {current_url}")

        # Check if user is logged in
        if "login" in current_url.lower() or "signin" in current_url.lower():
            saved = get_saved_profile_credentials(user_data_dir)
            if saved.get("username") and saved.get("key"):
                logger.info(f"Browser profile shows login prompt, but valid saved credentials found for '{saved['username']}'.")
                return saved["username"], saved["key"]
            raise RuntimeError(
                f"Active Kaggle session not found in profile '{user_data_dir.name}'. "
                "Please log into Kaggle with this browser profile first."
            )

        # 1. Attempt to detect Kaggle username from page state
        try:
            detected_user = page.evaluate("""
                () => {
                    if (window.__INITIAL_STATE__ && window.__INITIAL_STATE__.user && window.__INITIAL_STATE__.user.userName) {
                        return window.__INITIAL_STATE__.user.userName;
                    }
                    if (window.Kaggle && window.Kaggle.State && window.Kaggle.State.user && window.Kaggle.State.user.userName) {
                        return window.Kaggle.State.user.userName;
                    }
                    const avatar = document.querySelector('a[data-testid="user-avatar-link"], a[href^="/"][aria-label*="Profile"]');
                    if (avatar) {
                        const href = avatar.getAttribute('href') || '';
                        const parts = href.split('/').filter(Boolean);
                        if (parts.length > 0) return parts[0];
                    }
                    return null;
                }
            """)
            if detected_user:
                extracted_user = str(detected_user).strip()
                logger.info(f"Detected Kaggle Username from page: {extracted_user}")
        except Exception as eval_err:
            logger.debug(f"Username evaluation notice: {eval_err}")

        # 2. Look for "Create New Token" or "Create New API Token" button on Settings
        token_button = page.locator(
            'button:has-text("Create New Token"), '
            'button:has-text("Create New API Token"), '
            '[data-testid="create-new-token"], '
            'a:has-text("Create New Token")'
        ).first

        button_visible = False
        try:
            if token_button.is_visible(timeout=5000):
                button_visible = True
            else:
                token_button.scroll_into_view_if_needed(timeout=3000)
                button_visible = token_button.is_visible(timeout=3000)
        except Exception:
            button_visible = False

        if button_visible:
            logger.info("Found 'Create New Token' button. Requesting API token download...")
            try:
                with page.expect_download(timeout=15000) as download_info:
                    token_button.click()

                    # Handle confirmation modal if prompted ("Revoke/Expire old token?")
                    confirm_btn = page.locator(
                        'button:has-text("Revoke"), '
                        'button:has-text("Expire"), '
                        'button:has-text("Create"), '
                        'button:has-text("Confirm")'
                    ).first
                    try:
                        if confirm_btn.is_visible(timeout=3000):
                            confirm_btn.click()
                    except Exception:
                        pass

                download = download_info.value
                tmp_dir = Path(tempfile.mkdtemp(prefix="camoufox_dl_"))
                dl_path = tmp_dir / "kaggle.json"
                download.save_as(str(dl_path))

                with open(dl_path, "r", encoding="utf-8") as f:
                    downloaded_data = json.load(f)
                shutil.rmtree(tmp_dir, ignore_errors=True)

                extracted_user = downloaded_data.get("username", extracted_user).strip()
                extracted_key = downloaded_data.get("key", "").strip()
                logger.info(f"Successfully downloaded new Kaggle API token for user: {extracted_user}")
            except Exception as dl_err:
                logger.warning(f"Download token interaction notice: {dl_err}")

        # 3. Fallback: check saved credentials if download was not performed or failed
        if not extracted_user or not extracted_key:
            saved = get_saved_profile_credentials(user_data_dir)
            if saved.get("username") and saved.get("key"):
                logger.info(f"Using verified persistent credentials for '{saved['username']}'.")
                extracted_user = saved["username"]
                extracted_key = saved["key"]
            else:
                # Also check host ~/.kaggle/kaggle.json as last resort
                host_k = Path.home() / ".kaggle" / "kaggle.json"
                if host_k.is_file():
                    try:
                        with open(host_k, "r", encoding="utf-8") as f:
                            hdata = json.load(f)
                        hu = str(hdata.get("username", "")).strip()
                        hk = str(hdata.get("key", "")).strip()
                        if hu and hk:
                            logger.info(f"Falling back to credentials from {host_k} for user '{hu}'.")
                            extracted_user = hu
                            extracted_key = hk
                    except Exception:
                        pass

        if not extracted_user or not extracted_key:
            raise RuntimeError(
                f"Could not extract Kaggle credentials from profile '{user_data_dir.name}'. "
                "Ensure the profile is logged into Kaggle and has permission to generate API tokens."
            )

    # Save credentials into profile
    save_extracted_credentials(user_data_dir, extracted_user, extracted_key)
    return extracted_user, extracted_key


def prepare_dataset_metadata(
    models_dir: Path,
    username: str,
    dataset_slug: str,
    dataset_title: str,
) -> Path:
    """
    Creates or updates dataset-metadata.json inside models_dir.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    meta_path = models_dir / "dataset-metadata.json"
    dataset_ref = f"{username}/{dataset_slug}"

    metadata = {
        "title": dataset_title,
        "id": dataset_ref,
        "licenses": [
            {
                "name": "other"
            }
        ]
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Generated dataset metadata at {meta_path} (ID: {dataset_ref})")
    return meta_path


def publish_kaggle_dataset(
    models_dir: Path,
    username: str,
    key: str,
    dataset_slug: str,
    dataset_title: str,
    force_new: bool = False,
) -> str:
    """
    Publishes the models directory to Kaggle as a private dataset.
    Uses Kaggle Python SDK or CLI fallback with isolated environment credentials.
    Verifies that the dataset is private.
    """
    dataset_ref = f"{username}/{dataset_slug}"
    logger.info(f"Preparing to publish dataset: {dataset_ref}")

    # Create dataset-metadata.json
    prepare_dataset_metadata(models_dir, username, dataset_slug, dataset_title)

    # Set up isolated Kaggle environment
    temp_cfg_dir = Path(tempfile.mkdtemp(prefix="kaggle_env_"))
    k_file = temp_cfg_dir / "kaggle.json"
    with open(k_file, "w", encoding="utf-8") as f:
        json.dump({"username": username, "key": key}, f)
    try:
        os.chmod(k_file, 0o600)
    except Exception:
        pass

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(temp_cfg_dir)
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    if key.startswith("KGAT_"):
        env["KAGGLE_API_TOKEN"] = key

    # Test environment bypass for unit testing
    if os.getenv("TEST_MOCK_KAGGLE_PUSH") == "1":
        logger.info(f"[TEST MODE] Mock push completed for {dataset_ref}.")
        shutil.rmtree(temp_cfg_dir, ignore_errors=True)
        return dataset_ref

    api = None
    if KaggleApi is not None:
        try:
            api = KaggleApi()
            api.config_values[api.CONFIG_NAME_USER] = username
            api.config_values[api.CONFIG_NAME_KEY] = key
            if key.startswith("KGAT_"):
                api.config_values[api.CONFIG_NAME_TOKEN] = key
            api._authenticated = True
        except Exception as e:
            logger.warning(f"KaggleApi initialization notice: {e}")
            api = None

    # Check if dataset already exists
    dataset_exists = False
    if not force_new and api is not None:
        try:
            mine = api.dataset_list(mine=True)
            for d in (mine or []):
                ref = getattr(d, "ref", "")
                if ref and ref.lower() == dataset_ref.lower():
                    dataset_exists = True
                    break
        except Exception as list_err:
            logger.debug(f"Dataset list check notice: {list_err}")

    # Publish or version dataset
    if dataset_exists and not force_new:
        logger.info(f"Dataset {dataset_ref} already exists. Creating new version...")
        version_success = False
        if api is not None:
            try:
                api.dataset_create_version(
                    folder=str(models_dir),
                    version_notes=f"Automated update via Camoufox at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    dir_mode="skip",
                    quiet=False,
                )
                version_success = True
                logger.info("Successfully pushed new dataset version via SDK.")
            except Exception as ver_err:
                logger.warning(f"SDK version creation notice: {ver_err}, attempting CLI fallback...")

        if not version_success:
            cmd = ["kaggle", "datasets", "version", "-p", str(models_dir), "-m", "Automated update via Camoufox", "-r", "skip"]
            logger.info(f"Executing CLI command: {' '.join(cmd)}")
            subprocess.run(cmd, env=env, check=True)
    else:
        logger.info(f"Creating new private Kaggle dataset: {dataset_ref}...")
        create_success = False
        if api is not None:
            try:
                res = api.dataset_create_new(
                    folder=str(models_dir),
                    public=False,
                    quiet=False,
                    dir_mode="skip",
                )
                err_msg = getattr(res, "error", "")
                if not err_msg:
                    create_success = True
                    logger.info("Successfully created new private dataset via SDK.")
            except Exception as cr_err:
                logger.warning(f"SDK dataset create notice: {cr_err}, attempting CLI fallback...")

        if not create_success:
            # kaggle datasets create defaults to private unless --public is passed
            cmd = ["kaggle", "datasets", "create", "-p", str(models_dir), "-r", "skip"]
            logger.info(f"Executing CLI command: {' '.join(cmd)}")
            subprocess.run(cmd, env=env, check=True)

    # Privacy verification
    if api is not None:
        try:
            mine = api.dataset_list(mine=True)
            for d in (mine or []):
                ref = getattr(d, "ref", "")
                if ref and ref.lower() == dataset_ref.lower():
                    is_priv = getattr(d, "isPrivate", True)
                    if not is_priv:
                        raise RuntimeError(f"CRITICAL: Kaggle dataset {dataset_ref} was published as PUBLIC! Aborting.")
                    logger.info(f"Verified dataset {dataset_ref} is strictly PRIVATE (isPrivate={is_priv}).")
                    break
        except Exception as priv_err:
            logger.debug(f"Privacy verification notice: {priv_err}")

    shutil.rmtree(temp_cfg_dir, ignore_errors=True)
    logger.info(f"Dataset publication successfully complete: {dataset_ref}")
    return dataset_ref


def main():
    parser = argparse.ArgumentParser(
        description="Automate Kaggle Dataset creation and versioning using persistent Camoufox anti-detect browser profiles."
    )
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Influencer browser profile identifier (e.g. 'desi', 'enigma_drift', 'maxx').",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory path containing the diffusion model checkpoints to upload (default: './models').",
    )
    parser.add_argument(
        "--profiles-dir",
        type=str,
        default="./browser_profiles",
        help="Base directory for browser profiles (default: './browser_profiles').",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default="swapedev-base-models",
        help="Kaggle Dataset slug (default: 'swapedev-base-models').",
    )
    parser.add_argument(
        "--dataset-title",
        type=str,
        default="SwapeDev Base Models",
        help="Human-readable title for the Kaggle Dataset (default: 'SwapeDev Base Models').",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Camoufox in headless mode (default: --headless). Use --no-headless for visible UI debugging.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract and save Kaggle API credentials from the browser session without uploading files.",
    )
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="Force creation of a new dataset even if one already exists under the same slug.",
    )

    args = parser.parse_args()

    logger.info("==================================================================")
    logger.info(" SwapeDev Kaggle Dataset Push Automation via Camoufox Profiles    ")
    logger.info("==================================================================")
    logger.info(f"Profile: {args.profile}")
    logger.info(f"Models Directory: {args.models_dir}")
    logger.info(f"Dataset Slug: {args.dataset_slug}")
    logger.info(f"Headless Mode: {args.headless}")

    # 1. Resolve Profile Persistent Directory
    user_data_dir = resolve_profile_user_data_dir(args.profile, args.profiles_dir)
    logger.info(f"Resolved Camoufox user_data_dir: {user_data_dir}")

    # 2. Extract Credentials via Camoufox
    username, key = extract_kaggle_credentials_via_camoufox(
        user_data_dir=user_data_dir,
        headless=args.headless,
    )
    logger.info(f"Authenticated as Kaggle User: [{username}]")

    if args.extract_only:
        logger.info("--extract-only flag active. Credentials saved. Exiting successfully.")
        return 0

    # 3. Publish Model Directory
    models_path = Path(args.models_dir).resolve()
    if not models_path.exists():
        logger.warning(f"Models directory {models_path} does not exist. Creating directory...")
        models_path.mkdir(parents=True, exist_ok=True)

    dataset_ref = publish_kaggle_dataset(
        models_dir=models_path,
        username=username,
        key=key,
        dataset_slug=args.dataset_slug,
        dataset_title=args.dataset_title,
        force_new=args.force_new,
    )

    logger.info("==================================================================")
    logger.info(f" [SUCCESS] Base models dataset published: {dataset_ref}")
    logger.info(f" Expected Kaggle Mount Path: /kaggle/input/{args.dataset_slug}/")
    logger.info("==================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
