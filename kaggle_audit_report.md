# Kaggle Diagnostic Audit Report

```text
============================================================
      SWAPEDEV KAGGLE DIAGNOSTIC AUDIT REPORT
      Execution Timestamp: 2026-09-29T14:58:27.960305
============================================================

[1] CREDENTIAL RESOLUTION AUDIT
  * Selected Profile     : maxx
  * Credential Source    : Camoufox profile 'maxx' (/home/lovish/.gemini/antigravity/scratch/avido-browser-platform/profiles/profile_maxx/config.json)
  * KAGGLE_USERNAME      : tester_maxx
  * KAGGLE_KEY           : test_kaggle_key_123
  * Key Type Prefix      : Legacy Key / Custom

[1.1] HOST CREDENTIAL HIJACKING CHECK (~/.kaggle)
  [!] WARNING: Host access_token exists: /home/lovish/.kaggle/access_token
      Token value: KGAT_81...735e
      (Risk: Kaggle Python SDK will default to this token unless KAGGLE_CONFIG_DIR or KAGGLE_API_TOKEN is isolated!)
  [!] WARNING: Host kaggle.json exists: /home/lovish/.kaggle/kaggle.json (user: avidok)

[2] KERNEL METADATA AUDIT
  * Metadata Path        : /home/lovish/.gemini/antigravity/scratch/SwapeDev/configs/kernel-metadata.json
  * Exact Contents:
      {
        "id": "tester_maxx/swapedev-backend",
        "title": "SwapeDev Backend",
        "code_file": "../scripts/kaggle_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": true,
        "enable_gpu": true,
        "enable_internet": true,
        "dataset_sources": [
          "avidok/swapedev-base-models",
          "tester_maxx/swapedev-secrets-maxx"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
      }

[2.1] METADATA INTEGRITY VERIFICATION
  * Expected 'id'        : tester_maxx/swapedev-backend
  * Actual 'id'          : tester_maxx/swapedev-backend
  * ID Strict Match      : PASSED
  * Kernel Title         : 'SwapeDev Backend'
  * Title Slug Derived   : 'swapedev-backend'

[3] KAGGLE PYTHON API QUERY
  [!] API Initialization / Auth Error: HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded with url: /v1/security.OAuthService/IntrospectToken (Caused by NameResolutionError("HTTPSConnection(host='api.kaggle.com', port=443): Failed to resolve 'api.kaggle.com' ([Errno -2] Name or service not known)"))

[4] DIAGNOSTIC ASSESSMENT & ROOT CAUSE
  [!] ISSUE: Host ~/.kaggle/access_token threatens to hijack Kaggle SDK credentials.
  [✓] Audit execution complete.
============================================================
```
