"""
Shared Hugging Face Hub setup used by train/finetune_qlora.py,
train/merge_adapter.py, and eval/evaluate.py.

- Points the HF cache at a folder next to this file, so if you run these
  scripts from a Google Drive-mounted Colab notebook, downloaded weights
  persist across runtime restarts instead of re-downloading every session.
  (Trade-off: reads/writes to Drive are slower than local Colab disk. If
  you'd rather cache locally and accept a re-download after a disconnect,
  set HF_HOME yourself before importing this module and it will be left
  alone.)
- Reads HF_TOKEN from the environment and passes it to from_pretrained
  calls, avoiding the unauthenticated-request rate limit.
- from_pretrained_cached() tries local_files_only=True first, so a warm
  cache never even makes the network round-trip transformers normally
  does to check for updates - only downloads if the model truly isn't
  cached yet.

This module is imported by scripts run as `!python ...` subprocesses, not
inside the notebook's IPython kernel, so it cannot call
`google.colab.userdata.get()` directly (that API only works in the live
kernel process). If you're using Colab Secrets, fetch the secret in a
notebook cell and export it before invoking the script:

    from google.colab import userdata
    import os
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    !python train/finetune_qlora.py
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".hf_cache"

os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster, resumable downloads if hf_transfer is installed

HF_TOKEN = os.environ.get("HF_TOKEN")


def _patch_loss_kwargs_compat():
    """
    Phi-4-mini isn't in every transformers release's native model mapping,
    so loading it falls back to the model repo's bundled remote code even
    with trust_remote_code=False. That remote code imports `LossKwargs`
    from transformers.utils, a symbol later transformers releases renamed
    or removed. LossKwargs is a typing-only TypedDict (used for
    `**kwargs: Unpack[LossKwargs]` signatures) with no runtime behavior,
    so a no-op stand-in satisfies the import without pinning an exact
    transformers version, which would risk breaking peft/trl compatibility
    elsewhere.
    """
    import transformers.utils as _tu

    if not hasattr(_tu, "LossKwargs"):
        from typing import TypedDict

        class LossKwargs(TypedDict, total=False):
            pass

        _tu.LossKwargs = LossKwargs


_patch_loss_kwargs_compat()


def from_pretrained_cached(cls, model_id, **kwargs):
    """Load via `cls.from_pretrained`, preferring an already-downloaded local cache."""
    try:
        return cls.from_pretrained(model_id, local_files_only=True, token=HF_TOKEN, **kwargs)
    except OSError:
        print(f"'{model_id}' not found in local cache ({os.environ['HF_HOME']}) - downloading...")
        return cls.from_pretrained(model_id, local_files_only=False, token=HF_TOKEN, **kwargs)
