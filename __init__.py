import os as _os

# Suppress huggingface_hub progress bars and transformers warnings.
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# tqdm creates a TMonitor daemon thread (interval default = 10s) that
# polls progress bars. After Metal GPU inference, a late GPU callback can
# fire on that thread without the GIL held → Fatal Python error.
#
# The `monitor_interval` CLASS ATTRIBUTE must be set to 0 IN PYTHON —
# there is no env-var for it (TQDM_MONITOR_INTERVAL is not read by tqdm).
# This prevents the thread from ever being created when a progress bar
# is instantiated later by huggingface_hub / snapshot_download.
try:
    import tqdm
    tqdm.tqdm.monitor_interval = 0
    # Kill any already-running monitor (belt-and-suspenders)
    if getattr(tqdm.tqdm, "monitor", None) is not None:
        try:
            tqdm.tqdm.monitor.kill()
        except Exception:
            pass
        tqdm.tqdm.monitor = None
except Exception:
    pass

__version__ = "0.1.0"
