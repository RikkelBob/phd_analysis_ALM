from pathlib import Path
import os
import sys

from dotenv import load_dotenv

# Repo root = the folder this file is in
PROJECT_ROOT = Path(__file__).resolve().parent

# Load machine-specific settings from .env, if it exists
load_dotenv(PROJECT_ROOT / ".env")

# If data location not specified by env, default to /data
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
DATA_DIR_RAW = Path(os.getenv("DATA_DIR_RAW", PROJECT_ROOT / "data_raw"))

DATA_DIR_EXP = DATA_DIR / "data_exp"
DATA_DIR_PP = DATA_DIR / "data_pp"
DATA_DIR_PSYCH = DATA_DIR / "data_psych"

# Outputs live inside the repo
FIGURE_DIR = PROJECT_ROOT / "figures"

# When analyses are redone, newly processed data lands here
DATA_DIR_PP_NEW = DATA_DIR / "data_pp_new"

DIR_TEMP = PROJECT_ROOT / "temp"

# Create output folder
for dir_mk in [FIGURE_DIR, DIR_TEMP, DATA_DIR_PP_NEW]:
    dir_mk.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Loading preprocessed (_PP) data
#   - SAVE new results to DATA_DIR_PP_NEW / <relative path>
#   - LOAD with resolve_pp(<relative path>): returns the shipped file in DATA_DIR_PP,
#     unless the same relative path also exists in DATA_DIR_PP_NEW, in which case the
#     user is asked which version to use (the answer is remembered for the session).
#   - Non-interactive runs (no terminal) / overrides: env var PP_SOURCE = "new" | "original" | "ask"
# ---------------------------------------------------------------------------
_pp_choice_cache = {}


def _has_content(p):
    p = Path(p)
    if p.is_dir():
        return any(p.iterdir())
    return p.exists()


def _describe(p):
    from datetime import datetime
    p = Path(p)
    try:
        mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "unknown"
    if p.is_dir():
        n = sum(1 for _ in p.iterdir())
        return f"folder, {n} item(s), last modified {mtime}"
    return f"file, last modified {mtime}"


def resolve_pp(*parts):
    """Path to load preprocessed data from, relative to the _PP root.

    Looks for the same relative path in DATA_DIR_PP_NEW (your own re-run results) and
    DATA_DIR_PP (shipped data). If only one exists it is returned; if both exist the
    user is asked which one to use.
    """
    rel = Path(*parts)
    old, new = DATA_DIR_PP / rel, DATA_DIR_PP_NEW / rel

    if not _has_content(new):
        return old
    if not _has_content(old):
        print(f"[data] '{rel}' only exists in your re-run results ({DATA_DIR_PP_NEW}); loading that.")
        return new
    if str(new) in _pp_choice_cache:
        return _pp_choice_cache[str(new)]

    mode = os.getenv("PP_SOURCE", "ask").strip().lower()
    if mode == "ask" and not sys.stdin.isatty():
        print(f"[data] '{rel}' exists in both locations but no terminal is available; "
              f"using the original data. Set PP_SOURCE=new to use your re-run results.")
        mode = "original"

    if mode == "ask":
        print(f"\n[data] Found two versions of '{rel}':")
        print(f"  [n] NEW      (your re-run results)  {new}\n        {_describe(new)}")
        print(f"  [o] ORIGINAL (shipped data)         {old}\n        {_describe(old)}")
        while True:
            ans = input("Which one should be loaded? [n]ew / [o]riginal (default: n): ").strip().lower()
            if ans in ("", "n", "new"):
                mode = "new"; break
            if ans in ("o", "original"):
                mode = "original"; break
            print("Please answer 'n' or 'o'.")

    chosen = new if mode == "new" else old
    print(f"[data] Loading '{rel}' from {'NEW' if chosen == new else 'ORIGINAL'}: {chosen}")
    _pp_choice_cache[str(new)] = chosen
    return chosen
