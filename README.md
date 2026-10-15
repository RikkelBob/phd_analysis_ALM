# ed_analysis

Neural and behavioural analysis code (dPCA, ROC, SVM, motion subspaces, behavioural statistics).
The Python package is `ed_analysis` (code in `neural_analysis/`, plus `config.py`).
The behavioural statistics call **R** (mixed models via `lme4`, `glmmTMB`, `emmeans`, …) through `rpy2`, so both Python and R must be installed.

Everything below is written for a machine with nothing installed. The commands were tested on
Ubuntu-family Linux with Python 3.12 (Linux Mint 22 / Ubuntu 24.04). macOS and Windows notes are at the end and are
**not** tested.

## 1. System requirements

| Needed | Version | Used for |
|---|---|---|
| Python | 3.10 or newer (3.12 tested) | all analyses |
| R | 4.1 or newer (4.3 tested) | mixed-model statistics via `rpy2` |
| git | any | the `subspaceID` submodule |
| C/C++ compiler, system libraries | – | building some Python/R packages (see 2) |

## 2. Install system packages (Ubuntu / Debian / Linux Mint)

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip \
    r-base r-base-dev build-essential cmake \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    libfreetype6-dev libpng-dev libtiff-dev libjpeg-dev
```

Check:

```bash
python3 --version     # must be 3.10 or newer
R --version | head -1 # must be 4.1 or newer
```

## 3. Install the Python dependencies

Run these from the repo root. Use a virtual environment so the project does not touch your system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

`pip install -e .` reads `pyproject.toml`, which is the single list of Python dependencies
(numpy, scipy, pandas, scikit-learn, matplotlib, statsmodels, mne, dPCA, rpy2, numba, …).
The packages are **not** version-pinned, so you get the newest releases. The last verified set was
numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, mne 1.13.2, numba 0.68.0, rpy2 3.6.9 and dPCA 1.0.5.
To record exactly what you have installed: `pip freeze --exclude-editable > requirements.lock.txt`.

Activate the environment (`source .venv/bin/activate`) in every new terminal before running any script.

The `subspaceID` git submodule (https://github.com/economolab/subspaceID) is only needed by
`neural_analysis/motion/null_potent_subspace/calc_np_subspaces.py`. Make sure it is checked out
(`git submodule update --init`) into `subspaceID/` in the repo root, and run that script from the repo root.

## 4. Install the R packages

The R scripts need these packages: `lme4`, `lmerTest`, `pbkrtest`, `glmmTMB`, `emmeans`, `car`, `performance`, `DHARMa`.

Start R (`R`) and run the following. It installs precompiled binaries from Posit Package Manager, so nothing needs compiling
and it takes a couple of minutes instead of an hour.

```r
# Find your Ubuntu codename first, in a shell:  grep UBUNTU_CODENAME /etc/os-release
# (Ubuntu 24.04 / Mint 22 = noble, Ubuntu 22.04 / Mint 21 = jammy). Replace "noble" below if yours differs.
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest"))

# Install Matrix first, so it matches the binaries below (see the Matrix warning under Troubleshooting)
install.packages("Matrix")

install.packages(c("lme4", "lmerTest", "pbkrtest", "glmmTMB",
                   "emmeans", "car", "performance", "DHARMa"))
```

If R asks whether to use a personal library, answer `yes`. Check that everything loads:

```bash
Rscript -e 'for (p in c("lme4","lmerTest","pbkrtest","glmmTMB","emmeans","car","performance","DHARMa")) cat(p, requireNamespace(p, quietly=TRUE), "\n")'
```

Every line must end in `TRUE`.

## 5. Point the code at your data

All paths are set in `config.py`. By default it looks for data inside the repo:

```
ed_analysis/
├── data/               # DATA_DIR      (override with DATA_DIR)
│   ├── data_exp/
│   ├── data_pp/        # preprocessed data
│   ├── data_psych/
│   └── data_pp_new/    # created automatically for newly re-run results
├── data_raw/           # DATA_DIR_RAW  (override with DATA_DIR_RAW)
├── figures/            # created automatically, all plots are saved here
└── temp/               # created automatically
```

If your data is somewhere else, copy the example settings file and edit it:

```bash
cp .env_example .env
```

then set `DATA_DIR=/path/to/your/data` (and `DATA_DIR_RAW=...` if used) in `.env`.
`.env` is machine-specific and is git-ignored.

When a result exists both in the shipped `data_pp/` and in your own `data_pp_new/`, scripts ask which version to load.
For non-interactive runs set `PP_SOURCE=new` or `PP_SOURCE=original` as an environment variable.

## 6. Check the installation

With the virtual environment active, from the repo root:

```bash
python - <<'EOF'
import numpy, scipy, pandas, sklearn, matplotlib, statsmodels, mne, pingouin, numba
from dPCA.dPCA import dPCA
import config, neural_analysis
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
for pkg in ["base", "lme4", "lmerTest", "glmmTMB", "emmeans", "car", "performance", "DHARMa"]:
    importr(pkg)
print("R", ro.r("R.version.string")[0])
print("All good.")
EOF
```

If this prints `All good.`, the installation is complete. Scripts are run directly, for example
`python neural_analysis/dPCA/dPCA_calc_subspaces.py`; figures are written to `figures/`.

## Troubleshooting

- **`ModuleNotFoundError: ... numexpr` when importing `dPCA.dPCA`**: `numexpr` is a hidden dependency of dPCA and is listed in
  `pyproject.toml`. Re-run `pip install -e .`.
- **`pip install` fails building `rpy2` or `numba`**: make sure you are on Python 3.10–3.12 and that `R` is on your `PATH`
  (`which R`). Very new Python releases sometimes lack wheels for `numba`.
- **R warns `lme4 was built with Matrix ABI version 1 / Current Matrix ABI version is 2`**: the system `Matrix` package is newer than
  the one the R binaries were built against. Run `install.packages("Matrix")`, then reinstall the packages that warn
  (`install.packages(c("lme4", "TMB", "glmmTMB"))`) using the Posit repository line from step 4, and restart R.
- **`ImportError: No module named 'subspaceID'`**: the submodule is missing or the repo root is not on `sys.path`
  (run from the repo root or set `PYTHONPATH` to it).
- **`ImportError: dpca_deviation`**: `dPCA_deviation_batch.py` imports a sibling file by bare name, so run it from
  `neural_analysis/dPCA/deviation_bsl/` or add that folder to `PYTHONPATH`.
- **Files not found**: check `DATA_DIR` in `.env`; the paths actually used are defined in `config.py`.

## macOS and Windows (not tested)

- **macOS**: `brew install git python r`, then follow steps 3 and 5–6. For step 4 use the standard CRAN mirror, which serves
  macOS binaries: `options(repos = c(CRAN = "https://cloud.r-project.org"))` instead of the Posit line.
- **Windows**: use WSL2 with Ubuntu 24.04 (`wsl --install -d Ubuntu-24.04`) and follow the Linux instructions inside it.
  `rpy2` and the R toolchain are much harder to set up on native Windows.
