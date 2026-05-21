# Environment Setup

This folder contains everything needed to create and manage the conda environment for this project.

```
conda/
├── environment.yml   # Dependency definitions (edit this to add packages)
├── setup.sh          # Setup script for Linux / macOS
├── setup.bat         # Setup script for Windows
└── README.md
```

---

## Prerequisites

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) before running the setup scripts.

---

## Setting Up the Environment

Run the appropriate script **from the project root** (not from inside the `conda/` folder):

**Linux / macOS:**
```bash
./conda/setup.sh
```

**Windows:**
```bat
conda\setup.bat
```

Then activate the environment:
```bash
conda activate maze-project
```

The scripts are idempotent — running them again will update an existing environment to match the current `environment.yml`.

---

## Adding New Dependencies

Open `conda/environment.yml` and add your package in the appropriate section:

**Conda package** (preferred when available):
```yaml
dependencies:
  - numpy
  - your-new-package       # <-- add here
```

**Pip-only package** (use when not available on conda):
```yaml
  - pip:
      - pybullet
      - your-pip-package   # <-- add here
```

After editing, re-run the setup script to apply the changes:
```bash
./conda/setup.sh      # Linux / macOS
conda\setup.bat       # Windows
```
