# Data setup on macOS and Linux

All pipeline scripts read the data location from `CRISPR_DATA_ROOT`. The value
must name the directory containing `raw/` and `processed/`.

If the variable is not set, scripts use `data/` inside this repository.

## macOS (Google Drive for desktop)

```bash
export CRISPR_DATA_ROOT="/Users/nimafakouri/Library/CloudStorage/GoogleDrive-nima.fakouri@gmail.com/My Drive/informatics/data/active_project_data_repo/crispr_ai_ml_gd/data"
```

Mark `crispr_ai_ml_gd` as **Available offline** before running the pipeline.

## Linux

Download the data to a normal local directory with your preferred Google Drive
client (for example, `rclone`), then set the variable to that machine's path:

```bash
export CRISPR_DATA_ROOT="$HOME/data/crispr_ai_ml_gd/data"
```

For repeated use, add the appropriate `export` line to the shell profile on
each computer. Do not copy the macOS path to Linux.

To confirm the selected directory before a run:

```bash
python -c 'from src.data_paths import DATA_ROOT; print(DATA_ROOT)'
```
