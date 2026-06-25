"""
Phase 3, Step III.d — Normalize counts to Reads Per Million (RPM).

Consumes the Step III.c outputs (built by p3_a_c_build_phase3_inputs.py) and
normalizes the screen and pDNA count matrices INDEPENDENTLY. Per the roadmap,
screens and pDNA baselines must not be normalized together.

    RPM = (count / column_sum) * 1e6   (computed per replicate / column)

Inputs  (data/processed/phase3_inputs/):
    raw_screen_counts_pass.csv      - QC-passing Avana 2DS screen counts
    raw_pdna_avana_counts_pass.csv  - Avana pDNA baseline counts

Outputs (data/processed/phase3_inputs/):
    screen_rpm.csv      - RPM-normalized screen replicates
    pdna_avana_rpm.csv  - RPM-normalized Avana pDNA baselines

Run:
    python src/p3_d_normalize_rpm.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# The data folder lives on an external SSD (too large for the laptop drive).
# Override at runtime with:  export CRISPR_DATA_ROOT=/path/to/data
DEFAULT_DATA_ROOT = "/Volumes/nimas_usb/project_data_repo/crispr_ai_ml/data"
DATA_ROOT = Path(os.environ.get("CRISPR_DATA_ROOT", DEFAULT_DATA_ROOT))

PHASE3_DIR = DATA_ROOT / "processed" / "phase3_inputs"

GUIDE_ID_COL = "sgRNA_seq"   # first column of the III.c count matrices


def normalize_to_rpm(input_path: Path, output_path: Path) -> None:
    """Normalize a count matrix to RPM, one replicate (column) at a time."""
    if not input_path.exists():
        sys.exit(f"ERROR: input not found: {input_path}\n"
                 f"Run p3_a_c_build_phase3_inputs.py first to build the III.c outputs.")

    print(f"Loading {input_path.name} ...")
    # First column is the sgRNA identifier -> use it as the index.
    df = pd.read_csv(input_path, index_col=GUIDE_ID_COL)

    print(f"  matrix: {df.shape[0]} sgRNAs x {df.shape[1]} replicates")
    print("  normalizing to RPM (per replicate)...")
    col_sums = df.sum(axis=0)                 # total reads per replicate
    df_rpm = df.divide(col_sums, axis=1) * 1e6

    # Sanity check: each replicate's RPM column should sum to ~1e6.
    sums = df_rpm.sum(axis=0)
    print(f"  RPM column sums: min={sums.min():.1f}, max={sums.max():.1f} (expect ~1e6)")

    df_rpm.to_csv(output_path)
    print(f"  wrote {output_path.name}: {df_rpm.shape}\n")


def main() -> None:
    print("=== Phase 3 / Step III.d — RPM normalization ===")
    print(f"Working in {PHASE3_DIR}\n")

    # Screens and pDNA baselines are normalized SEPARATELY.
    normalize_to_rpm(
        PHASE3_DIR / "raw_screen_counts_pass.csv",
        PHASE3_DIR / "screen_rpm.csv",
    )
    normalize_to_rpm(
        PHASE3_DIR / "raw_pdna_avana_counts_pass.csv",
        PHASE3_DIR / "pdna_avana_rpm.csv",
    )

    print("Normalization complete.")


if __name__ == "__main__":
    main()
