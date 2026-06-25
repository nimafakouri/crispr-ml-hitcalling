"""
Build Phase-3 input files from the raw Avana CRISPR screen data.

This script is the productionized version of notebooks/01_exploration.ipynb.
It loads the three raw DepMap/Avana files, filters them down to the
QC-passing Avana 2DS screens and the matching Avana pDNA baselines, and
writes the 5 CSV files used downstream in phase 3.

Outputs (written to data/processed/phase3_inputs/):
    1. screen_metadata_pass.csv        - metadata for QC-passing Avana 2DS screen replicates
    2. pdna_metadata_all.csv           - metadata for all pDNA baseline replicates
    3. pdna_metadata_avana.csv         - metadata for Avana-only pDNA baselines
    4. raw_screen_counts_pass.csv      - raw sgRNA counts for the QC-passing screen replicates
    5. raw_pdna_avana_counts_pass.csv  - raw sgRNA counts for the Avana pDNA baselines

Run:
    python src/01_build_phase3_inputs.py
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

RAW_DIR = DATA_ROOT / "raw"
OUT_DIR = DATA_ROOT / "processed" / "phase3_inputs"

RAW_READCOUNTS = RAW_DIR / "AvanaRawReadcounts.csv"
GUIDE_MAP = RAW_DIR / "AvanaGuideMap.csv"
SCREEN_SEQ_MAP = RAW_DIR / "ScreenSequenceMap.csv"

GUIDE_ID_COL = "sgRNA_seq"   # renamed from the unnamed first column of the raw matrix


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_raw_data():
    """Load the three raw input files."""
    for path in (RAW_READCOUNTS, GUIDE_MAP, SCREEN_SEQ_MAP):
        if not path.exists():
            sys.exit(f"ERROR: required input file not found: {path}")

    print(f"Loading raw read counts: {RAW_READCOUNTS}")
    raw = pd.read_csv(RAW_READCOUNTS)

    print(f"Loading guide map:       {GUIDE_MAP}")
    guide = pd.read_csv(GUIDE_MAP)

    print(f"Loading screen seq map:  {SCREEN_SEQ_MAP}")
    seqmap = pd.read_csv(SCREEN_SEQ_MAP)

    # The first column of the raw matrix is the sgRNA sequence (unnamed on disk).
    raw = raw.rename(columns={"Unnamed: 0": GUIDE_ID_COL})

    print(f"  raw    : {raw.shape}")
    print(f"  guide  : {guide.shape}")
    print(f"  seqmap : {seqmap.shape}")
    return raw, guide, seqmap


# --------------------------------------------------------------------------- #
# Filter metadata
# --------------------------------------------------------------------------- #
def build_metadata_tables(seqmap):
    """Split the sequence map into QC-passing screens and pDNA baselines."""
    # QC-passing Avana 2-day screen replicates.
    screen_pass = seqmap[
        seqmap["SequenceID"].notna()
        & (seqmap["ScreenType"] == "2DS")
        & (seqmap["ExcludeFromCRISPRCombined"] == False)  # noqa: E712
        & (seqmap["Library"] == "Avana")
        & (seqmap["PassesQC"] == True)  # noqa: E712
    ].copy()

    # All pDNA baseline replicates (any library).
    pdna_pass = seqmap[
        seqmap["SequenceID"].notna()
        & (seqmap["ScreenType"] == "pDNA")
        & seqmap["pDNABatch"].notna()
    ].copy()

    # Avana-only pDNA baselines.
    pdna_pass_avana = pdna_pass[
        pdna_pass["pDNABatch"].str.startswith("Avana", na=False)
    ].copy()

    print(f"  screen_pass     : {screen_pass.shape}")
    print(f"  pdna_pass       : {pdna_pass.shape}")
    print(f"  pdna_pass_avana : {pdna_pass_avana.shape}")
    return screen_pass, pdna_pass, pdna_pass_avana


# --------------------------------------------------------------------------- #
# Subset count matrices
# --------------------------------------------------------------------------- #
def build_count_matrices(raw, screen_pass, pdna_pass_avana):
    """Subset the raw matrix to the screen and Avana-pDNA replicate columns."""
    screen_seqids = list(screen_pass["SequenceID"].unique())
    pdna_seqids = list(pdna_pass_avana["SequenceID"].unique())

    # Warn if any expected replicate column is missing from the raw matrix.
    missing_screen = [c for c in screen_seqids if c not in raw.columns]
    missing_pdna = [c for c in pdna_seqids if c not in raw.columns]
    if missing_screen:
        print(f"  WARNING: {len(missing_screen)} screen columns missing from raw: "
              f"{missing_screen[:10]}{' ...' if len(missing_screen) > 10 else ''}")
    if missing_pdna:
        print(f"  WARNING: {len(missing_pdna)} pDNA columns missing from raw: {missing_pdna}")

    screen_seqids = [c for c in screen_seqids if c in raw.columns]
    pdna_seqids = [c for c in pdna_seqids if c in raw.columns]

    raw_screen_pass = raw[[GUIDE_ID_COL] + screen_seqids]
    raw_pdna_avana_pass = raw[[GUIDE_ID_COL] + pdna_seqids]

    print(f"  raw_screen_pass     : {raw_screen_pass.shape}")
    print(f"  raw_pdna_avana_pass : {raw_pdna_avana_pass.shape}")
    return raw_screen_pass, raw_pdna_avana_pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=== Building Phase-3 inputs ===")
    raw, guide, seqmap = load_raw_data()

    print("\nFiltering metadata...")
    screen_pass, pdna_pass, pdna_pass_avana = build_metadata_tables(seqmap)

    print("\nBuilding count matrices...")
    raw_screen_pass, raw_pdna_avana_pass = build_count_matrices(
        raw, screen_pass, pdna_pass_avana
    )

    print(f"\nWriting outputs to {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "screen_metadata_pass.csv": screen_pass,
        "pdna_metadata_all.csv": pdna_pass,
        "pdna_metadata_avana.csv": pdna_pass_avana,
        "raw_screen_counts_pass.csv": raw_screen_pass,
        "raw_pdna_avana_counts_pass.csv": raw_pdna_avana_pass,
    }
    for name, df in outputs.items():
        path = OUT_DIR / name
        df.to_csv(path, index=False)
        print(f"  wrote {name}: {df.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
