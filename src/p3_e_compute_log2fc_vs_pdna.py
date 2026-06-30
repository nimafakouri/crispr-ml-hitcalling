"""
Phase 3, Step III.e - Compute log2 fold-change vs pDNA.

This script calculates sgRNA-level log2 fold-change for every QC-passing
screen replicate using its matched Avana pDNA batch baseline.

Formula:
    log2FC = log2((screen_RPM + pseudocount) / (pDNA_batch_RPM + pseudocount))

Inputs in data/processed/phase3_inputs/:
    screen_rpm.csv
    pdna_avana_rpm.csv
    screen_metadata_pass.csv
    pdna_metadata_avana.csv

Output in data/processed/phase3_inputs/:
    screen_log2fc_vs_pdna.csv

Run only after reviewing:
    python src/p3_e_compute_log2fc_vs_pdna.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_ROOT = Path("/Volumes/nimas_usb/project_data_repo/crispr_ai_ml/data")
PHASE3_DIR = DATA_ROOT / "processed" / "phase3_inputs"

SCREEN_RPM_PATH = PHASE3_DIR / "screen_rpm.csv"
PDNA_RPM_PATH = PHASE3_DIR / "pdna_avana_rpm.csv"
SCREEN_METADATA_PATH = PHASE3_DIR / "screen_metadata_pass.csv"
PDNA_METADATA_PATH = PHASE3_DIR / "pdna_metadata_avana.csv"
OUTPUT_PATH = PHASE3_DIR / "screen_log2fc_vs_pdna.csv"


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
GUIDE_COL = "sgRNA_seq"
REPLICATE_ID_COL = "SequenceID"
PDNA_BATCH_COL = "pDNABatch"

PSEUDOCOUNT = 1.0
RPM_EXPECTED_SUM = 1_000_000.0
RPM_SUM_TOLERANCE = 10.0

# screen_rpm.csv is large, so process it row-wise in chunks.
CHUNK_SIZE = 5_000


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def check_required_files(paths: list[Path]) -> None:
    """Fail early if any required input file is missing."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required input file(s):\n{missing_text}")


def require_columns(df: pd.DataFrame, columns: list[str], file_name: str) -> None:
    """Fail if a dataframe does not contain the columns this step needs."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing required column(s): {missing}")


def warn_if_rpm_sums_look_wrong(label: str, sums: pd.Series) -> None:
    """Print a warning if RPM column sums are not approximately 1,000,000."""
    bad = sums[~np.isclose(sums.to_numpy(float), RPM_EXPECTED_SUM, atol=RPM_SUM_TOLERANCE)]

    print(f"{label} RPM column sums:")
    print(f"  min={sums.min():.2f}")
    print(f"  max={sums.max():.2f}")

    if bad.empty:
        print("  OK: all columns sum to approximately 1,000,000")
    else:
        print(f"  WARNING: {len(bad)} column(s) do not sum to approximately 1,000,000")
        print(f"  first few problematic columns: {bad.head(10).round(2).to_dict()}")


def update_summary(summary: dict, values: np.ndarray) -> None:
    """Update running sanity-check statistics for the log2FC matrix."""
    flat = values.ravel()
    finite = flat[np.isfinite(flat)]

    summary["total_values"] += flat.size
    summary["missing_values"] += int(np.isnan(flat).sum())
    summary["infinite_values"] += int(np.isinf(flat).sum())

    if finite.size == 0:
        return

    summary["finite_values"] += finite.size
    summary["finite_sum"] += float(finite.sum())
    summary["min"] = min(summary["min"], float(finite.min()))
    summary["max"] = max(summary["max"], float(finite.max()))

    # Keep a small sample for a practical median sanity check without loading
    # the full output matrix into memory.
    stride = max(1, finite.size // 100_000)
    summary["median_sample"].append(finite[::stride][:100_000])


def print_log2fc_summary(summary: dict) -> None:
    """Print final sanity-check statistics for the computed log2FC values."""
    if summary["finite_values"] == 0:
        print("\nlog2FC sanity checks:")
        print("  WARNING: no finite log2FC values were calculated")
        print(f"  total values       : {summary['total_values']}")
        print(f"  missing values     : {summary['missing_values']}")
        print(f"  infinite values    : {summary['infinite_values']}")
        return

    mean = summary["finite_sum"] / summary["finite_values"]
    median_sample = np.concatenate(summary["median_sample"])

    print("\nlog2FC sanity checks:")
    print(f"  total values       : {summary['total_values']}")
    print(f"  missing values     : {summary['missing_values']}")
    print(f"  infinite values    : {summary['infinite_values']}")
    print(f"  min                : {summary['min']:.3f}")
    print(f"  max                : {summary['max']:.3f}")
    print(f"  mean               : {mean:.3f}")
    print(f"  sampled median     : {np.median(median_sample):.3f}")


# --------------------------------------------------------------------------- #
# Load files
# --------------------------------------------------------------------------- #
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load pDNA RPM, metadata tables, and screen RPM header."""
    check_required_files(
        [
            SCREEN_RPM_PATH,
            PDNA_RPM_PATH,
            SCREEN_METADATA_PATH,
            PDNA_METADATA_PATH,
        ]
    )

    print("Loading input files...")
    pdna_rpm = pd.read_csv(PDNA_RPM_PATH)
    screen_metadata = pd.read_csv(SCREEN_METADATA_PATH)
    pdna_metadata = pd.read_csv(PDNA_METADATA_PATH)
    screen_header = pd.read_csv(SCREEN_RPM_PATH, nrows=0).columns.tolist()

    require_columns(pdna_rpm, [GUIDE_COL], PDNA_RPM_PATH.name)
    require_columns(screen_metadata, [REPLICATE_ID_COL, PDNA_BATCH_COL], SCREEN_METADATA_PATH.name)
    require_columns(pdna_metadata, [REPLICATE_ID_COL, PDNA_BATCH_COL], PDNA_METADATA_PATH.name)

    if GUIDE_COL not in screen_header:
        raise ValueError(f"{SCREEN_RPM_PATH.name} is missing required column: {GUIDE_COL}")

    pdna_rpm = pdna_rpm.set_index(GUIDE_COL)
    screen_sample_cols = [col for col in screen_header if col != GUIDE_COL]

    print(f"  pdna_rpm shape        : {pdna_rpm.shape}")
    print(f"  screen_metadata shape : {screen_metadata.shape}")
    print(f"  pdna_metadata shape   : {pdna_metadata.shape}")
    print(f"  screen sample columns : {len(screen_sample_cols)}")
    print(f"  pDNA sample columns   : {len(pdna_rpm.columns)}")

    return pdna_rpm, screen_metadata, pdna_metadata, screen_sample_cols


# --------------------------------------------------------------------------- #
# Build pDNA baselines
# --------------------------------------------------------------------------- #
def build_pdna_baselines(pdna_rpm: pd.DataFrame, pdna_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Build one baseline vector per pDNA batch.

    For each pDNABatch, this takes the median RPM across pDNA replicate columns.
    The output has sgRNAs as rows and pDNA batches as columns.
    """
    warn_if_rpm_sums_look_wrong("pDNA", pdna_rpm.sum(axis=0))

    baselines = {}

    print("\nBuilding pDNA baseline vectors...")
    for batch, batch_metadata in pdna_metadata.groupby(PDNA_BATCH_COL):
        batch_cols = [
            seq_id
            for seq_id in batch_metadata[REPLICATE_ID_COL].dropna().unique()
            if seq_id in pdna_rpm.columns
        ]

        if not batch_cols:
            raise ValueError(f"No pDNA RPM columns found for pDNA batch: {batch}")

        baselines[batch] = pdna_rpm[batch_cols].median(axis=1)
        print(f"  {batch}: median across {len(batch_cols)} pDNA column(s)")

    return pd.DataFrame(baselines, index=pdna_rpm.index)


# --------------------------------------------------------------------------- #
# Match screen replicates to pDNA batches
# --------------------------------------------------------------------------- #
def build_screen_to_pdna_batch_map(
    screen_sample_cols: list[str],
    screen_metadata: pd.DataFrame,
    available_pdna_batches: set[str],
) -> dict[str, str]:
    """Build a dictionary mapping each screen replicate column to pDNABatch."""
    metadata = screen_metadata[[REPLICATE_ID_COL, PDNA_BATCH_COL]].dropna().copy()

    duplicated = metadata[REPLICATE_ID_COL].duplicated()
    if duplicated.any():
        examples = metadata.loc[duplicated, REPLICATE_ID_COL].head(10).tolist()
        raise ValueError(f"Duplicate screen replicate IDs in metadata, examples: {examples}")

    metadata_map = metadata.set_index(REPLICATE_ID_COL)[PDNA_BATCH_COL].to_dict()

    missing_metadata = [col for col in screen_sample_cols if col not in metadata_map]
    if missing_metadata:
        raise ValueError(
            "Some screen RPM columns do not have pDNA-batch metadata. "
            f"First few: {missing_metadata[:10]}"
        )

    screen_to_batch = {col: metadata_map[col] for col in screen_sample_cols}

    missing_baselines = sorted(
        {batch for batch in screen_to_batch.values() if batch not in available_pdna_batches}
    )
    if missing_baselines:
        raise ValueError(f"No pDNA baseline was built for batch(es): {missing_baselines}")

    print("\nScreen replicate to pDNA batch mapping counts:")
    for batch, count in pd.Series(screen_to_batch).value_counts().sort_index().items():
        print(f"  {batch}: {count} screen replicate column(s)")

    return screen_to_batch


# --------------------------------------------------------------------------- #
# Compute log2 fold-change
# --------------------------------------------------------------------------- #
def compute_log2fc(
    pdna_baselines: pd.DataFrame,
    screen_sample_cols: list[str],
    screen_to_batch: dict[str, str],
) -> None:
    """Compute log2FC in chunks and save the wide sgRNA x screen matrix."""
    temp_output_path = OUTPUT_PATH.with_suffix(".csv.tmp")

    screen_cols_by_batch = {
        batch: [col for col in screen_sample_cols if screen_to_batch[col] == batch]
        for batch in sorted(set(screen_to_batch.values()))
    }

    screen_rpm_sums = pd.Series(0.0, index=screen_sample_cols)
    summary = {
        "total_values": 0,
        "missing_values": 0,
        "infinite_values": 0,
        "finite_values": 0,
        "finite_sum": 0.0,
        "min": np.inf,
        "max": -np.inf,
        "median_sample": [],
    }

    print(f"\nComputing log2FC in chunks of {CHUNK_SIZE:,} sgRNAs...")

    first_chunk = True
    rows_written = 0

    for chunk in pd.read_csv(SCREEN_RPM_PATH, chunksize=CHUNK_SIZE):
        require_columns(chunk, [GUIDE_COL], SCREEN_RPM_PATH.name)

        chunk = chunk.set_index(GUIDE_COL)
        screen_values = chunk[screen_sample_cols].astype(float)
        screen_rpm_sums = screen_rpm_sums.add(screen_values.sum(axis=0), fill_value=0.0)

        missing_guides = screen_values.index.difference(pdna_baselines.index)
        if len(missing_guides) > 0:
            raise ValueError(
                "Some screen sgRNAs are missing from the pDNA baseline table. "
                f"First few: {missing_guides[:10].tolist()}"
            )

        log2fc = pd.DataFrame(index=screen_values.index, columns=screen_sample_cols, dtype=float)

        for batch, batch_screen_cols in screen_cols_by_batch.items():
            screen_matrix = screen_values[batch_screen_cols].to_numpy(float)
            pdna_baseline = pdna_baselines.loc[screen_values.index, batch].to_numpy(float)
            pdna_matrix = pdna_baseline[:, None]

            batch_log2fc = np.log2((screen_matrix + PSEUDOCOUNT) / (pdna_matrix + PSEUDOCOUNT))
            log2fc.loc[:, batch_screen_cols] = batch_log2fc
            update_summary(summary, batch_log2fc)

        log2fc.insert(0, GUIDE_COL, log2fc.index)
        log2fc.to_csv(
            temp_output_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )

        first_chunk = False
        rows_written += len(log2fc)
        print(f"  processed {rows_written:,} sgRNAs")

    temp_output_path.replace(OUTPUT_PATH)

    print()
    warn_if_rpm_sums_look_wrong("Screen", screen_rpm_sums)
    print_log2fc_summary(summary)

    print(f"\nSaved log2FC file to: {OUTPUT_PATH}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=== Phase 3 / Step III.e: log2 fold-change vs pDNA ===")
    print(f"Working folder: {PHASE3_DIR}")
    print(f"Pseudocount: {PSEUDOCOUNT}\n")

    pdna_rpm, screen_metadata, pdna_metadata, screen_sample_cols = load_inputs()
    pdna_baselines = build_pdna_baselines(pdna_rpm, pdna_metadata)
    screen_to_batch = build_screen_to_pdna_batch_map(
        screen_sample_cols=screen_sample_cols,
        screen_metadata=screen_metadata,
        available_pdna_batches=set(pdna_baselines.columns),
    )
    compute_log2fc(pdna_baselines, screen_sample_cols, screen_to_batch)

    print("\nStep III.e complete.")


if __name__ == "__main__":
    main()
