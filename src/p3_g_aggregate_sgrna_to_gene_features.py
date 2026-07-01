"""
Phase 3, Step III.g - Aggregate sgRNAs to gene-level features.

This script converts sgRNA-level log2 fold-change values into gene-level
features per screen replicate. It keeps replicate-level information separate;
cell-line-level aggregation happens in a later step.

Input:
    data/processed/phase3_inputs/screen_log2fc_vs_pdna_gene_mapped.csv

Outputs:
    data/processed/phase3_inputs/gene_level_log2fc_features_by_replicate.csv
    data/processed/phase3_inputs/gene_level_aggregation_qc.csv

Final output format:
    gene
    replicate_id
    mean_log2fc
    median_log2fc
    min_log2fc
    max_log2fc
    std_log2fc
    iqr_log2fc
    n_guides
    frac_guides_negative
    frac_guides_strongly_negative

Run only after reviewing:
    python src/p3_g_aggregate_sgrna_to_gene_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path("/Volumes/nimas_usb/project_data_repo/crispr_ai_ml")
DATA_DIR = PROJECT_DIR / "data"
PHASE3_DIR = DATA_DIR / "processed" / "phase3_inputs"

INPUT_PATH = PHASE3_DIR / "screen_log2fc_vs_pdna_gene_mapped.csv"
OUTPUT_PATH = PHASE3_DIR / "gene_level_log2fc_features_by_replicate.csv"
QC_OUTPUT_PATH = PHASE3_DIR / "gene_level_aggregation_qc.csv"


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
SGRNA_COL = "sgRNA_seq"
GENE_COL = "gene"

METADATA_COLS = {
    SGRNA_COL,
    GENE_COL,
    "Gene",
    "gene_symbol",
    "entrez_id",
    "Entrez ID",
    "entrez_gene_id",
}

STRONGLY_NEGATIVE_THRESHOLD = -1.0
# log2FC < -1 means at least about 2-fold depletion.
REPLICATE_NUMERIC_FRACTION_MIN = 0.95
PROGRESS_EVERY_N_REPLICATES = 25


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def check_required_file(path: Path) -> None:
    """Fail early if the required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")


def require_columns(df: pd.DataFrame, columns: list[str], file_name: str) -> None:
    """Fail if a dataframe does not contain required columns."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing required column(s): {missing}")


def print_column_preview(columns: list[str], label: str, n: int = 10) -> None:
    """Print a compact column preview."""
    print(f"{label} first {min(n, len(columns))} column(s): {columns[:n]}")


def summarize_series(series: pd.Series) -> dict[str, float]:
    """Return compact numeric summary statistics for QC output."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }

    return {
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def count_valid_gene_log2fc_rows(df: pd.DataFrame, replicate_col: str) -> int:
    """Count rows with both a gene and a numeric log2FC value."""
    values = pd.to_numeric(df[replicate_col], errors="coerce")
    return int((df[GENE_COL].notna() & values.notna()).sum())


# --------------------------------------------------------------------------- #
# Load mapped sgRNA logFC
# --------------------------------------------------------------------------- #
def load_gene_mapped_logfc() -> pd.DataFrame:
    """
    Load the sgRNA-level gene-mapped log2FC matrix and run initial checks.

    Expected input structure:
        sgRNA_seq | gene | replicate_1 | replicate_2 | ...
    """
    check_required_file(INPUT_PATH)

    print("Loading gene-mapped sgRNA log2FC table...")
    df = pd.read_csv(INPUT_PATH)
    require_columns(df, [SGRNA_COL, GENE_COL], INPUT_PATH.name)

    print(f"input shape: {df.shape}")
    print_column_preview(df.columns.tolist(), "input")
    print(f"missing gene count: {df[GENE_COL].isna().sum()}")
    print(f"unique sgRNAs: {df[SGRNA_COL].nunique(dropna=True)}")
    print(f"unique genes: {df[GENE_COL].nunique(dropna=True)}")

    return df


# --------------------------------------------------------------------------- #
# Identify replicate columns
# --------------------------------------------------------------------------- #
def identify_replicate_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify screen replicate log2FC columns.

    First exclude known metadata columns, then keep only candidate columns where
    nearly all non-missing values can be converted to numeric log2FC values.
    This prevents future metadata columns from being accidentally aggregated.
    """
    candidate_cols = [col for col in df.columns if col not in METADATA_COLS]
    replicate_cols = []
    rejected_cols = []

    for col in candidate_cols:
        non_missing = df[col].notna()
        non_missing_count = int(non_missing.sum())

        if non_missing_count == 0:
            rejected_cols.append((col, 0.0))
            continue

        numeric_values = pd.to_numeric(df.loc[non_missing, col], errors="coerce")
        numeric_fraction = float(numeric_values.notna().mean())

        if numeric_fraction >= REPLICATE_NUMERIC_FRACTION_MIN:
            replicate_cols.append(col)
        else:
            rejected_cols.append((col, numeric_fraction))

    if not replicate_cols:
        raise ValueError("No replicate log2FC columns were identified.")

    print("\nReplicate column checks:")
    print(f"candidate non-metadata columns: {len(candidate_cols)}")
    print(f"number of replicate columns: {len(replicate_cols)}")
    print(f"first 5 replicate columns: {replicate_cols[:5]}")
    print(f"last 5 replicate columns: {replicate_cols[-5:]}")
    print(f"rejected non-numeric metadata-like columns: {len(rejected_cols)}")
    if rejected_cols:
        print(f"first rejected columns: {rejected_cols[:5]}")

    if SGRNA_COL in replicate_cols or GENE_COL in replicate_cols:
        raise ValueError("sgRNA or gene column was incorrectly included as a replicate column.")

    return replicate_cols


# --------------------------------------------------------------------------- #
# Aggregate sgRNAs to genes
# --------------------------------------------------------------------------- #
def aggregate_one_replicate(df: pd.DataFrame, replicate_col: str) -> pd.DataFrame:
    """
    Aggregate one replicate column from sgRNA-level values to gene-level values.

    Missing log2FC values are dropped before aggregation. n_guides records how
    many valid sgRNAs contributed to each gene's features in this replicate.
    """
    temp = df[[GENE_COL, replicate_col]].copy()
    temp = temp.rename(columns={replicate_col: "log2fc"})
    temp["log2fc"] = pd.to_numeric(temp["log2fc"], errors="coerce")
    temp = temp.dropna(subset=[GENE_COL, "log2fc"])

    temp["guide_negative"] = temp["log2fc"] < 0
    temp["guide_strongly_negative"] = temp["log2fc"] < STRONGLY_NEGATIVE_THRESHOLD

    grouped = temp.groupby(GENE_COL, dropna=True)
    aggregated = grouped.agg(
        mean_log2fc=("log2fc", "mean"),
        median_log2fc=("log2fc", "median"),
        min_log2fc=("log2fc", "min"),
        max_log2fc=("log2fc", "max"),
        std_log2fc=("log2fc", "std"),
        n_guides=("log2fc", "count"),
        frac_guides_negative=("guide_negative", "mean"),
        frac_guides_strongly_negative=("guide_strongly_negative", "mean"),
    ).reset_index()
    q75 = grouped["log2fc"].quantile(0.75)
    q25 = grouped["log2fc"].quantile(0.25)
    iqr = (q75 - q25).rename("iqr_log2fc").reset_index()
    aggregated = aggregated.merge(iqr, on=GENE_COL, how="left")

    ordered_cols = [
        GENE_COL,
        "mean_log2fc",
        "median_log2fc",
        "min_log2fc",
        "max_log2fc",
        "std_log2fc",
        "iqr_log2fc",
        "n_guides",
        "frac_guides_negative",
        "frac_guides_strongly_negative",
    ]
    aggregated = aggregated[ordered_cols]

    aggregated.insert(1, "replicate_id", replicate_col)
    return aggregated


def aggregate_replicates_one_by_one(
    df: pd.DataFrame,
    replicate_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Process one replicate column at a time and append results to the output CSV.

    Returns a compact sample dataframe for QC summaries plus counters collected
    during the aggregation.
    """
    temp_output_path = OUTPUT_PATH.with_suffix(".csv.tmp")
    if temp_output_path.exists():
        temp_output_path.unlink()

    rows_before_missing_drop_total = 0
    rows_after_missing_drop_total = 0
    gene_replicate_rows = 0
    rows_with_less_than_3_guides = 0
    first_chunk = True
    qc_samples = []

    print("\nAggregating sgRNAs to gene-level features one replicate at a time...")

    for i, replicate_col in enumerate(replicate_cols, start=1):
        rows_before = len(df)
        valid_rows = count_valid_gene_log2fc_rows(df, replicate_col)

        rows_before_missing_drop_total += rows_before
        rows_after_missing_drop_total += valid_rows

        aggregated = aggregate_one_replicate(df, replicate_col)
        gene_replicate_rows += len(aggregated)
        rows_with_less_than_3_guides += int((aggregated["n_guides"] < 3).sum())

        aggregated.to_csv(
            temp_output_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False

        if i <= 5 or i % PROGRESS_EVERY_N_REPLICATES == 0 or i == len(replicate_cols):
            removed = rows_before - valid_rows
            print(
                f"  replicate {i:,}/{len(replicate_cols):,}: {replicate_col} "
                f"({removed:,} missing log2FC row(s) removed)"
            )

        # Keep a small in-memory QC sample from early and periodic replicates.
        if i <= 5 or i % 100 == 0 or i == len(replicate_cols):
            qc_samples.append(aggregated)

    temp_output_path.replace(OUTPUT_PATH)

    qc_sample_df = pd.concat(qc_samples, ignore_index=True) if qc_samples else pd.DataFrame()
    counters = {
        "rows_before_missing_drop_total": rows_before_missing_drop_total,
        "rows_after_missing_drop_total": rows_after_missing_drop_total,
        "missing_log2fc_rows_removed_total": (
            rows_before_missing_drop_total - rows_after_missing_drop_total
        ),
        "gene_replicate_rows": gene_replicate_rows,
        "rows_with_less_than_3_guides": rows_with_less_than_3_guides,
    }
    return qc_sample_df, counters


# --------------------------------------------------------------------------- #
# QC and output summaries
# --------------------------------------------------------------------------- #
def build_qc_metrics(
    df: pd.DataFrame,
    replicate_cols: list[str],
    qc_sample_df: pd.DataFrame,
    counters: dict[str, float],
) -> pd.DataFrame:
    """Build a compact QC dataframe for console output and CSV saving."""
    guide_counts_per_gene = df.dropna(subset=[GENE_COL]).groupby(GENE_COL)[SGRNA_COL].nunique()

    metrics = [
        ("input_rows", len(df)),
        ("unique_sgrnas", df[SGRNA_COL].nunique(dropna=True)),
        ("unique_genes", df[GENE_COL].nunique(dropna=True)),
        ("missing_gene_rows", int(df[GENE_COL].isna().sum())),
        ("replicate_columns", len(replicate_cols)),
        ("gene_replicate_rows", counters["gene_replicate_rows"]),
        ("median_guides_per_gene", float(guide_counts_per_gene.median())),
        ("min_guides_per_gene", int(guide_counts_per_gene.min())),
        ("max_guides_per_gene", int(guide_counts_per_gene.max())),
        ("rows_with_less_than_3_guides", counters["rows_with_less_than_3_guides"]),
        ("rows_before_missing_drop_total", counters["rows_before_missing_drop_total"]),
        ("rows_after_missing_drop_total", counters["rows_after_missing_drop_total"]),
        ("missing_log2fc_rows_removed_total", counters["missing_log2fc_rows_removed_total"]),
    ]

    if not qc_sample_df.empty:
        for feature in ["mean_log2fc", "median_log2fc", "std_log2fc", "iqr_log2fc", "n_guides"]:
            summary = summarize_series(qc_sample_df[feature])
            for stat_name, value in summary.items():
                metrics.append((f"sample_{feature}_{stat_name}", value))

    return pd.DataFrame(metrics, columns=["metric", "value"])


def print_qc_summary(qc_df: pd.DataFrame) -> None:
    """Print the core QC metrics in a readable format."""
    metrics = dict(zip(qc_df["metric"], qc_df["value"]))

    print("\nAggregation QC:")
    print(f"  input sgRNA rows                 : {metrics['input_rows']}")
    print(f"  unique genes                     : {metrics['unique_genes']}")
    print(f"  replicate columns                : {metrics['replicate_columns']}")
    print(f"  gene-replicate rows              : {metrics['gene_replicate_rows']}")
    print(f"  median guides per gene           : {metrics['median_guides_per_gene']}")
    print(f"  min guides per gene              : {metrics['min_guides_per_gene']}")
    print(f"  max guides per gene              : {metrics['max_guides_per_gene']}")
    print(f"  rows with n_guides < 3           : {metrics['rows_with_less_than_3_guides']}")
    print(f"  missing log2FC rows removed      : {metrics['missing_log2fc_rows_removed_total']}")

    sample_keys = [
        "sample_mean_log2fc_median",
        "sample_median_log2fc_median",
        "sample_std_log2fc_median",
        "sample_iqr_log2fc_median",
        "sample_n_guides_median",
    ]
    if all(key in metrics for key in sample_keys):
        print("\nSample feature distributions:")
        print(f"  median of mean_log2fc             : {metrics['sample_mean_log2fc_median']}")
        print(f"  median of median_log2fc           : {metrics['sample_median_log2fc_median']}")
        print(f"  median of std_log2fc              : {metrics['sample_std_log2fc_median']}")
        print(f"  median of iqr_log2fc              : {metrics['sample_iqr_log2fc_median']}")
        print(f"  median n_guides                   : {metrics['sample_n_guides_median']}")


def save_qc_output(qc_df: pd.DataFrame) -> None:
    """Save the aggregation QC metrics."""
    qc_df.to_csv(QC_OUTPUT_PATH, index=False)
    print(f"\nSaved QC output to: {QC_OUTPUT_PATH}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=== Phase 3 / Step III.g: Aggregate sgRNAs to gene-level features ===")
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Phase 3 folder: {PHASE3_DIR}\n")

    df = load_gene_mapped_logfc()
    replicate_cols = identify_replicate_columns(df)
    qc_sample_df, counters = aggregate_replicates_one_by_one(df, replicate_cols)
    qc_df = build_qc_metrics(df, replicate_cols, qc_sample_df, counters)

    print_qc_summary(qc_df)
    save_qc_output(qc_df)

    print(f"Saved gene-level feature output to: {OUTPUT_PATH}")
    print("\nStep III.g complete.")


if __name__ == "__main__":
    main()
