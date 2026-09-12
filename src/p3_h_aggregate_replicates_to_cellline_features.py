"""
Phase 3, Step III.h - Aggregate replicates to cell-line features.

This script converts gene-level replicate features into gene-level cell-line
features. It preserves cell-line-specific CRISPR effects and does not aggregate
the same gene across different cell lines.

Inputs:
    data/processed/phase3_inputs/gene_level_log2fc_features_by_replicate.csv
    data/processed/phase3_inputs/screen_metadata_pass.csv

Outputs:
    data/processed/phase3_inputs/gene_level_log2fc_features_by_cellline.csv
    data/processed/phase3_inputs/cellline_level_aggregation_qc.csv

Final output format:
    gene
    cell_line_id
    mean_log2fc_cellline
    median_log2fc_cellline
    min_log2fc_cellline
    max_log2fc_cellline
    std_log2fc_across_replicates
    replicate_count
    median_n_guides
    min_n_guides
    mean_frac_guides_negative
    mean_frac_guides_strongly_negative

Run only after reviewing:
    python src/p3_h_aggregate_replicates_to_cellline_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_paths import DATA_ROOT
except ImportError:  # Support running this file directly.
    from data_paths import DATA_ROOT


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PHASE3_DIR = DATA_ROOT / "processed" / "phase3_inputs"

GENE_REPLICATE_FEATURES_PATH = PHASE3_DIR / "gene_level_log2fc_features_by_replicate.csv"
SCREEN_METADATA_PATH = PHASE3_DIR / "screen_metadata_pass.csv"
OUTPUT_PATH = PHASE3_DIR / "gene_level_log2fc_features_by_cellline.csv"
QC_OUTPUT_PATH = PHASE3_DIR / "cellline_level_aggregation_qc.csv"


# --------------------------------------------------------------------------- #
# Column names
# --------------------------------------------------------------------------- #
GENE_COL = "gene"
REPLICATE_COL = "replicate_id"

# These names match the local screen_metadata_pass.csv file.
METADATA_REPLICATE_COL = "SequenceID"
CELL_LINE_ID_COL = "ModelID"
CELL_LINE_NAME_CANDIDATES = ["CellLineName", "CCLEName", "stripped_cell_line_name"]

STD_CELL_LINE_ID_COL = "cell_line_id"
STD_CELL_LINE_NAME_COL = "cell_line_name"

REQUIRED_FEATURE_COLS = [
    GENE_COL,
    REPLICATE_COL,
    "mean_log2fc",
    "median_log2fc",
    "min_log2fc",
    "max_log2fc",
    "n_guides",
    "frac_guides_negative",
    "frac_guides_strongly_negative",
]

NUMERIC_FEATURE_COLS = [
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

MIN_REPLICATE_MAPPING_RATE = 0.99


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def check_required_files(paths: list[Path]) -> None:
    """Fail early if any required input file is missing."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required input file(s):\n{missing_text}")


def require_columns(df: pd.DataFrame, columns: list[str], file_name: str) -> None:
    """Fail if a dataframe does not contain required columns."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing required column(s): {missing}")


def normalize_ids(values: pd.Series) -> pd.Series:
    """Normalize replicate/model IDs for exact joins."""
    return values.astype("string").str.strip()


def print_column_preview(label: str, columns: list[str], n: int = 10) -> None:
    """Print a compact column preview."""
    print(f"{label} first {min(n, len(columns))} column(s): {columns[:n]}")


def summarize_series(series: pd.Series) -> dict[str, float]:
    """Return compact numeric distribution summaries."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"min": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "max": np.nan}

    return {
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def first_non_null(values: pd.Series):
    """Return the first non-null value in a group, or NA if none exists."""
    clean = values.dropna()
    if clean.empty:
        return pd.NA
    return clean.iloc[0]


# --------------------------------------------------------------------------- #
# Load and inspect inputs
# --------------------------------------------------------------------------- #
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load gene-level replicate features and screen metadata."""
    check_required_files([GENE_REPLICATE_FEATURES_PATH, SCREEN_METADATA_PATH])

    print("Loading input files...")
    gene_features = pd.read_csv(GENE_REPLICATE_FEATURES_PATH)
    screen_metadata = pd.read_csv(SCREEN_METADATA_PATH)

    require_columns(gene_features, REQUIRED_FEATURE_COLS, GENE_REPLICATE_FEATURES_PATH.name)
    require_columns(screen_metadata, [METADATA_REPLICATE_COL, CELL_LINE_ID_COL], SCREEN_METADATA_PATH.name)

    print(f"Gene-level replicate features shape: {gene_features.shape}")
    print_column_preview("Gene-level replicate features", gene_features.columns.tolist())
    print(f"Screen metadata shape: {screen_metadata.shape}")
    print(f"Screen metadata columns: {screen_metadata.columns.tolist()}")
    print("First few replicate IDs from gene features:")
    print(gene_features[REPLICATE_COL].dropna().head().to_list())
    print("First few replicate IDs from metadata:")
    print(screen_metadata[METADATA_REPLICATE_COL].dropna().head().to_list())
    print("First few cell-line IDs from metadata:")
    print(screen_metadata[CELL_LINE_ID_COL].dropna().head().to_list())

    return gene_features, screen_metadata


def inspect_metadata_columns(screen_metadata: pd.DataFrame) -> None:
    """Print candidate replicate and cell-line metadata columns."""
    replicate_candidates = [
        "SequenceID",
        "sequence_id",
        "replicate_id",
        "ScreenID",
        "ScreenSequenceID",
    ]
    cell_line_candidates = [
        "DepMap_ID",
        "ModelID",
        "cell_line",
        "CellLineName",
        "CCLEName",
        "stripped_cell_line_name",
    ]

    print("\nMetadata column inspection:")
    print(f"All metadata columns: {screen_metadata.columns.tolist()}")

    for col in replicate_candidates:
        if col in screen_metadata.columns:
            print(f"  replicate candidate {col}:")
            print(f"    unique values: {screen_metadata[col].nunique(dropna=True)}")
            print(f"    first values : {screen_metadata[col].dropna().head().to_list()}")

    for col in cell_line_candidates:
        if col in screen_metadata.columns:
            print(f"  cell-line candidate {col}:")
            print(f"    unique values: {screen_metadata[col].nunique(dropna=True)}")
            print(f"    first values : {screen_metadata[col].dropna().head().to_list()}")


# --------------------------------------------------------------------------- #
# Metadata lookup and overlap checks
# --------------------------------------------------------------------------- #
def choose_cell_line_name_col(screen_metadata: pd.DataFrame) -> str | None:
    """Return the first available optional cell-line name column, if present."""
    for col in CELL_LINE_NAME_CANDIDATES:
        if col in screen_metadata.columns:
            return col
    return None


def prepare_metadata_lookup(screen_metadata: pd.DataFrame) -> pd.DataFrame:
    """Create a clean replicate_id to cell_line_id lookup table."""
    cell_line_name_col = choose_cell_line_name_col(screen_metadata)
    selected_cols = [METADATA_REPLICATE_COL, CELL_LINE_ID_COL]
    if cell_line_name_col is not None:
        selected_cols.append(cell_line_name_col)

    lookup = screen_metadata[selected_cols].copy()
    rename_map = {
        METADATA_REPLICATE_COL: REPLICATE_COL,
        CELL_LINE_ID_COL: STD_CELL_LINE_ID_COL,
    }
    if cell_line_name_col is not None:
        rename_map[cell_line_name_col] = STD_CELL_LINE_NAME_COL

    lookup = lookup.rename(columns=rename_map)
    lookup[REPLICATE_COL] = normalize_ids(lookup[REPLICATE_COL])
    lookup[STD_CELL_LINE_ID_COL] = normalize_ids(lookup[STD_CELL_LINE_ID_COL])
    if STD_CELL_LINE_NAME_COL in lookup.columns:
        lookup[STD_CELL_LINE_NAME_COL] = lookup[STD_CELL_LINE_NAME_COL].astype("string").str.strip()

    before_drop = len(lookup)
    lookup = lookup.dropna(subset=[REPLICATE_COL, STD_CELL_LINE_ID_COL])
    lookup = lookup[(lookup[REPLICATE_COL] != "") & (lookup[STD_CELL_LINE_ID_COL] != "")]
    removed_missing = before_drop - len(lookup)

    before_dedup = len(lookup)
    lookup = lookup.drop_duplicates()
    removed_duplicates = before_dedup - len(lookup)

    conflicts = lookup.groupby(REPLICATE_COL)[STD_CELL_LINE_ID_COL].nunique(dropna=True)
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        examples = conflicts.head(10).index.to_list()
        raise ValueError(
            "Some replicate IDs map to more than one cell_line_id. "
            f"Inspect metadata before merging. First examples: {examples}"
        )

    # If duplicate replicate rows remain only because optional names differ, keep one.
    before_replicate_dedup = len(lookup)
    lookup = lookup.drop_duplicates(subset=[REPLICATE_COL])
    removed_replicate_duplicates = before_replicate_dedup - len(lookup)

    print("\nPrepared metadata lookup:")
    print(f"  rows before missing-ID filtering      : {before_drop}")
    print(f"  rows removed for missing IDs          : {removed_missing}")
    print(f"  exact duplicate rows removed          : {removed_duplicates}")
    print(f"  duplicate replicate rows collapsed    : {removed_replicate_duplicates}")
    print(f"  final lookup rows                     : {len(lookup)}")
    print(f"  unique replicate IDs                  : {lookup[REPLICATE_COL].nunique()}")
    print(f"  unique cell lines                     : {lookup[STD_CELL_LINE_ID_COL].nunique()}")

    return lookup


def check_replicate_overlap(
    gene_features: pd.DataFrame,
    metadata_lookup: pd.DataFrame,
) -> dict[str, float]:
    """Compare replicate IDs between the gene feature table and metadata."""
    feature_replicates = set(normalize_ids(gene_features[REPLICATE_COL]).dropna())
    metadata_replicates = set(metadata_lookup[REPLICATE_COL].dropna())

    overlap = feature_replicates & metadata_replicates
    missing_metadata = feature_replicates - metadata_replicates
    metadata_only = metadata_replicates - feature_replicates
    mapping_rate = len(overlap) / len(feature_replicates) if feature_replicates else 0.0

    print("\nReplicate overlap QC:")
    print(f"  replicate IDs in gene features        : {len(feature_replicates)}")
    print(f"  replicate IDs in metadata             : {len(metadata_replicates)}")
    print(f"  overlapping replicate IDs             : {len(overlap)}")
    print(f"  feature replicate IDs missing metadata: {len(missing_metadata)}")
    print(f"  metadata replicate IDs not in features: {len(metadata_only)}")
    print(f"  replicate mapping rate                : {mapping_rate:.4%}")

    if missing_metadata:
        print(f"  first missing metadata replicate IDs  : {sorted(missing_metadata)[:10]}")
    if metadata_only:
        print(f"  first metadata-only replicate IDs     : {sorted(metadata_only)[:10]}")

    if mapping_rate < MIN_REPLICATE_MAPPING_RATE:
        raise ValueError(
            f"Replicate mapping rate is below {MIN_REPLICATE_MAPPING_RATE:.0%}. "
            "The metadata replicate ID column is likely wrong or incomplete."
        )

    return {
        "feature_replicates": len(feature_replicates),
        "metadata_replicates": len(metadata_replicates),
        "overlapping_replicates": len(overlap),
        "missing_metadata_replicates": len(missing_metadata),
        "metadata_only_replicates": len(metadata_only),
        "replicate_mapping_rate": mapping_rate,
    }


# --------------------------------------------------------------------------- #
# Merge and aggregate
# --------------------------------------------------------------------------- #
def merge_gene_features_with_metadata(
    gene_features: pd.DataFrame,
    metadata_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach cell-line information to every gene-replicate row."""
    gene_features = gene_features.copy()
    gene_features[REPLICATE_COL] = normalize_ids(gene_features[REPLICATE_COL])

    rows_before = len(gene_features)
    merged = gene_features.merge(metadata_lookup, how="left", on=REPLICATE_COL)
    rows_after = len(merged)

    if rows_after != rows_before:
        raise ValueError(
            "Rows changed after metadata merge. "
            f"Before: {rows_before}; after: {rows_after}. "
            "This usually means metadata lookup has duplicate replicate IDs."
        )

    missing_cell_line_rows = int(merged[STD_CELL_LINE_ID_COL].isna().sum())
    missing_cell_line_pct = missing_cell_line_rows / rows_after * 100.0 if rows_after else 0.0

    print("\nMetadata merge QC:")
    print(f"  rows before merge               : {rows_before}")
    print(f"  rows after merge                : {rows_after}")
    print(f"  missing cell_line_id rows       : {missing_cell_line_rows}")
    print(f"  missing cell_line_id percentage : {missing_cell_line_pct:.4f}%")

    return merged


def aggregate_to_cellline_features(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Aggregate gene-replicate rows to gene-cell-line rows."""
    required_id_cols = [GENE_COL, REPLICATE_COL, STD_CELL_LINE_ID_COL]
    before_drop = len(merged)
    clean = merged.dropna(subset=required_id_cols).copy()
    removed_required_missing = before_drop - len(clean)

    for col in NUMERIC_FEATURE_COLS:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    replicate_counts_per_cellline = (
        clean.groupby(STD_CELL_LINE_ID_COL)[REPLICATE_COL]
        .nunique()
        .sort_values()
    )
    print("\nReplicate counts per cell line before gene aggregation:")
    print(f"  cell lines                         : {len(replicate_counts_per_cellline)}")
    print(f"  min replicate IDs per cell line    : {replicate_counts_per_cellline.min()}")
    print(f"  median replicate IDs per cell line : {replicate_counts_per_cellline.median()}")
    print(f"  max replicate IDs per cell line    : {replicate_counts_per_cellline.max()}")
    print("  first few lowest-replicate cell lines:")
    print(replicate_counts_per_cellline.head(10).to_dict())
    if replicate_counts_per_cellline.max() == 1:
        print("  WARNING: all cell lines have only one replicate ID; confirm ModelID is the right key.")

    group_cols = [GENE_COL, STD_CELL_LINE_ID_COL]

    aggregation_spec = {
        "mean_log2fc_cellline": ("mean_log2fc", "mean"),
        "median_log2fc_cellline": ("median_log2fc", "median"),
        "min_log2fc_cellline": ("min_log2fc", "min"),
        "max_log2fc_cellline": ("max_log2fc", "max"),
        "std_log2fc_across_replicates": ("median_log2fc", "std"),
        "replicate_count": (REPLICATE_COL, "nunique"),
        "median_n_guides": ("n_guides", "median"),
        "min_n_guides": ("n_guides", "min"),
        "mean_frac_guides_negative": ("frac_guides_negative", "mean"),
        "mean_frac_guides_strongly_negative": ("frac_guides_strongly_negative", "mean"),
    }
    if STD_CELL_LINE_NAME_COL in clean.columns:
        aggregation_spec[STD_CELL_LINE_NAME_COL] = (STD_CELL_LINE_NAME_COL, first_non_null)

    print("\nAggregating replicate-level gene features to cell-line level...")
    grouped = clean.groupby(group_cols, dropna=True)
    output = grouped.agg(**aggregation_spec).reset_index()

    ordered_cols = [
        GENE_COL,
        STD_CELL_LINE_ID_COL,
    ]
    if STD_CELL_LINE_NAME_COL in output.columns:
        ordered_cols.append(STD_CELL_LINE_NAME_COL)
    ordered_cols.extend(
        [
            "mean_log2fc_cellline",
            "median_log2fc_cellline",
            "min_log2fc_cellline",
            "max_log2fc_cellline",
            "std_log2fc_across_replicates",
            "replicate_count",
            "median_n_guides",
            "min_n_guides",
            "mean_frac_guides_negative",
            "mean_frac_guides_strongly_negative",
        ]
    )
    output = output[ordered_cols]

    counters = {
        "input_rows_before_required_drop": before_drop,
        "rows_removed_missing_required_ids": removed_required_missing,
        "input_rows_after_required_drop": len(clean),
        "min_replicates_per_cellline_before_gene_aggregation": int(replicate_counts_per_cellline.min()),
        "median_replicates_per_cellline_before_gene_aggregation": float(
            replicate_counts_per_cellline.median()
        ),
        "max_replicates_per_cellline_before_gene_aggregation": int(replicate_counts_per_cellline.max()),
    }
    return output, counters


# --------------------------------------------------------------------------- #
# QC and save outputs
# --------------------------------------------------------------------------- #
def build_qc_metrics(
    merged: pd.DataFrame,
    cellline_features: pd.DataFrame,
    overlap_qc: dict[str, float],
    aggregation_counters: dict[str, int],
) -> pd.DataFrame:
    """Build a compact QC summary table."""
    replicate_counts = cellline_features["replicate_count"]
    genes_per_cellline = cellline_features.groupby(STD_CELL_LINE_ID_COL)[GENE_COL].nunique()

    print("\nGenes per cell line after aggregation:")
    print(f"  cell lines                 : {len(genes_per_cellline)}")
    print(f"  min genes per cell line    : {genes_per_cellline.min()}")
    print(f"  median genes per cell line : {genes_per_cellline.median()}")
    print(f"  max genes per cell line    : {genes_per_cellline.max()}")
    print("  first few lowest-gene cell lines:")
    print(genes_per_cellline.sort_values().head(10).to_dict())

    metrics = [
        ("input_gene_replicate_rows", len(merged)),
        ("unique_genes", merged[GENE_COL].nunique(dropna=True)),
        ("unique_replicates", merged[REPLICATE_COL].nunique(dropna=True)),
        ("unique_cell_lines", merged[STD_CELL_LINE_ID_COL].nunique(dropna=True)),
        ("output_gene_cellline_rows", len(cellline_features)),
        ("replicate_mapping_rate", overlap_qc["replicate_mapping_rate"]),
        ("missing_cell_line_rows_after_merge", int(merged[STD_CELL_LINE_ID_COL].isna().sum())),
        ("rows_removed_missing_required_ids", aggregation_counters["rows_removed_missing_required_ids"]),
        ("median_replicate_count", float(replicate_counts.median())),
        ("min_replicate_count", int(replicate_counts.min())),
        ("max_replicate_count", int(replicate_counts.max())),
        ("rows_with_replicate_count_1", int((replicate_counts == 1).sum())),
        ("rows_with_replicate_count_ge_2", int((replicate_counts >= 2).sum())),
        (
            "min_replicates_per_cellline_before_gene_aggregation",
            aggregation_counters["min_replicates_per_cellline_before_gene_aggregation"],
        ),
        (
            "median_replicates_per_cellline_before_gene_aggregation",
            aggregation_counters["median_replicates_per_cellline_before_gene_aggregation"],
        ),
        (
            "max_replicates_per_cellline_before_gene_aggregation",
            aggregation_counters["max_replicates_per_cellline_before_gene_aggregation"],
        ),
        ("min_genes_per_cellline", int(genes_per_cellline.min())),
        ("median_genes_per_cellline", float(genes_per_cellline.median())),
        ("max_genes_per_cellline", int(genes_per_cellline.max())),
    ]

    for col in [
        "median_log2fc_cellline",
        "mean_log2fc_cellline",
        "std_log2fc_across_replicates",
        "median_n_guides",
        "replicate_count",
    ]:
        summary = summarize_series(cellline_features[col])
        for stat_name, value in summary.items():
            metrics.append((f"{col}_{stat_name}", value))

    return pd.DataFrame(metrics, columns=["metric", "value"])


def print_qc_summary(qc_df: pd.DataFrame) -> None:
    """Print key QC metrics for quick review."""
    metrics = dict(zip(qc_df["metric"], qc_df["value"]))

    print("\nCell-line aggregation QC:")
    print(f"  input gene-replicate rows        : {metrics['input_gene_replicate_rows']}")
    print(f"  unique genes                     : {metrics['unique_genes']}")
    print(f"  unique replicate IDs             : {metrics['unique_replicates']}")
    print(f"  unique cell lines                : {metrics['unique_cell_lines']}")
    print(f"  output gene-cellline rows        : {metrics['output_gene_cellline_rows']}")
    print(f"  replicate mapping rate           : {metrics['replicate_mapping_rate']:.4%}")
    print(f"  missing cell_line_id rows        : {metrics['missing_cell_line_rows_after_merge']}")
    print(f"  rows removed before aggregation  : {metrics['rows_removed_missing_required_ids']}")
    print(f"  median replicate count           : {metrics['median_replicate_count']}")
    print(f"  min replicate count              : {metrics['min_replicate_count']}")
    print(f"  max replicate count              : {metrics['max_replicate_count']}")
    print(f"  rows with replicate_count = 1    : {metrics['rows_with_replicate_count_1']}")
    print(f"  rows with replicate_count >= 2   : {metrics['rows_with_replicate_count_ge_2']}")
    print(
        "  median replicate IDs per cell line before gene aggregation: "
        f"{metrics['median_replicates_per_cellline_before_gene_aggregation']}"
    )
    print(f"  min genes per cell line          : {metrics['min_genes_per_cellline']}")
    print(f"  median genes per cell line       : {metrics['median_genes_per_cellline']}")
    print(f"  max genes per cell line          : {metrics['max_genes_per_cellline']}")

    print("\nKey output distributions:")
    print(f"  median_log2fc_cellline median    : {metrics['median_log2fc_cellline_median']}")
    print(f"  mean_log2fc_cellline median      : {metrics['mean_log2fc_cellline_median']}")
    print(f"  std across replicates median     : {metrics['std_log2fc_across_replicates_median']}")
    print(f"  median_n_guides median           : {metrics['median_n_guides_median']}")


def save_outputs(cellline_features: pd.DataFrame, qc_df: pd.DataFrame) -> None:
    """Save final cell-line features and QC table."""
    cellline_features.to_csv(OUTPUT_PATH, index=False)
    qc_df.to_csv(QC_OUTPUT_PATH, index=False)
    print(f"\nSaved cell-line feature output to: {OUTPUT_PATH}")
    print(f"Saved QC output to: {QC_OUTPUT_PATH}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=== Phase 3 / Step III.h: Aggregate replicates to cell-line features ===")
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Phase 3 folder: {PHASE3_DIR}\n")

    gene_features, screen_metadata = load_inputs()
    inspect_metadata_columns(screen_metadata)
    metadata_lookup = prepare_metadata_lookup(screen_metadata)
    overlap_qc = check_replicate_overlap(gene_features, metadata_lookup)
    merged = merge_gene_features_with_metadata(gene_features, metadata_lookup)
    cellline_features, aggregation_counters = aggregate_to_cellline_features(merged)
    qc_df = build_qc_metrics(merged, cellline_features, overlap_qc, aggregation_counters)

    print_qc_summary(qc_df)
    save_outputs(cellline_features, qc_df)

    print("\nStep III.h complete.")


if __name__ == "__main__":
    main()
