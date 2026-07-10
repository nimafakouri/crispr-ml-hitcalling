"""
Phase 4 - Build ML training dataset.

This script joins Phase 3 gene-cell-line CRISPR features with Hart
essential/non-essential gene labels.

Inputs:
    data/processed/phase3_inputs/gene_level_log2fc_features_by_cellline.csv
    data/raw/hart_essential_genes.csv
    data/raw/hart_nonessential_genes.csv

Outputs:
    data/processed/phase4_ml/ml_training_dataset_gene_cellline.csv
    data/processed/phase4_ml/hart_gene_labels.csv
    data/processed/phase4_ml/phase4_training_dataset_qc.csv
    data/processed/phase4_ml/phase4_feature_missingness_report.csv
    data/processed/phase4_ml/unlabeled_gene_cellline_features.csv

Final training output format:
    gene
    cell_line_id
    engineered CRISPR feature columns
    essential_label

Important:
    Do not randomly split this dataset row-by-row later. Hart labels are
    gene-level labels repeated across cell lines, so split strategy should be
    chosen deliberately, such as by gene, by cell line, or by lineage.

Run only after reviewing:
    python src/p4_build_ml_training_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path("/Volumes/nimas_usb/project_data_repo/crispr_ai_ml")
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PHASE3_DIR = PROCESSED_DIR / "phase3_inputs"
PHASE4_DIR = PROCESSED_DIR / "phase4_ml"

FEATURES_PATH = PHASE3_DIR / "gene_level_log2fc_features_by_cellline.csv"
HART_ESSENTIAL_PATH = RAW_DIR / "hart_essential_genes.csv"
HART_NONESSENTIAL_PATH = RAW_DIR / "hart_nonessential_genes.csv"

TRAINING_OUTPUT_PATH = PHASE4_DIR / "ml_training_dataset_gene_cellline.csv"
LABEL_OUTPUT_PATH = PHASE4_DIR / "hart_gene_labels.csv"
QC_OUTPUT_PATH = PHASE4_DIR / "phase4_training_dataset_qc.csv"
MISSINGNESS_OUTPUT_PATH = PHASE4_DIR / "phase4_feature_missingness_report.csv"
UNLABELED_OUTPUT_PATH = PHASE4_DIR / "unlabeled_gene_cellline_features.csv"


# --------------------------------------------------------------------------- #
# Column names and validation thresholds
# --------------------------------------------------------------------------- #
GENE_COL = "gene"
CELL_LINE_ID_COL = "cell_line_id"
CELL_LINE_NAME_COL = "cell_line_name"
LABEL_COL = "essential_label"

GENE_COLUMN_CANDIDATES = [
    "gene",
    "Gene",
    "symbol",
    "GeneSymbol",
    "HugoSymbol",
]

EXPECTED_FEATURE_COLS = [
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

REQUIRED_FEATURE_TABLE_COLS = [
    GENE_COL,
    CELL_LINE_ID_COL,
    *EXPECTED_FEATURE_COLS,
]

MIN_OVERLAPPING_GENES = 100
MIN_HART_GENE_FOUND_IN_FEATURES_PCT = 50.0


# --------------------------------------------------------------------------- #
# General helpers
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


def read_csv_with_clean_headers(path: Path) -> pd.DataFrame:
    """Read a CSV and strip BOM/whitespace from column headers."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df


def print_column_preview(label: str, columns: list[str], n: int = 12) -> None:
    """Print a compact column preview."""
    print(f"{label} columns ({len(columns)}): {columns[:n]}")
    if len(columns) > n:
        print("  ...")


def find_gene_column(df: pd.DataFrame, file_name: str) -> str:
    """Find the likely gene-symbol column in a dataframe."""
    for candidate in GENE_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    normalized_lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in GENE_COLUMN_CANDIDATES:
        normalized = candidate.strip().lower()
        if normalized in normalized_lookup:
            return normalized_lookup[normalized]

    raise ValueError(
        f"Could not identify a gene-symbol column in {file_name}. "
        f"Expected one of: {GENE_COLUMN_CANDIDATES}. "
        f"Observed columns: {df.columns.tolist()}"
    )


def clean_gene_symbol(values: pd.Series) -> pd.Series:
    """Standardize gene symbols for exact joins."""
    cleaned = values.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"\s+\(\s*\d+\s*\)$", "", regex=True)
    cleaned = cleaned.str.upper()
    cleaned = cleaned.mask(cleaned == "")
    return cleaned


def count_infinite_values(series: pd.Series) -> int:
    """Count positive or negative infinite values in a numeric-like series."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=float, na_value=np.nan)
    return int(np.isinf(numeric_array).sum())


def summarize_numeric_series(series: pd.Series) -> dict[str, float]:
    """Return compact summaries for finite numeric values only."""
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return {"min": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "max": np.nan}

    return {
        "min": float(finite.min()),
        "p25": float(finite.quantile(0.25)),
        "median": float(finite.median()),
        "p75": float(finite.quantile(0.75)),
        "max": float(finite.max()),
    }


# --------------------------------------------------------------------------- #
# Hart labels
# --------------------------------------------------------------------------- #
def load_hart_gene_sets() -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Load Hart essential and non-essential gene sets."""
    print("Loading Hart gene sets...")
    essential_df = read_csv_with_clean_headers(HART_ESSENTIAL_PATH)
    nonessential_df = read_csv_with_clean_headers(HART_NONESSENTIAL_PATH)

    essential_gene_col = find_gene_column(essential_df, HART_ESSENTIAL_PATH.name)
    nonessential_gene_col = find_gene_column(nonessential_df, HART_NONESSENTIAL_PATH.name)

    print("\nHart essential genes:")
    print(f"  shape       : {essential_df.shape}")
    print(f"  columns     : {essential_df.columns.tolist()}")
    print(f"  gene column : {essential_gene_col}")
    print("  first 5 rows:")
    print(essential_df.head())
    print("  missing values:")
    print(essential_df.isna().sum().to_dict())

    print("\nHart non-essential genes:")
    print(f"  shape       : {nonessential_df.shape}")
    print(f"  columns     : {nonessential_df.columns.tolist()}")
    print(f"  gene column : {nonessential_gene_col}")
    print("  first 5 rows:")
    print(nonessential_df.head())
    print("  missing values:")
    print(nonessential_df.isna().sum().to_dict())

    return essential_df, nonessential_df, essential_gene_col, nonessential_gene_col


def inspect_gene_cleaning(raw_genes: pd.Series, cleaned_genes: pd.Series, label: str) -> None:
    """Print gene-symbol cleaning diagnostics."""
    raw_as_string = raw_genes.astype("string")
    stripped = raw_as_string.str.strip()
    bracket_id_count = int(stripped.str.contains(r"\(\s*\d+\s*\)$", regex=True, na=False).sum())
    whitespace_count = int(raw_as_string.ne(stripped).fillna(False).sum())
    internal_space_count = int(stripped.str.contains(r"\s", regex=True, na=False).sum())
    missing_after_cleaning = int(cleaned_genes.isna().sum())
    duplicate_gene_count = int(cleaned_genes.dropna().duplicated(keep=False).sum())
    duplicate_unique_count = int(cleaned_genes.dropna()[cleaned_genes.dropna().duplicated()].nunique())

    print(f"\n{label} gene-symbol cleaning QC:")
    print(f"  raw rows                         : {len(raw_genes)}")
    print(f"  missing after cleaning           : {missing_after_cleaning}")
    print(f"  raw values with edge whitespace  : {whitespace_count}")
    print(f"  raw values with internal spaces  : {internal_space_count}")
    print(f"  raw values with bracketed IDs    : {bracket_id_count}")
    print(f"  duplicated cleaned-gene rows     : {duplicate_gene_count}")
    print(f"  unique duplicated cleaned genes  : {duplicate_unique_count}")
    print(f"  first cleaned gene examples      : {cleaned_genes.dropna().head().to_list()}")


def clean_unique_genes(df: pd.DataFrame, gene_col: str, label: str) -> pd.Series:
    """Clean a gene column and return unique non-missing gene symbols."""
    cleaned = clean_gene_symbol(df[gene_col])
    inspect_gene_cleaning(df[gene_col], cleaned, label)
    return pd.Series(cleaned.dropna().drop_duplicates().sort_values().to_list(), name=GENE_COL)


def build_label_table(
    essential_df: pd.DataFrame,
    nonessential_df: pd.DataFrame,
    essential_gene_col: str,
    nonessential_gene_col: str,
) -> pd.DataFrame:
    """Build the clean Hart label table."""
    essential_genes = clean_unique_genes(essential_df, essential_gene_col, "Hart essential")
    nonessential_genes = clean_unique_genes(nonessential_df, nonessential_gene_col, "Hart non-essential")

    essential_set = set(essential_genes)
    nonessential_set = set(nonessential_genes)
    overlap = essential_set & nonessential_set

    print("\nHart label-set overlap QC:")
    print(f"  unique essential genes      : {len(essential_set)}")
    print(f"  unique non-essential genes  : {len(nonessential_set)}")
    print(f"  overlap between label sets  : {len(overlap)}")
    if overlap:
        raise ValueError(
            "Some genes appear in both Hart essential and non-essential sets. "
            f"First examples: {sorted(overlap)[:10]}"
        )

    essential_labels = pd.DataFrame(
        {GENE_COL: essential_genes, LABEL_COL: 1}
    )
    nonessential_labels = pd.DataFrame(
        {GENE_COL: nonessential_genes, LABEL_COL: 0}
    )
    label_table = pd.concat([essential_labels, nonessential_labels], ignore_index=True)

    conflicting = label_table.groupby(GENE_COL)[LABEL_COL].nunique(dropna=True)
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        raise ValueError(
            "Conflicting Hart labels found after label-table construction. "
            f"First examples: {conflicting.head(10).index.to_list()}"
        )

    before_dedup = len(label_table)
    label_table = label_table.drop_duplicates(subset=[GENE_COL, LABEL_COL]).sort_values(GENE_COL)
    label_table[LABEL_COL] = label_table[LABEL_COL].astype(int)

    print("\nHart label table QC:")
    print(f"  rows before safe duplicate removal : {before_dedup}")
    print(f"  rows after safe duplicate removal  : {len(label_table)}")
    print(f"  total labelled genes               : {label_table[GENE_COL].nunique()}")
    print("  label counts:")
    print(label_table[LABEL_COL].value_counts().sort_index().to_dict())

    return label_table.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Engineered features
# --------------------------------------------------------------------------- #
def load_engineered_features() -> pd.DataFrame:
    """Load the Phase 3 gene-cell-line feature table."""
    print("\nLoading engineered cell-line features...")
    features = read_csv_with_clean_headers(FEATURES_PATH)
    require_columns(features, REQUIRED_FEATURE_TABLE_COLS, FEATURES_PATH.name)

    print(f"Feature table shape: {features.shape}")
    print_column_preview("Feature table", features.columns.tolist(), n=20)
    print("First few feature rows:")
    print(features.head())
    print(f"Unique genes before cleaning     : {features[GENE_COL].nunique(dropna=True)}")
    print(f"Unique cell lines                : {features[CELL_LINE_ID_COL].nunique(dropna=True)}")
    print(f"Missing gene values              : {int(features[GENE_COL].isna().sum())}")
    print(f"Missing cell_line_id values      : {int(features[CELL_LINE_ID_COL].isna().sum())}")

    return features


def clean_feature_genes(features: pd.DataFrame) -> pd.DataFrame:
    """Clean feature-table gene symbols and normalize IDs."""
    output = features.copy()
    unique_genes_before = output[GENE_COL].nunique(dropna=True)

    output[GENE_COL] = clean_gene_symbol(output[GENE_COL])
    output[CELL_LINE_ID_COL] = output[CELL_LINE_ID_COL].astype("string").str.strip()
    output[CELL_LINE_ID_COL] = output[CELL_LINE_ID_COL].mask(output[CELL_LINE_ID_COL] == "")
    if CELL_LINE_NAME_COL in output.columns:
        output[CELL_LINE_NAME_COL] = output[CELL_LINE_NAME_COL].astype("string").str.strip()
        output[CELL_LINE_NAME_COL] = output[CELL_LINE_NAME_COL].mask(output[CELL_LINE_NAME_COL] == "")

    print("\nFeature-table gene cleaning QC:")
    print(f"  unique genes before cleaning : {unique_genes_before}")
    print(f"  unique genes after cleaning  : {output[GENE_COL].nunique(dropna=True)}")
    print(f"  missing genes after cleaning : {int(output[GENE_COL].isna().sum())}")
    print(f"  missing cell_line_id values  : {int(output[CELL_LINE_ID_COL].isna().sum())}")
    print(f"  first cleaned gene examples  : {output[GENE_COL].dropna().drop_duplicates().head().to_list()}")

    return output


def check_gene_overlap(features: pd.DataFrame, label_table: pd.DataFrame) -> dict[str, int | float]:
    """Compare feature-table genes with Hart-labelled genes before joining."""
    feature_genes = set(features[GENE_COL].dropna())
    label_genes = set(label_table[GENE_COL].dropna())
    overlap = feature_genes & label_genes
    feature_only = feature_genes - label_genes
    label_only = label_genes - feature_genes

    feature_gene_label_coverage_pct = (
        len(overlap) / len(feature_genes) * 100.0 if feature_genes else 0.0
    )
    hart_gene_found_in_features_pct = (
        len(overlap) / len(label_genes) * 100.0 if label_genes else 0.0
    )

    print("\nGene overlap QC:")
    print(f"  feature genes                         : {len(feature_genes)}")
    print(f"  Hart-labelled genes                   : {len(label_genes)}")
    print(f"  overlapping genes                     : {len(overlap)}")
    print(f"  feature-only genes                    : {len(feature_only)}")
    print(f"  Hart-only genes                       : {len(label_only)}")
    print(f"  feature genes with Hart labels        : {feature_gene_label_coverage_pct:.2f}%")
    print(f"  Hart genes found in feature table     : {hart_gene_found_in_features_pct:.2f}%")
    print(f"  first feature-only genes              : {sorted(feature_only)[:10]}")
    print(f"  first Hart-only genes                 : {sorted(label_only)[:10]}")

    if len(overlap) < MIN_OVERLAPPING_GENES:
        raise ValueError(
            f"Only {len(overlap)} genes overlap between features and Hart labels. "
            "This is too low; inspect gene-symbol cleaning and selected gene columns."
        )
    if hart_gene_found_in_features_pct < MIN_HART_GENE_FOUND_IN_FEATURES_PCT:
        raise ValueError(
            "Hart gene coverage in the feature table is low "
            f"({hart_gene_found_in_features_pct:.2f}%). "
            "Inspect gene-symbol cleaning before building the training dataset."
        )

    return {
        "unique_feature_genes": len(feature_genes),
        "unique_hart_label_genes": len(label_genes),
        "overlapping_genes": len(overlap),
        "feature_only_genes": len(feature_only),
        "hart_only_genes": len(label_only),
        "feature_gene_label_coverage_pct": feature_gene_label_coverage_pct,
        "hart_gene_found_in_features_pct": hart_gene_found_in_features_pct,
    }


# --------------------------------------------------------------------------- #
# Join labels and validate ML columns
# --------------------------------------------------------------------------- #
def join_labels_to_features(
    features: pd.DataFrame,
    label_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Join Hart labels to features and keep unlabelled rows separately."""
    rows_before_required_drop = len(features)
    missing_gene_rows = int(features[GENE_COL].isna().sum())
    missing_cell_line_rows = int(features[CELL_LINE_ID_COL].isna().sum())

    joinable_features = features.dropna(subset=[GENE_COL, CELL_LINE_ID_COL]).copy()
    rows_removed_missing_required_ids = rows_before_required_drop - len(joinable_features)

    rows_before_join = len(joinable_features)
    joined = joinable_features.merge(label_table, how="left", on=GENE_COL)
    rows_after_left_join = len(joined)
    if rows_after_left_join != rows_before_join:
        raise ValueError(
            "Rows changed after joining labels. "
            f"Before join: {rows_before_join}; after join: {rows_after_left_join}. "
            "This usually means the label table has duplicate genes."
        )

    labelled_mask = joined[LABEL_COL].notna()
    training_df = joined.loc[labelled_mask].copy()
    unlabeled_df = joined.loc[~labelled_mask].drop(columns=[LABEL_COL]).copy()
    training_df[LABEL_COL] = training_df[LABEL_COL].astype(int)

    print("\nLabel join QC:")
    print(f"  feature rows before required-ID drop : {rows_before_required_drop}")
    print(f"  rows missing gene                    : {missing_gene_rows}")
    print(f"  rows missing cell_line_id            : {missing_cell_line_rows}")
    print(f"  rows removed before join             : {rows_removed_missing_required_ids}")
    print(f"  feature rows before join             : {rows_before_join}")
    print(f"  training rows after join             : {len(training_df)}")
    print(f"  unlabelled rows                      : {len(unlabeled_df)}")
    print(f"  unique labelled genes                : {training_df[GENE_COL].nunique(dropna=True)}")
    print(f"  unique labelled cell lines           : {training_df[CELL_LINE_ID_COL].nunique(dropna=True)}")
    print("  label counts:")
    print(training_df[LABEL_COL].value_counts().sort_index().to_dict())

    return training_df, unlabeled_df, {
        "feature_rows_before_required_drop": rows_before_required_drop,
        "rows_missing_gene": missing_gene_rows,
        "rows_missing_cell_line_id": missing_cell_line_rows,
        "rows_removed_missing_required_ids": rows_removed_missing_required_ids,
        "feature_rows_before_join": rows_before_join,
        "training_rows_after_join": len(training_df),
        "unlabeled_rows": len(unlabeled_df),
    }


def summarize_label_repetition(training_df: pd.DataFrame) -> dict[str, int | float]:
    """Summarize how gene-level Hart labels repeat across cell lines."""
    gene_repetition = (
        training_df.groupby(GENE_COL)
        .agg(
            rows_per_gene=(LABEL_COL, "size"),
            cell_lines_per_gene=(CELL_LINE_ID_COL, "nunique"),
        )
        .reset_index()
    )

    cell_lines_per_gene = gene_repetition["cell_lines_per_gene"]
    rows_per_gene = gene_repetition["rows_per_gene"]

    print("\nLabel repetition across cell lines:")
    print(f"  labelled genes                      : {len(gene_repetition)}")
    print(f"  median rows per labelled gene       : {float(rows_per_gene.median())}")
    print(f"  min rows per labelled gene          : {int(rows_per_gene.min())}")
    print(f"  max rows per labelled gene          : {int(rows_per_gene.max())}")
    print(f"  median cell lines per labelled gene : {float(cell_lines_per_gene.median())}")
    print(f"  min cell lines per labelled gene    : {int(cell_lines_per_gene.min())}")
    print(f"  max cell lines per labelled gene    : {int(cell_lines_per_gene.max())}")
    print("  first gene repetition examples:")
    print(gene_repetition.head().to_dict(orient="records"))

    return {
        "median_rows_per_labelled_gene": float(rows_per_gene.median()),
        "min_rows_per_labelled_gene": int(rows_per_gene.min()),
        "max_rows_per_labelled_gene": int(rows_per_gene.max()),
        "median_cell_lines_per_labelled_gene": float(cell_lines_per_gene.median()),
        "min_cell_lines_per_labelled_gene": int(cell_lines_per_gene.min()),
        "max_cell_lines_per_labelled_gene": int(cell_lines_per_gene.max()),
    }


def coerce_numeric_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    table_name: str,
) -> pd.DataFrame:
    """Convert expected feature columns to numeric without hiding bad strings."""
    require_columns(df, feature_cols, table_name)
    output = df.copy()
    for col in feature_cols:
        original_missing = output[col].isna()
        converted = pd.to_numeric(output[col], errors="coerce")
        coerced_to_missing = int((converted.isna() & ~original_missing).sum())
        if coerced_to_missing:
            examples = output.loc[converted.isna() & ~original_missing, col].head().to_list()
            raise ValueError(
                f"{table_name}.{col} has {coerced_to_missing} non-numeric value(s). "
                f"First examples: {examples}"
            )
        output[col] = converted
    return output


def define_feature_columns(training_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str, list[str]]:
    """Define identifier, label, and feature columns for supervised ML."""
    id_cols = [GENE_COL, CELL_LINE_ID_COL]
    if CELL_LINE_NAME_COL in training_df.columns:
        id_cols.append(CELL_LINE_NAME_COL)

    require_columns(training_df, id_cols + [LABEL_COL] + EXPECTED_FEATURE_COLS, "training dataframe")
    training_df = coerce_numeric_feature_columns(
        training_df,
        EXPECTED_FEATURE_COLS,
        "training dataframe",
    )

    print("\nML column definitions:")
    print(f"  identifier columns : {id_cols}")
    print(f"  label column       : {LABEL_COL}")
    print(f"  feature columns    : {EXPECTED_FEATURE_COLS}")

    return training_df, id_cols, LABEL_COL, EXPECTED_FEATURE_COLS


def validate_feature_columns(training_df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Print checks for numeric features, missingness, infinities, constants, and range."""
    non_numeric_cols = [
        col for col in feature_cols if not pd.api.types.is_numeric_dtype(training_df[col])
    ]
    if non_numeric_cols:
        raise ValueError(f"Feature columns are not numeric: {non_numeric_cols}")

    infinite_counts = {col: count_infinite_values(training_df[col]) for col in feature_cols}
    cols_with_infinite = {col: count for col, count in infinite_counts.items() if count > 0}
    if cols_with_infinite:
        raise ValueError(f"Feature columns contain infinite values: {cols_with_infinite}")

    missing_counts = {col: int(training_df[col].isna().sum()) for col in feature_cols}
    constant_cols = [
        col for col in feature_cols if training_df[col].dropna().nunique() <= 1
    ]

    print("\nFeature-column validation:")
    print(f"  non-numeric feature columns : {non_numeric_cols}")
    print(f"  columns with infinite values: {cols_with_infinite}")
    print(f"  constant feature columns    : {constant_cols}")
    print("  missing values by feature:")
    print(missing_counts)
    print("  finite feature ranges:")
    for col in feature_cols:
        summary = summarize_numeric_series(training_df[col])
        print(f"    {col}: {summary}")


def build_missingness_report(training_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build a feature-level missingness report."""
    rows = []
    denominator = len(training_df)
    for col in feature_cols:
        missing_count = int(training_df[col].isna().sum())
        missing_pct = missing_count / denominator * 100.0 if denominator else 0.0
        rows.append(
            {
                "feature": col,
                "dtype": str(training_df[col].dtype),
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
                "infinite_count": count_infinite_values(training_df[col]),
            }
        )

    missingness_report = pd.DataFrame(rows)
    print("\nFeature missingness report:")
    print(missingness_report)

    std_missing = missingness_report.loc[
        missingness_report["feature"] == "std_log2fc_across_replicates",
        "missing_count",
    ]
    if not std_missing.empty and int(std_missing.iloc[0]) > 0:
        print(
            "  Note: missing std_log2fc_across_replicates can be normal when "
            "replicate_count = 1."
        )

    return missingness_report


def check_class_balance(training_df: pd.DataFrame) -> dict[str, int | float]:
    """Check essential/non-essential balance at row and gene levels."""
    observed_labels = set(training_df[LABEL_COL].dropna().unique())
    if observed_labels - {0, 1}:
        raise ValueError(f"Unexpected labels found: {sorted(observed_labels)}")

    row_counts = training_df[LABEL_COL].value_counts().reindex([0, 1], fill_value=0)
    gene_counts = (
        training_df[[GENE_COL, LABEL_COL]]
        .drop_duplicates()
        .groupby(LABEL_COL)[GENE_COL]
        .nunique()
        .reindex([0, 1], fill_value=0)
    )

    total_rows = int(row_counts.sum())
    total_genes = int(gene_counts.sum())
    essential_rows = int(row_counts.loc[1])
    nonessential_rows = int(row_counts.loc[0])
    essential_genes = int(gene_counts.loc[1])
    nonessential_genes = int(gene_counts.loc[0])

    if essential_rows == 0 or nonessential_rows == 0:
        raise ValueError("Training data must contain both essential and non-essential rows.")
    if essential_genes == 0 or nonessential_genes == 0:
        raise ValueError("Training data must contain both essential and non-essential genes.")

    essential_row_pct = essential_rows / total_rows * 100.0 if total_rows else 0.0
    nonessential_row_pct = nonessential_rows / total_rows * 100.0 if total_rows else 0.0
    essential_gene_pct = essential_genes / total_genes * 100.0 if total_genes else 0.0
    nonessential_gene_pct = nonessential_genes / total_genes * 100.0 if total_genes else 0.0

    print("\nClass balance QC:")
    print(f"  essential rows        : {essential_rows} ({essential_row_pct:.2f}%)")
    print(f"  non-essential rows    : {nonessential_rows} ({nonessential_row_pct:.2f}%)")
    print(f"  essential genes       : {essential_genes} ({essential_gene_pct:.2f}%)")
    print(f"  non-essential genes   : {nonessential_genes} ({nonessential_gene_pct:.2f}%)")

    return {
        "essential_rows": essential_rows,
        "nonessential_rows": nonessential_rows,
        "essential_row_pct": essential_row_pct,
        "nonessential_row_pct": nonessential_row_pct,
        "essential_genes": essential_genes,
        "nonessential_genes": nonessential_genes,
        "essential_gene_pct": essential_gene_pct,
        "nonessential_gene_pct": nonessential_gene_pct,
    }


# --------------------------------------------------------------------------- #
# QC and output
# --------------------------------------------------------------------------- #
def build_qc_report(
    features: pd.DataFrame,
    label_table: pd.DataFrame,
    training_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    overlap_qc: dict[str, int | float],
    join_qc: dict[str, int],
    repetition_qc: dict[str, int | float],
    class_balance_qc: dict[str, int | float],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build a compact Phase 4 QC report."""
    metrics = [
        ("raw_feature_rows_loaded", len(features)),
        ("feature_rows_before_join", join_qc["feature_rows_before_join"]),
        ("training_rows_after_join", len(training_df)),
        ("unlabeled_rows", len(unlabeled_df)),
        ("rows_removed_missing_required_ids", join_qc["rows_removed_missing_required_ids"]),
        ("rows_missing_gene", join_qc["rows_missing_gene"]),
        ("rows_missing_cell_line_id", join_qc["rows_missing_cell_line_id"]),
        ("unique_feature_genes", overlap_qc["unique_feature_genes"]),
        ("unique_hart_label_genes", overlap_qc["unique_hart_label_genes"]),
        ("overlapping_genes", overlap_qc["overlapping_genes"]),
        ("feature_only_genes", overlap_qc["feature_only_genes"]),
        ("hart_only_genes", overlap_qc["hart_only_genes"]),
        ("feature_gene_label_coverage_pct", overlap_qc["feature_gene_label_coverage_pct"]),
        ("hart_gene_found_in_features_pct", overlap_qc["hart_gene_found_in_features_pct"]),
        ("unique_training_genes", training_df[GENE_COL].nunique(dropna=True)),
        ("unique_training_cell_lines", training_df[CELL_LINE_ID_COL].nunique(dropna=True)),
        ("unique_unlabeled_genes", unlabeled_df[GENE_COL].nunique(dropna=True)),
        ("unique_unlabeled_cell_lines", unlabeled_df[CELL_LINE_ID_COL].nunique(dropna=True)),
        ("hart_label_table_rows", len(label_table)),
        ("essential_rows", class_balance_qc["essential_rows"]),
        ("nonessential_rows", class_balance_qc["nonessential_rows"]),
        ("essential_row_pct", class_balance_qc["essential_row_pct"]),
        ("nonessential_row_pct", class_balance_qc["nonessential_row_pct"]),
        ("essential_genes", class_balance_qc["essential_genes"]),
        ("nonessential_genes", class_balance_qc["nonessential_genes"]),
        ("essential_gene_pct", class_balance_qc["essential_gene_pct"]),
        ("nonessential_gene_pct", class_balance_qc["nonessential_gene_pct"]),
        ("number_of_feature_columns", len(feature_cols)),
    ]

    for metric_name, metric_value in repetition_qc.items():
        metrics.append((metric_name, metric_value))

    for col in feature_cols:
        summary = summarize_numeric_series(training_df[col])
        for stat_name, value in summary.items():
            metrics.append((f"{col}_{stat_name}", value))

    qc_df = pd.DataFrame(metrics, columns=["metric", "value"])
    print("\nPhase 4 QC report:")
    print(qc_df)
    return qc_df


def save_outputs(
    training_df: pd.DataFrame,
    label_table: pd.DataFrame,
    qc_df: pd.DataFrame,
    missingness_report: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
) -> None:
    """Save all Phase 4 output tables."""
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)

    training_df.to_csv(TRAINING_OUTPUT_PATH, index=False)
    label_table.to_csv(LABEL_OUTPUT_PATH, index=False)
    qc_df.to_csv(QC_OUTPUT_PATH, index=False)
    missingness_report.to_csv(MISSINGNESS_OUTPUT_PATH, index=False)
    unlabeled_df.to_csv(UNLABELED_OUTPUT_PATH, index=False)

    print("\nSaved Phase 4 outputs:")
    print(f"  training dataset       : {TRAINING_OUTPUT_PATH}")
    print(f"  Hart label table       : {LABEL_OUTPUT_PATH}")
    print(f"  QC report              : {QC_OUTPUT_PATH}")
    print(f"  missingness report     : {MISSINGNESS_OUTPUT_PATH}")
    print(f"  unlabelled feature rows: {UNLABELED_OUTPUT_PATH}")


def validate_saved_outputs(feature_cols: list[str], expected_training_rows: int) -> None:
    """Reload saved outputs and run final lightweight validation."""
    required_outputs = [
        TRAINING_OUTPUT_PATH,
        LABEL_OUTPUT_PATH,
        QC_OUTPUT_PATH,
        MISSINGNESS_OUTPUT_PATH,
        UNLABELED_OUTPUT_PATH,
    ]
    missing = [path for path in required_outputs if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Expected output file(s) were not created:\n{missing_text}")

    saved_training = read_csv_with_clean_headers(TRAINING_OUTPUT_PATH)
    require_columns(saved_training, [GENE_COL, CELL_LINE_ID_COL, LABEL_COL] + feature_cols, "saved training dataset")
    if "Unnamed: 0" in saved_training.columns:
        raise ValueError("Saved training dataset contains an accidental index column: Unnamed: 0")
    if len(saved_training) != expected_training_rows:
        raise ValueError(
            "Saved training row count does not match QC. "
            f"Expected {expected_training_rows}; observed {len(saved_training)}."
        )
    if set(saved_training[LABEL_COL].dropna().unique()) - {0, 1}:
        raise ValueError("Saved training dataset has labels outside {0, 1}.")
    if saved_training[GENE_COL].isna().any():
        raise ValueError("Saved training dataset has missing gene values.")
    if saved_training[CELL_LINE_ID_COL].isna().any():
        raise ValueError("Saved training dataset has missing cell_line_id values.")

    saved_training = coerce_numeric_feature_columns(
        saved_training,
        feature_cols,
        "saved training dataset",
    )
    labels_present = set(saved_training[LABEL_COL].dropna().unique())
    if labels_present != {0, 1}:
        raise ValueError(f"Saved training dataset does not contain both labels: {labels_present}")
    if saved_training[GENE_COL].nunique(dropna=True) < MIN_OVERLAPPING_GENES:
        raise ValueError("Saved training dataset has too few labelled genes.")
    if saved_training[CELL_LINE_ID_COL].nunique(dropna=True) < 2:
        raise ValueError("Saved training dataset has too few cell lines.")

    print("\nFinal saved-output validation:")
    print(f"  saved training shape : {saved_training.shape}")
    print(f"  saved columns        : {saved_training.columns.tolist()}")
    print("  saved label counts:")
    print(saved_training[LABEL_COL].value_counts().sort_index().to_dict())
    print("  first saved rows:")
    print(saved_training.head())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=== Phase 4: Build ML training dataset ===")
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Phase 3 input folder: {PHASE3_DIR}")
    print(f"Phase 4 output folder: {PHASE4_DIR}\n")

    check_required_files([FEATURES_PATH, HART_ESSENTIAL_PATH, HART_NONESSENTIAL_PATH])
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)

    essential_df, nonessential_df, essential_gene_col, nonessential_gene_col = load_hart_gene_sets()
    label_table = build_label_table(
        essential_df,
        nonessential_df,
        essential_gene_col,
        nonessential_gene_col,
    )

    features = load_engineered_features()
    features = clean_feature_genes(features)
    overlap_qc = check_gene_overlap(features, label_table)

    training_df, unlabeled_df, join_qc = join_labels_to_features(features, label_table)
    repetition_qc = summarize_label_repetition(training_df)
    training_df, id_cols, label_col, feature_cols = define_feature_columns(training_df)
    unlabeled_df = coerce_numeric_feature_columns(
        unlabeled_df,
        feature_cols,
        "unlabelled feature dataframe",
    )

    validate_feature_columns(training_df, feature_cols)
    missingness_report = build_missingness_report(training_df, feature_cols)
    class_balance_qc = check_class_balance(training_df)
    qc_df = build_qc_report(
        features=features,
        label_table=label_table,
        training_df=training_df,
        unlabeled_df=unlabeled_df,
        overlap_qc=overlap_qc,
        join_qc=join_qc,
        repetition_qc=repetition_qc,
        class_balance_qc=class_balance_qc,
        feature_cols=feature_cols,
    )

    ordered_cols = id_cols + feature_cols + [label_col]
    training_df = training_df[ordered_cols]
    save_outputs(
        training_df=training_df,
        label_table=label_table,
        qc_df=qc_df,
        missingness_report=missingness_report,
        unlabeled_df=unlabeled_df,
    )
    validate_saved_outputs(feature_cols, expected_training_rows=len(training_df))

    print("\nStep Phase 4 complete.")
    print("Reminder: do not randomly split this dataset row-by-row for ML evaluation.")


if __name__ == "__main__":
    main()
