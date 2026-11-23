"""
Phase 5 - Modeling & Evaluation.

Predicts Hart essential (1) versus Hart non-essential (0) genes from the
Phase 4 gene-cell-line feature table, and measures performance on genuinely
unseen genes (leakage-safe evaluation).

Implementation progress
------------------------
This revision implements Steps 5.0-5.4 of
project_docs/handsoffs/phase5_modeling_evaluation_guidelines.md:

    5.0  Reproducible configuration and paths
    5.1  Load and validate the supervised (labelled) table
    5.2  Define and validate the feature contract (model matrices are
         assembled per train/test partition in Stage 2, not here)
    5.3  Inspect missing / infinite feature values
    5.4  Split by gene and freeze the train/test gene assignments

No model is trained yet. This script only loads, validates, cleans, and
splits data. It does not report any model performance and none should be
inferred from it.

Steps 5.5 onward (preprocessing pipelines, baselines, candidate models,
grouped cross-validation, model selection, final test evaluation, error
analysis, feature importance, and the complete saved pipeline) are added in
later revisions once this stage's output has been reviewed, per the staged
build plan in the handoff document.

Inputs:
    data/processed/phase4_ml/ml_training_dataset_gene_cellline.csv

Outputs (this revision):
    data/processed/phase5_modeling/train_genes.csv
    data/processed/phase5_modeling/test_genes.csv
    data/processed/phase5_modeling/phase5_feature_missingness_report.csv
    data/processed/phase5_modeling/phase5_modeling_qc.csv
    data/processed/phase5_modeling/run_metadata.json

Important:
    The unit of each row is a gene-cell-line pair; the Hart label belongs to
    the gene. The train/test split below is performed on unique genes, not
    rows, and is frozen to disk so later stages (and later reruns) reuse the
    same gene membership. Hart *labels* are never trusted from the frozen
    split files: they are always re-derived from the current Phase 4 output,
    so an upstream label change cannot silently go stale. Do not consult the
    test genes' features, labels, or any downstream score when choosing a
    model in later stages. The whole-dataset diagnostics printed below
    (feature ranges, missingness) are input-QC only, run before any split, to
    catch problems in the Phase 4 output itself -- they must not be reused to
    choose imputation values, thresholds, or models. Those decisions belong
    inside training-fold-only pipelines in later stages.

Run:
    python src/p5_modeling_evaluation.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

try:
    from .data_paths import DATA_ROOT
except ImportError:  # Support running this file directly.
    from data_paths import DATA_ROOT


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = DATA_ROOT / "processed"
PHASE4_DIR = PROCESSED_DIR / "phase4_ml"
PHASE5_DIR = PROCESSED_DIR / "phase5_modeling"

TRAINING_INPUT_PATH = PHASE4_DIR / "ml_training_dataset_gene_cellline.csv"

TRAIN_GENES_PATH = PHASE5_DIR / "train_genes.csv"
TEST_GENES_PATH = PHASE5_DIR / "test_genes.csv"
MISSINGNESS_REPORT_PATH = PHASE5_DIR / "phase5_feature_missingness_report.csv"
QC_OUTPUT_PATH = PHASE5_DIR / "phase5_modeling_qc.csv"
RUN_METADATA_PATH = PHASE5_DIR / "run_metadata.json"


# --------------------------------------------------------------------------- #
# Column names and feature contract
# --------------------------------------------------------------------------- #
GENE_COL = "gene"
CELL_LINE_ID_COL = "cell_line_id"
LABEL_COL = "essential_label"

# Mirrors the feature contract written by src/p4_build_ml_training_dataset.py.
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

# Features that are proportions and must fall within [0, 1] (small floating
# tolerance allowed).
FRACTION_FEATURE_COLS = [
    "mean_frac_guides_negative",
    "mean_frac_guides_strongly_negative",
]

# Features that are counts and must be non-negative.
COUNT_FEATURE_COLS = [
    "replicate_count",
    "median_n_guides",
    "min_n_guides",
]

REQUIRED_TRAINING_COLS = [GENE_COL, CELL_LINE_ID_COL, LABEL_COL, *EXPECTED_FEATURE_COLS]


# --------------------------------------------------------------------------- #
# Reproducible configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_FRACTION = 0.20  # fraction of unique genes held out for the final test set

# Reserved for later stages (grouped CV and thresholded metrics); recorded now
# so the run's full configuration is fixed from the start.
N_CV_FOLDS = 5
CLASSIFICATION_THRESHOLD = 0.5

# Set True only to deliberately discard an existing frozen split and draw a
# new one. Leave False so reruns reuse the same gene membership.
FORCE_NEW_SPLIT = False

FRACTION_TOLERANCE = 1e-6


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
    df.columns = [str(col).replace("﻿", "").strip() for col in df.columns]
    return df


def coerce_numeric_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> tuple[pd.DataFrame, int]:
    """Coerce feature columns to numeric in place, reporting values newly lost to coercion.

    Mutates ``df`` directly (no full-table copy) to avoid holding multiple
    ~150 MB copies of the training table in memory at once.
    """
    newly_missing_by_col: dict[str, int] = {}

    for col in feature_cols:
        before_missing = int(df[col].isna().sum())
        coerced = pd.to_numeric(df[col], errors="coerce")
        after_missing = int(coerced.isna().sum())
        newly_missing_by_col[col] = after_missing - before_missing
        df[col] = coerced

    total_newly_missing = sum(newly_missing_by_col.values())
    print(f"\n{label} numeric coercion QC:")
    print(f"  values newly missing due to coercion (by feature): {newly_missing_by_col}")
    print(f"  total newly missing due to coercion: {total_newly_missing}")

    return df, total_newly_missing


# --------------------------------------------------------------------------- #
# Step 5.1 - Load and validate the supervised table
# --------------------------------------------------------------------------- #
def load_training_dataset() -> pd.DataFrame:
    """Load the Phase 4 gene-cell-line training table."""
    print("Loading Phase 4 training dataset...")
    df = read_csv_with_clean_headers(TRAINING_INPUT_PATH)
    require_columns(df, REQUIRED_TRAINING_COLS, TRAINING_INPUT_PATH.name)

    print(f"  shape         : {df.shape}")
    print(f"  memory usage  : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  dtypes        :\n{df.dtypes.to_string()}")
    print(f"  unique genes  : {df[GENE_COL].nunique(dropna=True)}")
    print(f"  unique cell lines: {df[CELL_LINE_ID_COL].nunique(dropna=True)}")
    print(f"  label counts (rows):\n{df[LABEL_COL].value_counts(dropna=False).sort_index().to_string()}")

    return df


def check_gene_id_formatting(df: pd.DataFrame) -> tuple[int, int]:
    """Flag blank gene IDs and genes that differ only by case/whitespace.

    Phase 4 already normalizes gene symbols (strip + uppercase), so this is a
    defensive check: if it trips, the Phase 4 output has drifted from its own
    contract and should be fixed there rather than silently tolerated here,
    since inconsistent formatting could split what is really one gene across
    both train and test.
    """
    non_null_genes = df[GENE_COL].dropna().astype(str)

    blank_gene_count = int((non_null_genes.str.strip() == "").sum())

    non_blank_genes = non_null_genes[non_null_genes.str.strip() != ""]
    normalized = non_blank_genes.str.strip().str.upper()
    variants_per_group = non_blank_genes.groupby(normalized).nunique()
    inconsistent_groups = variants_per_group[variants_per_group > 1]
    inconsistent_gene_formatting_count = int(len(inconsistent_groups))

    print(f"  blank/whitespace-only gene IDs       : {blank_gene_count}")
    print(f"  genes with inconsistent formatting   : {inconsistent_gene_formatting_count}")
    if inconsistent_gene_formatting_count:
        examples = {
            key: sorted(non_blank_genes[normalized == key].unique())[:3]
            for key in inconsistent_groups.index[:5]
        }
        print(f"  formatting-inconsistency examples    : {examples}")

    return blank_gene_count, inconsistent_gene_formatting_count


def validate_training_dataset(df: pd.DataFrame) -> dict[str, int | float]:
    """Validate structural and label integrity of the training table."""
    print("\nValidating training dataset...")

    missing_gene = int(df[GENE_COL].isna().sum())
    missing_cell_line = int(df[CELL_LINE_ID_COL].isna().sum())
    missing_label = int(df[LABEL_COL].isna().sum())

    duplicate_keys = int(df.duplicated(subset=[GENE_COL, CELL_LINE_ID_COL]).sum())

    invalid_labels = set(df[LABEL_COL].dropna().unique()) - {0, 1}
    invalid_label_count = int(df[LABEL_COL].isin(invalid_labels).sum()) if invalid_labels else 0

    label_nunique_per_gene = df.groupby(GENE_COL)[LABEL_COL].nunique(dropna=True)
    inconsistent_genes = label_nunique_per_gene[label_nunique_per_gene > 1]

    blank_gene_count, inconsistent_gene_formatting_count = check_gene_id_formatting(df)

    print(f"  missing gene values          : {missing_gene}")
    print(f"  missing cell_line_id values  : {missing_cell_line}")
    print(f"  missing label values         : {missing_label}")
    print(f"  duplicate (gene, cell_line_id) rows: {duplicate_keys}")
    print(f"  invalid label values observed: {sorted(invalid_labels) if invalid_labels else 'none'}")
    print(f"  genes with inconsistent Hart label: {len(inconsistent_genes)}")

    if missing_gene or missing_cell_line or missing_label:
        raise ValueError(
            "Training dataset has missing identifier or label values "
            f"(gene={missing_gene}, cell_line_id={missing_cell_line}, label={missing_label})."
        )
    if duplicate_keys:
        raise ValueError(
            f"Training dataset has {duplicate_keys} duplicate (gene, cell_line_id) rows; "
            "expected exactly one row per gene-cell-line pair."
        )
    if invalid_labels:
        raise ValueError(f"Training dataset has label values outside {{0, 1}}: {sorted(invalid_labels)}")
    if not inconsistent_genes.empty:
        raise ValueError(
            "Some genes have more than one distinct Hart label across rows. "
            f"First examples: {inconsistent_genes.index.to_list()[:10]}"
        )
    if blank_gene_count:
        raise ValueError(f"Training dataset has {blank_gene_count} blank/whitespace-only gene IDs.")
    if inconsistent_gene_formatting_count:
        raise ValueError(
            f"Training dataset has {inconsistent_gene_formatting_count} gene(s) whose formatting "
            "is inconsistent (e.g. differing only by case or whitespace). Fix gene-symbol cleaning "
            "in Phase 4 before proceeding; this would otherwise risk splitting one gene's rows "
            "across both train and test."
        )

    gene_label_table = df.drop_duplicates(subset=[GENE_COL])[[GENE_COL, LABEL_COL]]
    essential_genes = int((gene_label_table[LABEL_COL] == 1).sum())
    nonessential_genes = int((gene_label_table[LABEL_COL] == 0).sum())
    essential_rows = int((df[LABEL_COL] == 1).sum())
    nonessential_rows = int((df[LABEL_COL] == 0).sum())

    print(f"  unique genes                 : {len(gene_label_table)}")
    print(f"  essential genes / rows       : {essential_genes} / {essential_rows}")
    print(f"  non-essential genes / rows   : {nonessential_genes} / {nonessential_rows}")

    return {
        "raw_rows_loaded": len(df),
        "raw_unique_genes": df[GENE_COL].nunique(dropna=True),
        "raw_unique_cell_lines": df[CELL_LINE_ID_COL].nunique(dropna=True),
        "duplicate_gene_cellline_keys": duplicate_keys,
        "rows_missing_gene": missing_gene,
        "rows_missing_cell_line_id": missing_cell_line,
        "rows_missing_label": missing_label,
        "invalid_label_values_count": invalid_label_count,
        "genes_with_inconsistent_label": len(inconsistent_genes),
        "blank_gene_id_count": blank_gene_count,
        "inconsistent_gene_formatting_count": inconsistent_gene_formatting_count,
        "essential_rows": essential_rows,
        "nonessential_rows": nonessential_rows,
        "essential_genes": essential_genes,
        "nonessential_genes": nonessential_genes,
        "essential_gene_pct": essential_genes / len(gene_label_table) * 100.0,
        "nonessential_gene_pct": nonessential_genes / len(gene_label_table) * 100.0,
    }


# --------------------------------------------------------------------------- #
# Step 5.2 - Feature contract (validate/coerce); model matrices come later
# --------------------------------------------------------------------------- #
def define_feature_contract(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """Validate, coerce, and range-check the feature contract.

    These diagnostics run on the FULL input table, before any train/test
    split, to catch data-quality problems in the Phase 4 output itself. They
    are not learned parameters and must never be reused to choose imputation
    values, thresholds, or models -- those must come from training folds only
    in later stages. X/y/groups are deliberately NOT built here: see
    ``assemble_model_matrices``, which only runs after infinities have also
    been neutralized (Step 5.3), so the model matrices are never built from a
    stale, partially-cleaned copy of the table.
    """
    print("\nDefining and validating the feature contract...")

    require_columns(df, EXPECTED_FEATURE_COLS, TRAINING_INPUT_PATH.name)
    feature_cols = list(EXPECTED_FEATURE_COLS)

    df, coercion_missing = coerce_numeric_feature_columns(df, feature_cols, "training dataset")

    summary = df[feature_cols].describe().T[["min", "max", "mean"]]
    print("\nFeature range summary (min / mean / max) -- whole-dataset input QC only:")
    print(summary.to_string())

    fraction_violations = {}
    for col in FRACTION_FEATURE_COLS:
        out_of_range = df[col].dropna()
        out_of_range = out_of_range[(out_of_range < -FRACTION_TOLERANCE) | (out_of_range > 1 + FRACTION_TOLERANCE)]
        fraction_violations[col] = int(len(out_of_range))

    count_violations = {}
    for col in COUNT_FEATURE_COLS:
        negative = df[col].dropna()
        negative = negative[negative < 0]
        count_violations[col] = int(len(negative))

    print(f"  fraction features outside [0, 1]: {fraction_violations}")
    print(f"  count features below 0          : {count_violations}")

    if any(fraction_violations.values()):
        raise ValueError(f"Fraction feature(s) contain values outside [0, 1]: {fraction_violations}")
    if any(count_violations.values()):
        raise ValueError(f"Count feature(s) contain negative values: {count_violations}")

    duplicate_feature_names = len(feature_cols) != len(set(feature_cols))
    if duplicate_feature_names:
        raise ValueError(f"Duplicate feature names in feature list: {feature_cols}")

    print(f"  selected feature count: {len(feature_cols)}")
    print(f"  selected features     : {feature_cols}")

    return df, feature_cols, coercion_missing


def assemble_model_matrices(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Build X, y, groups, and an identifier frame from the fully-cleaned table.

    Must be called only after both ``coerce_numeric_feature_columns`` and
    ``inspect_missing_and_infinite`` have run on ``df``, so X never silently
    contains un-coerced strings or un-neutralized infinities.

    Not called in Stage 1's ``main()``: from Stage 2 onward this is called
    separately on the train-gene and test-gene row subsets, once the split
    below exists, rather than once on the whole table.
    """
    id_frame = df[[GENE_COL, CELL_LINE_ID_COL]].reset_index(drop=True)
    X = df[feature_cols].reset_index(drop=True)
    y = df[LABEL_COL].astype(int).reset_index(drop=True)
    groups = df[GENE_COL].reset_index(drop=True)

    if not (len(X) == len(y) == len(groups) == len(id_frame)):
        raise ValueError("Identifier, feature, target, and group lengths do not match.")

    print("\nAssembled model matrices from the fully-cleaned table:")
    print(f"  X / y / groups shape  : {X.shape} / {y.shape} / {groups.shape}")

    return X, y, groups, id_frame


# --------------------------------------------------------------------------- #
# Step 5.3 - Inspect missing and infinite feature values
# --------------------------------------------------------------------------- #
def inspect_missing_and_infinite(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    """Report and neutralize infinite feature values; summarize missingness.

    Mutates ``df`` in place (no full-table copy) to bound memory use. This is
    whole-dataset input QC, run before any split; it neutralizes infinities
    (which imputers cannot handle) but performs no imputation itself -- that
    is deferred to per-fold pipelines in later stages.
    """
    print("\nInspecting missing and infinite feature values...")

    report_rows = []
    total_infinite = 0

    for col in feature_cols:
        dtype_before = str(df[col].dtype)
        numeric = df[col].to_numpy(dtype=float)
        infinite_mask = np.isinf(numeric)
        infinite_count = int(infinite_mask.sum())
        total_infinite += infinite_count

        if infinite_count:
            df.loc[infinite_mask, col] = np.nan

        missing_count = int(df[col].isna().sum())
        missing_pct = missing_count / len(df) * 100.0 if len(df) else 0.0

        report_rows.append(
            {
                "feature": col,
                "dtype": dtype_before,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "infinite_count_neutralized": infinite_count,
            }
        )

    report = pd.DataFrame(report_rows)
    print(report.to_string(index=False))

    # Check whether missingness in std_log2fc_across_replicates tracks
    # single-replicate observations, as expected (std of one value is
    # undefined).
    std_col = "std_log2fc_across_replicates"
    std_missing_single_replicate_pct = np.nan
    if std_col in df.columns and "replicate_count" in df.columns:
        std_missing_mask = df[std_col].isna()
        std_missing_count = int(std_missing_mask.sum())
        if std_missing_count:
            single_replicate_share = (df.loc[std_missing_mask, "replicate_count"] == 1).mean()
            std_missing_single_replicate_pct = float(single_replicate_share * 100.0)
            print(
                f"\n  '{std_col}' missing in {std_missing_count} rows; "
                f"{std_missing_single_replicate_pct:.2f}% of those have replicate_count == 1 "
                "(std of a single replicate is undefined, as expected)."
            )

    summary = {
        "features_with_missing": int((report["missing_count"] > 0).sum()),
        "features_with_infinite_values": int((report["infinite_count_neutralized"] > 0).sum()),
        "total_infinite_values_neutralized": total_infinite,
        "std_log2fc_missing_count": int(df[std_col].isna().sum()) if std_col in df.columns else 0,
        "std_log2fc_missing_pct": (
            float(df[std_col].isna().mean() * 100.0) if std_col in df.columns else 0.0
        ),
        "std_log2fc_missing_single_replicate_pct": std_missing_single_replicate_pct,
    }

    return df, report, summary


# --------------------------------------------------------------------------- #
# Step 5.4 - Split by gene and freeze the assignments
# --------------------------------------------------------------------------- #
def build_gene_label_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per gene with its (already-validated, consistent) label.

    This is the single, current, authoritative source of gene labels. Loaded
    frozen-split files below never override it -- only its gene membership
    lists are reused.
    """
    gene_label_table = (
        df.drop_duplicates(subset=[GENE_COL])[[GENE_COL, LABEL_COL]]
        .sort_values(GENE_COL)
        .reset_index(drop=True)
    )
    return gene_label_table


def create_grouped_train_test_split(gene_label_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split unique genes into train/test, stratified on the gene-level label.

    Splitting is done on the deduplicated gene table (one row per gene), so
    this is already a gene-level assignment: every row for a given gene will
    later be mapped to whichever side its gene was assigned to. Stratifying
    on the label keeps both classes represented in each partition.
    """
    train_genes, test_genes = train_test_split(
        gene_label_table,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=gene_label_table[LABEL_COL],
    )
    train_genes = train_genes.sort_values(GENE_COL).reset_index(drop=True)
    test_genes = test_genes.sort_values(GENE_COL).reset_index(drop=True)
    return train_genes, test_genes


def load_or_create_split(gene_label_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Reuse a previously frozen gene split unless it is absent, invalid, or a new split is forced.

    Gene *membership* in each partition is frozen to disk and reused as-is
    for reproducibility. Hart *labels* are never trusted from the saved
    file: they are always re-derived from the current ``gene_label_table``
    (built fresh from today's Phase 4 output). If a label has drifted since
    the split was frozen, this is treated as a stop-and-investigate error,
    not something to silently patch over -- letting a later run quietly
    redefine held-out test labels while keeping the same split would make
    results from different runs incomparable. A deliberate label change
    should be handled by explicitly setting FORCE_NEW_SPLIT = True, which
    establishes a new evaluation version (a fresh gene split) rather than
    mutating the existing one in place.

    FORCE_NEW_SPLIT = True always draws a fresh split and takes effect even
    if the existing frozen files are partial or otherwise inconsistent.
    """
    train_exists = TRAIN_GENES_PATH.exists()
    test_exists = TEST_GENES_PATH.exists()

    if FORCE_NEW_SPLIT:
        if train_exists or test_exists:
            print(
                "\nFORCE_NEW_SPLIT is True; discarding any existing frozen split file(s) "
                "and drawing a new one."
            )
        else:
            print("\nFORCE_NEW_SPLIT is True; no existing frozen split found. Drawing a new one.")
        train_genes, test_genes = create_grouped_train_test_split(gene_label_table)
        return train_genes, test_genes, "created_new"

    if train_exists != test_exists:
        raise FileNotFoundError(
            "Found only one of the two frozen split files "
            f"(train exists={train_exists}, test exists={test_exists}). "
            "Restore the missing file, delete the remaining one, or set FORCE_NEW_SPLIT = True "
            "to deliberately draw a new split."
        )

    if train_exists and test_exists:
        print(
            f"\nFound existing frozen split; reusing gene membership from "
            f"{TRAIN_GENES_PATH.name} / {TEST_GENES_PATH.name}."
        )
        train_raw = read_csv_with_clean_headers(TRAIN_GENES_PATH)
        test_raw = read_csv_with_clean_headers(TEST_GENES_PATH)
        # LABEL_COL is required, not optional: without it we cannot check for
        # label drift, and would otherwise silently skip the safeguard above.
        require_columns(train_raw, [GENE_COL, LABEL_COL], TRAIN_GENES_PATH.name)
        require_columns(test_raw, [GENE_COL, LABEL_COL], TEST_GENES_PATH.name)

        train_dupes = int(train_raw[GENE_COL].duplicated().sum())
        test_dupes = int(test_raw[GENE_COL].duplicated().sum())
        if train_dupes or test_dupes:
            raise ValueError(
                f"Frozen split files contain duplicate genes (train={train_dupes}, test={test_dupes}); "
                "the split file(s) may be corrupted. Fix or delete them before rerunning."
            )

        train_gene_set = set(train_raw[GENE_COL])
        test_gene_set = set(test_raw[GENE_COL])
        cross_overlap = train_gene_set & test_gene_set
        if cross_overlap:
            raise ValueError(
                f"Frozen split files overlap on {len(cross_overlap)} gene(s); "
                f"first examples: {sorted(cross_overlap)[:10]}. The split file(s) may be corrupted."
            )

        current_genes = set(gene_label_table[GENE_COL])
        frozen_genes = train_gene_set | test_gene_set
        genes_no_longer_present = frozen_genes - current_genes
        genes_not_yet_assigned = current_genes - frozen_genes

        if genes_no_longer_present:
            raise ValueError(
                "The frozen gene split references genes that are no longer in the "
                f"current training dataset (first examples: {sorted(genes_no_longer_present)[:10]}). "
                "Inspect the Phase 4 output before deciding whether to set FORCE_NEW_SPLIT = True."
            )
        if genes_not_yet_assigned:
            raise ValueError(
                f"{len(genes_not_yet_assigned)} gene(s) in the current training dataset are not "
                "covered by the frozen split (first examples: "
                f"{sorted(genes_not_yet_assigned)[:10]}). The Phase 4 output has changed since the "
                "split was frozen; inspect before deciding whether to set FORCE_NEW_SPLIT = True."
            )

        # Labels are cross-checked against the frozen files but never taken
        # from them: the current, authoritative table is gene_label_table.
        # LABEL_COL is required above precisely so this check can never be
        # silently skipped for lacking the column to compare against.
        label_lookup = gene_label_table.set_index(GENE_COL)[LABEL_COL]

        prior = pd.concat(
            [train_raw[[GENE_COL, LABEL_COL]], test_raw[[GENE_COL, LABEL_COL]]],
            ignore_index=True,
        ).set_index(GENE_COL)[LABEL_COL]
        current_for_prior = prior.index.to_series().map(label_lookup)
        drift_mask = prior.to_numpy() != current_for_prior.to_numpy()
        label_drift_count = int(drift_mask.sum())
        if label_drift_count:
            drifted = pd.DataFrame(
                {
                    GENE_COL: prior.index[drift_mask],
                    "label_when_frozen": prior.to_numpy()[drift_mask],
                    "current_label": current_for_prior.to_numpy()[drift_mask],
                }
            )
            raise ValueError(
                f"{label_drift_count} gene(s) have a different Hart label now than when this "
                "split was frozen. Continuing would let a later run silently redefine held-out "
                "test labels while keeping the same split, making results across runs "
                "incomparable. Investigate the label change first; first examples:\n"
                f"{drifted.head(10).to_string(index=False)}\n"
                "If the change is legitimate, deliberately establish a new evaluation version "
                "by setting FORCE_NEW_SPLIT = True (this draws a fresh gene split) rather than "
                "silently patching the existing one."
            )

        train_genes = pd.DataFrame({GENE_COL: sorted(train_gene_set)})
        test_genes = pd.DataFrame({GENE_COL: sorted(test_gene_set)})
        train_genes[LABEL_COL] = train_genes[GENE_COL].map(label_lookup).astype(int)
        test_genes[LABEL_COL] = test_genes[GENE_COL].map(label_lookup).astype(int)

        return train_genes, test_genes, "loaded_existing_frozen_files"

    print("\nNo usable frozen split found; creating a new gene-level train/test split.")
    train_genes, test_genes = create_grouped_train_test_split(gene_label_table)
    return train_genes, test_genes, "created_new"


def validate_split(
    train_genes: pd.DataFrame,
    test_genes: pd.DataFrame,
    df: pd.DataFrame,
) -> dict[str, int]:
    """Assert zero gene overlap and report split composition."""
    print("\nValidating gene-level train/test split...")

    train_gene_set = set(train_genes[GENE_COL])
    test_gene_set = set(test_genes[GENE_COL])
    overlap = train_gene_set & test_gene_set

    train_essential = int((train_genes[LABEL_COL] == 1).sum())
    train_nonessential = int((train_genes[LABEL_COL] == 0).sum())
    test_essential = int((test_genes[LABEL_COL] == 1).sum())
    test_nonessential = int((test_genes[LABEL_COL] == 0).sum())

    train_rows = int(df[GENE_COL].isin(train_gene_set).sum())
    test_rows = int(df[GENE_COL].isin(test_gene_set).sum())

    print(f"  train genes  : {len(train_gene_set)} (essential={train_essential}, non-essential={train_nonessential})")
    print(f"  test genes   : {len(test_gene_set)} (essential={test_essential}, non-essential={test_nonessential})")
    print(f"  train rows   : {train_rows}")
    print(f"  test rows    : {test_rows}")
    print(f"  gene overlap : {len(overlap)}")

    if overlap:
        raise ValueError(
            f"Train/test gene overlap detected ({len(overlap)} gene(s)); "
            f"first examples: {sorted(overlap)[:10]}"
        )
    if train_essential == 0 or train_nonessential == 0:
        raise ValueError("Training genes do not contain both classes.")
    if test_essential == 0 or test_nonessential == 0:
        raise ValueError("Test genes do not contain both classes.")
    if train_rows + test_rows != len(df):
        raise ValueError(
            "Train + test row counts do not add up to the full dataset "
            f"({train_rows} + {test_rows} != {len(df)}); some genes may be unassigned."
        )

    return {
        "train_gene_count": len(train_gene_set),
        "test_gene_count": len(test_gene_set),
        "train_essential_genes": train_essential,
        "train_nonessential_genes": train_nonessential,
        "test_essential_genes": test_essential,
        "test_nonessential_genes": test_nonessential,
        "train_test_gene_overlap_count": len(overlap),
        "train_rows": train_rows,
        "test_rows": test_rows,
    }


def save_split_assignments(train_genes: pd.DataFrame, test_genes: pd.DataFrame) -> None:
    """Persist the frozen train/test gene assignments (gene membership + current label)."""
    PHASE5_DIR.mkdir(parents=True, exist_ok=True)
    train_genes[[GENE_COL, LABEL_COL]].to_csv(TRAIN_GENES_PATH, index=False)
    test_genes[[GENE_COL, LABEL_COL]].to_csv(TEST_GENES_PATH, index=False)
    print(f"\nSaved frozen split (gene membership + current labels):\n  {TRAIN_GENES_PATH}\n  {TEST_GENES_PATH}")


# --------------------------------------------------------------------------- #
# QC and run metadata
# --------------------------------------------------------------------------- #
def build_qc_report(
    validation_qc: dict[str, int | float],
    feature_cols: list[str],
    coercion_missing: int,
    missingness_summary: dict[str, int | float],
    split_qc: dict[str, int],
    split_source: str,
) -> pd.DataFrame:
    """Assemble the Stage 1 (Steps 5.0-5.4) QC audit trail."""
    metrics: dict[str, object] = {
        "input_path": str(TRAINING_INPUT_PATH),
        **validation_qc,
        "selected_feature_count": len(feature_cols),
        "selected_features": ";".join(feature_cols),
        "values_newly_missing_from_numeric_coercion": coercion_missing,
        **missingness_summary,
        "test_fraction_config": TEST_FRACTION,
        "random_state_config": RANDOM_STATE,
        "n_cv_folds_config_reserved": N_CV_FOLDS,
        "classification_threshold_config_reserved": CLASSIFICATION_THRESHOLD,
        "split_source": split_source,
        **split_qc,
        "stage_completed": "5.0-5.4: paths, load, validate, feature contract, missingness inspection, gene-grouped split",
    }

    qc_df = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    return qc_df


def save_run_metadata(feature_cols: list[str]) -> None:
    """Save a small metadata file recording the exact run configuration."""
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_fraction": TEST_FRACTION,
        "n_cv_folds_reserved": N_CV_FOLDS,
        "classification_threshold_reserved": CLASSIFICATION_THRESHOLD,
        "gene_col": GENE_COL,
        "cell_line_col": CELL_LINE_ID_COL,
        "label_col": LABEL_COL,
        "feature_cols": feature_cols,
        "input_path": str(TRAINING_INPUT_PATH),
        "phase5_output_dir": str(PHASE5_DIR),
        "library_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "script_stage": "5.0-5.4",
    }
    with open(RUN_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved run metadata: {RUN_METADATA_PATH}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=== Phase 5: Modeling & Evaluation (Stage 1: Steps 5.0-5.4) ===")
    print(f"Project folder      : {PROJECT_ROOT}")
    print(f"Phase 4 input folder: {PHASE4_DIR}")
    print(f"Phase 5 output folder: {PHASE5_DIR}\n")

    check_required_files([TRAINING_INPUT_PATH])
    PHASE5_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_dataset()
    validation_qc = validate_training_dataset(df)

    df, feature_cols, coercion_missing = define_feature_contract(df)
    df, missingness_report, missingness_summary = inspect_missing_and_infinite(df, feature_cols)
    missingness_report.to_csv(MISSINGNESS_REPORT_PATH, index=False)
    print(f"\nSaved feature missingness report: {MISSINGNESS_REPORT_PATH}")

    # Model matrices (X/y/groups/id_frame) are deliberately NOT built here.
    # ``assemble_model_matrices`` is reused in Stage 2, called separately on
    # the train and test row subsets once the split below exists -- building
    # a whole-table copy now would be wasted memory and blur the boundary
    # between "cleaned data" and "data assigned to a partition".

    gene_label_table = build_gene_label_table(df)
    train_genes, test_genes, split_source = load_or_create_split(gene_label_table)
    split_qc = validate_split(train_genes, test_genes, df)

    # Save the frozen split. On reuse this is a no-op rewrite of identical
    # content (label drift is rejected with a hard error above, before this
    # point, rather than silently patched here).
    save_split_assignments(train_genes, test_genes)

    qc_df = build_qc_report(
        validation_qc=validation_qc,
        feature_cols=feature_cols,
        coercion_missing=coercion_missing,
        missingness_summary=missingness_summary,
        split_qc=split_qc,
        split_source=split_source,
    )
    qc_df.to_csv(QC_OUTPUT_PATH, index=False)
    print(f"\nSaved Stage 1 QC report: {QC_OUTPUT_PATH}")

    save_run_metadata(feature_cols)

    print("\nStage 1 (Steps 5.0-5.4) complete. No model has been trained or evaluated.")
    print("Reminder: do not consult test-gene features, labels, or scores when choosing a model later.")


if __name__ == "__main__":
    main()
