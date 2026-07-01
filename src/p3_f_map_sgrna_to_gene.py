"""
Phase 3, Step III.f - Map sgRNA-level log2FC values to target genes.

This script attaches target-gene information from AvanaGuideMap.csv to the
sgRNA-level log2 fold-change table produced in Step III.e.

Inputs:
    data/processed/phase3_inputs/screen_log2fc_vs_pdna.csv
    data/raw/AvanaGuideMap.csv

Outputs:
    data/processed/phase3_inputs/screen_log2fc_vs_pdna_gene_mapped.csv
    data/processed/phase3_inputs/unmapped_guides.csv

This step does not recompute raw counts, RPM, pDNA baselines, or log2FC.
It keeps one row per sgRNA and only adds gene annotation.

Run only after reviewing:
    python src/p3_f_map_sgrna_to_gene.py
"""

from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path("/Volumes/nimas_usb/project_data_repo/crispr_ai_ml")
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PHASE3_DIR = PROCESSED_DIR / "phase3_inputs"

GUIDE_MAP_PATH = RAW_DIR / "AvanaGuideMap.csv"
LOGFC_PATH = PHASE3_DIR / "screen_log2fc_vs_pdna.csv"
OUTPUT_PATH = PHASE3_DIR / "screen_log2fc_vs_pdna_gene_mapped.csv"
UNMAPPED_OUTPUT_PATH = PHASE3_DIR / "unmapped_guides.csv"


# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
# These names come from inspecting the actual file headers.
LOGFC_SGRNA_COL = "sgRNA_seq"
GUIDE_SGRNA_COL = "sgRNA"
GUIDE_GENE_COL = "Gene"
USED_BY_CHRONOS_COL = "UsedByChronos"
DROP_REASON_COL = "DropReason"
OUTPUT_GENE_COL = "gene"

CHUNK_SIZE = 5_000
LOW_MAPPING_RATE_WARNING = 90.0


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


def normalize_sgrna_ids(values: pd.Series) -> pd.Series:
    """Normalize sgRNA IDs for matching without changing the biological value."""
    return values.astype("string").str.strip().str.upper()


def truthy_mask(values: pd.Series) -> pd.Series:
    """Return True for common boolean encodings of yes/true/1."""
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def empty_text_mask(values: pd.Series) -> pd.Series:
    """Return True for missing or empty text values."""
    normalized = values.astype("string").str.strip()
    return normalized.isna() | (normalized == "")


def clean_gene_values(values: pd.Series) -> pd.Series:
    """
    Convert Avana guide-map gene values to gene symbols.

    AvanaGuideMap.csv stores values like 'SHOC2 (8036)' in the Gene column.
    For downstream gene-level aggregation and Hart-set joins, keep the symbol
    as the primary gene label.
    """
    gene = values.astype("string").str.strip()
    return gene.str.replace(r"\s+\([^)]+\)$", "", regex=True)


def print_columns(label: str, columns: list[str], max_columns: int = 12) -> None:
    """Print a compact column summary without flooding the console."""
    preview = columns[:max_columns]
    suffix = " ..." if len(columns) > max_columns else ""
    print(f"{label} columns ({len(columns)}): {preview}{suffix}")


# --------------------------------------------------------------------------- #
# Load and inspect input data
# --------------------------------------------------------------------------- #
def load_guide_map() -> pd.DataFrame:
    """Load AvanaGuideMap.csv and print basic structure."""
    guide_map = pd.read_csv(GUIDE_MAP_PATH)
    require_columns(guide_map, [GUIDE_SGRNA_COL, GUIDE_GENE_COL], GUIDE_MAP_PATH.name)

    print(f"AvanaGuideMap shape: {guide_map.shape}")
    print_columns("AvanaGuideMap", guide_map.columns.tolist())
    print("AvanaGuideMap first sgRNA values:")
    print(guide_map[GUIDE_SGRNA_COL].head().to_list())
    print("AvanaGuideMap first gene values:")
    print(guide_map[GUIDE_GENE_COL].head().to_list())

    return guide_map


def load_logfc_header() -> list[str]:
    """Read the log2FC file header without loading the full wide matrix."""
    logfc_columns = pd.read_csv(LOGFC_PATH, nrows=0).columns.tolist()
    print_columns("screen_log2fc_vs_pdna", logfc_columns)

    if LOGFC_SGRNA_COL not in logfc_columns:
        raise ValueError(f"{LOGFC_PATH.name} is missing required column: {LOGFC_SGRNA_COL}")

    print(f"Using sgRNA column in logFC table: {LOGFC_SGRNA_COL}")
    print(f"Using sgRNA column in guide map: {GUIDE_SGRNA_COL}")
    print(f"Using gene column in guide map: {GUIDE_GENE_COL}")
    return logfc_columns


# --------------------------------------------------------------------------- #
# Prepare guide map
# --------------------------------------------------------------------------- #
def filter_guide_map_to_curated_rows(guide_map: pd.DataFrame) -> pd.DataFrame:
    """Filter AvanaGuideMap to rows DepMap/Chronos used and did not drop."""
    filtered = guide_map.copy()

    print("\nGuide map curation filters:")
    print(f"  rows before filtering              : {len(filtered)}")

    if USED_BY_CHRONOS_COL in filtered.columns:
        filtered = filtered[truthy_mask(filtered[USED_BY_CHRONOS_COL])].copy()
        print(f"  rows after UsedByChronos filter    : {len(filtered)}")
    else:
        print(f"  {USED_BY_CHRONOS_COL} column absent; skipping filter")

    if DROP_REASON_COL in filtered.columns:
        filtered = filtered[empty_text_mask(filtered[DROP_REASON_COL])].copy()
        print(f"  rows after DropReason filter       : {len(filtered)}")
    else:
        print(f"  {DROP_REASON_COL} column absent; skipping filter")

    if filtered.empty:
        raise ValueError("Guide map has zero rows after curation filters.")

    return filtered


def prepare_guide_map(guide_map: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce AvanaGuideMap.csv to one row per sgRNA with a clean gene symbol.

    If duplicated sgRNA IDs map to multiple different genes, stop instead of
    allowing the merge to multiply rows or create ambiguous gene assignments.
    """
    guide_map = filter_guide_map_to_curated_rows(guide_map)

    guide_lookup = guide_map[[GUIDE_SGRNA_COL, GUIDE_GENE_COL]].copy()
    guide_lookup[LOGFC_SGRNA_COL] = normalize_sgrna_ids(guide_lookup[GUIDE_SGRNA_COL])
    guide_lookup[OUTPUT_GENE_COL] = clean_gene_values(guide_lookup[GUIDE_GENE_COL])
    guide_lookup = guide_lookup[[LOGFC_SGRNA_COL, OUTPUT_GENE_COL]]
    guide_lookup = guide_lookup.dropna(subset=[LOGFC_SGRNA_COL])
    guide_lookup = guide_lookup[guide_lookup[LOGFC_SGRNA_COL] != ""]

    guide_rows = len(guide_lookup)
    unique_guides = guide_lookup[LOGFC_SGRNA_COL].nunique(dropna=True)
    duplicate_count = guide_rows - unique_guides

    print("\nGuide map duplicate checks after filtering:")
    print(f"  guide map rows           : {guide_rows}")
    print(f"  unique sgRNAs            : {unique_guides}")
    print(f"  duplicated sgRNA rows    : {duplicate_count}")

    conflict_counts = guide_lookup.groupby(LOGFC_SGRNA_COL)[OUTPUT_GENE_COL].nunique(dropna=True)
    conflicting_guides = conflict_counts[conflict_counts > 1]
    print(f"  conflicting sgRNAs       : {len(conflicting_guides)}")
    if not conflicting_guides.empty:
        examples = conflicting_guides.head(10).index.to_list()
        raise ValueError(
            "Some sgRNAs map to more than one gene in AvanaGuideMap.csv. "
            f"Inspect these before merging. First examples: {examples}"
        )

    # Exact duplicate sgRNA/gene mappings are safe to collapse. Prefer rows
    # with a gene value if duplicate guide rows differ only by missingness.
    guide_lookup = guide_lookup.assign(_gene_missing=guide_lookup[OUTPUT_GENE_COL].isna())
    guide_lookup = guide_lookup.sort_values("_gene_missing")
    guide_lookup = guide_lookup.drop_duplicates(subset=[LOGFC_SGRNA_COL])
    guide_lookup = guide_lookup.drop(columns="_gene_missing")
    return guide_lookup


# --------------------------------------------------------------------------- #
# Merge and save
# --------------------------------------------------------------------------- #
def map_logfc_to_genes(logfc_columns: list[str], guide_lookup: pd.DataFrame) -> None:
    """Left-join the sgRNA-level log2FC table to the cleaned guide map."""
    temp_output_path = OUTPUT_PATH.with_suffix(".csv.tmp")
    temp_unmapped_path = UNMAPPED_OUTPUT_PATH.with_suffix(".csv.tmp")

    replicate_cols = [col for col in logfc_columns if col != LOGFC_SGRNA_COL]
    final_columns = [LOGFC_SGRNA_COL, OUTPUT_GENE_COL] + replicate_cols

    rows_before = 0
    rows_after = 0
    mapped_count = 0
    unmapped_count = 0
    first_output_chunk = True
    first_unmapped_chunk = True

    print("\nMerging guide map with sgRNA logFC table...")
    for chunk in pd.read_csv(LOGFC_PATH, chunksize=CHUNK_SIZE):
        require_columns(chunk, [LOGFC_SGRNA_COL], LOGFC_PATH.name)

        chunk = chunk.copy()
        chunk[LOGFC_SGRNA_COL] = normalize_sgrna_ids(chunk[LOGFC_SGRNA_COL])

        before_chunk_rows = len(chunk)
        mapped_chunk = chunk.merge(guide_lookup, how="left", on=LOGFC_SGRNA_COL)
        after_chunk_rows = len(mapped_chunk)

        if after_chunk_rows != before_chunk_rows:
            raise ValueError(
                "Unexpected row-count change after merge. "
                f"Before: {before_chunk_rows}; after: {after_chunk_rows}. "
                "This usually means duplicated sgRNA IDs remain in the guide map."
            )

        mapped_chunk = mapped_chunk[final_columns]
        unmapped_chunk = mapped_chunk[mapped_chunk[OUTPUT_GENE_COL].isna()].copy()

        rows_before += before_chunk_rows
        rows_after += after_chunk_rows
        unmapped_count += len(unmapped_chunk)
        mapped_count += after_chunk_rows - len(unmapped_chunk)

        mapped_chunk.to_csv(
            temp_output_path,
            index=False,
            mode="w" if first_output_chunk else "a",
            header=first_output_chunk,
        )
        first_output_chunk = False

        unmapped_chunk.to_csv(
            temp_unmapped_path,
            index=False,
            mode="w" if first_unmapped_chunk else "a",
            header=first_unmapped_chunk,
        )
        first_unmapped_chunk = False

        print(f"  processed {rows_before:,} sgRNAs")

    temp_output_path.replace(OUTPUT_PATH)
    temp_unmapped_path.replace(UNMAPPED_OUTPUT_PATH)

    mapping_rate = (mapped_count / rows_after * 100.0) if rows_after else 0.0

    print("\nMerge QC:")
    print(f"  rows before merge : {rows_before}")
    print(f"  rows after merge  : {rows_after}")
    print(f"  mapped guides     : {mapped_count}")
    print(f"  unmapped guides   : {unmapped_count}")
    print(f"  mapping rate      : {mapping_rate:.2f}%")

    if rows_before != rows_after:
        raise ValueError("Rows before and after merge do not match.")

    if mapping_rate < LOW_MAPPING_RATE_WARNING:
        print(
            f"  WARNING: mapping rate is below {LOW_MAPPING_RATE_WARNING:.1f}%. "
            "Check whether the sgRNA ID columns use the same format."
        )

    print(f"\nSaved mapped output to: {OUTPUT_PATH}")
    print(f"Saved unmapped guides to: {UNMAPPED_OUTPUT_PATH}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=== Phase 3 / Step III.f: Map sgRNAs to genes ===")
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Phase 3 folder: {PHASE3_DIR}\n")

    check_required_files([LOGFC_PATH, GUIDE_MAP_PATH])

    guide_map = load_guide_map()
    logfc_columns = load_logfc_header()
    guide_lookup = prepare_guide_map(guide_map)
    map_logfc_to_genes(logfc_columns, guide_lookup)

    print("\nStep III.f complete.")


if __name__ == "__main__":
    main()
