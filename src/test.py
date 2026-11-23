from datetime import datetime, timezone
import os
from pathlib import Path


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



#print(f"DATA_ROOT: {DATA_ROOT}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
#print(f"PROCESSED_DIR: {PROCESSED_DIR}")
#print(f"PHASE4_DIR: {PHASE4_DIR}")
#print(f"PHASE5_DIR: {PHASE5_DIR}")
#print(f"TRAINING_INPUT_PATH: {TRAINING_INPUT_PATH}")
#print(f"TRAIN_GENES_PATH: {TRAIN_GENES_PATH}")
#print(f"TEST_GENES_PATH: {TEST_GENES_PATH}")
#print(f"MISSINGNESS_REPORT_PATH: {MISSINGNESS_REPORT_PATH}")
#print(f"QC_OUTPUT_PATH: {QC_OUTPUT_PATH}")
#print(f"RUN_METADATA_PATH: {RUN_METADATA_PATH}")



#configured = os.environ.get("CRISPR_DATA_ROOT")

#if configured:
#    print(Path(configured).expanduser().resolve())

#else:
#    print('Project Root is: ',PROJECT_ROOT / "data")