import pandas as pd
import os

def normalize_to_rpm(input_path, output_path):
    """
    Normalizes raw CRISPR read counts to Reads Per Million (RPM).
     Formula: (count / total_reads_in_sample) * 1e6
    """
    print(f"Loading raw data from {input_path}...")
    # Load data. First column is sgRNA sequence, set as index.
    df = pd.read_csv(input_path, index_col=0)
    
    print("Normalizing to RPM...")
    # Calculate column sums (total reads per sample)
    col_sums = df.sum()
    
    # Normalize: (df / sum) * 1,000,000
    df_rpm = df.divide(col_sums, axis=1) * 1e6
    
    print(f"Saving normalized data to {output_path}...")
    # Save as parquet for better performance/size if pyarrow is available, 
    # but using CSV for now to stay consistent with raw data.
    # Note: AvanaRawReadcounts is large (~1GB), so RPM file will also be large.
    df_rpm.to_csv(output_path)
    print("Normalization complete.")

if __name__ == "__main__":
    PROJECT_ROOT = "/home/nima/Google Drive/informatics/AI projects/crispr_ai_ml/crispr-ml-hitcalling"
    INPUT_FILE = os.path.join(PROJECT_ROOT, "data_raw/AvanaRawReadcounts.csv")
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data_processed/AvanaRPM.csv")
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    normalize_to_rpm(INPUT_FILE, OUTPUT_FILE)
