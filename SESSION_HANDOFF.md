# CRISPR ML Hitcalling: Session Handoff

## 📅 Session Date: May 14, 2026

## 🚀 Accomplishments
1.  **Project Organization:**
    *   Identified the core project at `/home/nima/Google Drive/informatics/AI projects/crispr_ai_ml/crispr-ml-hitcalling/`.
    *   Transitioned from exploratory notebooks (`.ipynb`) to a modular Python pipeline in `src/`.
2.  **GitHub Integration:**
    *   Initialized a Git repository.
    *   Configured SSH authentication for `nimafakouri`.
    *   Pushed the project to: [https://github.com/nimafakouri/crispr-ml-hitcalling](https://github.com/nimafakouri/crispr-ml-hitcalling)
    *   Created a `.gitignore` to exclude the 991MB raw data file while keeping local access.
3.  **Environment Setup:**
    *   Switched to the **`bioenv`** mamba environment.
    *   Installed missing dependencies: `matplotlib`, `seaborn`, and `pyarrow`.
    *   Updated `GEMINI.md` to document the environment usage.
4.  **Phase 3 Implementation (Started):**
    *   Created `src/03a_normalize_rpm.py` to normalize raw read counts to Reads Per Million (RPM).

---

## 📍 Where We Left Off
We are at the beginning of **Phase 3: Feature Engineering**. 
*   The script `src/03a_normalize_rpm.py` is written but **has not been executed yet**.
*   The raw data (`data_raw/AvanaRawReadcounts.csv`) is ready for processing.

---

## 📋 Next Steps for Next Session

### 1. Execute Normalization (Phase 3a)
Run the newly created script to generate the processed RPM data:
```bash
mamba activate bioenv
python src/03a_normalize_rpm.py
```

### 2. Implement Log2 Fold Change (Phase 3b)
Create `src/03b_compute_log2fc.py`. This script should:
*   Load `data_processed/AvanaRPM.csv`.
*   Identify the **pDNA baseline** columns (referenced in the roadmap).
*   Calculate the log2 fold change: `log2(Sample_RPM + 1) - log2(pDNA_RPM + 1)`.

### 3. Feature Aggregation (Phase 3c)
Create a script to aggregate sgRNA-level fold changes into **Gene-level features** (Mean, Median, Std Dev).

### 4. Label Assembly (Phase 4)
Integrate the **Hart 2014/2017** gold standard gene sets to create the training labels for the ML model.

---

## 🛠 Notes
*   **Data Size:** `AvanaRawReadcounts.csv` is ~1GB. Processing might take a few minutes.
*   **Corrupted File:** The file `CRISPR ML Hitcalling.docx` appeared to be 0 bytes/corrupted; future logic should continue to follow the `Practical Roadmap` in the project root.
