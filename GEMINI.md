# Project Overview: crispr-ml-hitcalling

This project, named "crispr-ml-hitcalling," is a data science and machine learning endeavor focused on analyzing CRISPR screening data. Its primary goal appears to be the identification of "hits" (e.g., genes or genetic perturbations that significantly impact a phenotype) from high-throughput CRISPR experiments. The project is currently in an exploratory phase, with data loading and initial inspection being performed within Jupyter notebooks.

## Key Technologies and Libraries
-   **Python:** The primary programming language used.
-   **Pandas:** Extensively used for data manipulation and analysis, as evidenced by its import and usage in the exploratory notebook.
-   **Jupyter Notebooks:** Used for interactive data exploration, analysis, and potentially model development.

## Data Structure

The project utilizes raw data stored in the `data_raw/` directory, consisting of three main CSV files:

-   `AvanaRawReadcounts.csv`: Contains raw sgRNA read counts across various screens and pDNA baselines.
-   `AvanaGuideMap.csv`: Provides mapping information between sgRNAs (single guide RNAs) and their target genes.
-   `ScreenSequenceMap.csv`: Contains metadata for replicates, including cell line information, QC status, and pDNA indicators.

Processed data is expected to be stored in `data_processed/`, and analysis results (figures, tables) in `results/`.

## Project Structure

-   `data_processed/`: Intended for storing cleaned and processed data.
-   `data_raw/`: Contains the original, raw CSV data files.
-   `notebooks/`: Houses Jupyter notebooks for data exploration, analysis, and possibly model building.
    -   `01_exploration.ipynb`: An initial notebook for data loading and exploratory analysis.
    -   `01_exploration_notes.txt`: Contains brief notes related to the exploration.
-   `results/`: Stores output from analyses, including:
    -   `figures/`: For generated plots and images.
    -   `tables/`: For tabular data outputs.
-   `src/`: Currently empty, but typically reserved for reusable source code (e.g., utility functions, module definitions) if the project were to grow beyond notebook-only code.

## Building and Running

This project primarily relies on Python and Jupyter notebooks. To run the analysis:

1.  **Environment Setup:** Ensure you have the `bioenv` mamba environment installed. This environment contains the necessary libraries for data science and CRISPR analysis.
    ```bash
    mamba activate bioenv
    ```
2.  Open the Jupyter notebooks (e.g., `notebooks/01_exploration.ipynb`) in a Jupyter environment (Jupyter Lab or Jupyter Notebook).
3.  Execute the cells sequentially to reproduce the data loading and exploration steps.

**TODO:** A `requirements.txt` file is currently missing. It would be beneficial to create one to explicitly list all Python dependencies for easier environment setup.

## Development Conventions

-   **Exploratory Development:** The project seems to follow an exploratory development style, with initial analysis being performed directly within Jupyter notebooks.
-   **Data Organization:** Raw data is kept separate from processed data.
-   **Naming Conventions:** Files and directories follow a clear, descriptive naming convention.
