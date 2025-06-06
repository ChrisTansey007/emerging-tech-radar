# Innovation Prediction System

This project implements a system for predicting technological innovation trends.
It collects data from various sources (patents, funding, research papers),
engineers features, trains predictive models, and generates forecasts.

## Project Structure

- \`config/\`: Configuration files for different modules.
- \`data_collection/\`: Modules for collecting data from sources like USPTO, Crunchbase, arXiv, PubMed.
- \`feature_engineering/\`: Modules for creating predictive features from raw data.
- \`main/\`: Main execution script (\`run.py\`).
- \`model_development/\`: Modules for training and validating predictive models.
- \`monitoring/\`: Modules for system monitoring, data drift detection, and model performance evaluation.
- \`prediction/\`: Modules for generating forecasts and identifying emerging technologies.
- \`tests/\`: Unit and integration tests (to be added).
- \`uncertainty_handling/\`: Modules for managing and communicating prediction uncertainty.
- \`utils/\`: Utility functions (if any).

## Setup

1.  **Clone the repository:**
    \`\`\`bash
    git clone <repository_url>
    cd innovation_prediction_system
    \`\`\`

2.  **Create a virtual environment (recommended):**
    \`\`\`bash
    python -m venv venv
    source venv/bin/activate  # On Windows use \`venv\Scripts\activate\`
    \`\`\`

3.  **Install dependencies:**
    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`

4.  **Configure API Keys:**
    - Update API keys in relevant configuration files or via environment variables as needed.
      For example, Crunchbase API key in \`FundingDataCollector\` and potentially a PubMed API key in \`ResearchDataCollector\`.
      These are currently placeholder values in the code (e.g., "YOUR_CRUNCHBASE_KEY").

## Running the System

The main conceptual demonstration script is `main/run.py`. It can be run with default settings or customized using command-line arguments.

**Note:** The `main/run.py` script currently uses placeholder data (mock data generation) for demonstration purposes.
Full functionality (live data collection, actual model training on real data) would require uncommenting and completing the data collection phase,
valid API keys, and potentially more extensive historical data for model training.

### Command-Line Interface (CLI)

The `main/run.py` script supports command-line arguments to customize the analysis parameters:

**Arguments:**

*   `--sectors "SECTOR1,SECTOR2"`
    *   **Description:** Specifies a comma-separated list of technology sectors for the analysis.
    *   **Default:** `"AI,Biotech"`
*   `--start-date YYYY-MM-DD`
    *   **Description:** Sets the start date for data collection (and mock data generation period).
    *   **Default:** 90 days prior to the current date.
*   `--end-date YYYY-MM-DD`
    *   **Description:** Sets the end date for data collection (and mock data generation period).
    *   **Default:** The current date.
*   `--horizons "H1,H2,H3"`
    *   **Description:** Defines a comma-separated list of prediction horizons in months (e.g., for forecasts).
    *   **Default:** `"6,12,24"`
*   `--force-collect`
    *   **Description:** A boolean flag. If specified, it forces the system to regenerate data (currently mock data) and save it, even if existing data files are found in `data/raw/`. Without this flag, the system attempts to load pre-existing data from `data/raw/` to speed up runs.
    *   **Default:** Not set (i.e., False).

**Usage Examples:**

1.  **Run with default settings:**
    ```bash
    python innovation_system/main/run.py
    ```
    *(This will use the default sectors, date range, and horizons for the mock data simulation.)*

2.  **Analyze specific sectors for a defined period and horizons:**
    ```bash
    python innovation_system/main/run.py --sectors "Quantum Computing,Renewable Energy" --start-date 2023-01-01 --end-date 2023-12-31 --horizons "6,18"
    ```

3.  **Analyze default sectors but with custom prediction horizons:**
    ```bash
    python innovation_system/main/run.py --horizons "3,9,15"
    ```

4.  **Analyze a single sector:**
    ```bash
    python innovation_system/main/run.py --sectors "AI"
    ```

5.  **Force data regeneration even if files exist:**
    ```bash
    python innovation_system/main/run.py --force-collect --sectors "AI"
    ```

**Note on Data Handling:**

The script now incorporates a basic data persistence layer. When data is generated (currently mock data, based on CLI parameters), it is saved to Parquet files within the `data/raw/` directory (e.g., `patents.parquet`, `funding.parquet`, `research_papers.parquet`).

On subsequent runs, the system will load data from these files by default, if they exist. This behavior can be overridden using the `--force-collect` flag, which will force regeneration and saving of the data. This feature is intended to speed up development and repeated runs by avoiding redundant data processing.
The `data/raw/` directory contains a `.gitkeep` file to ensure the directory structure is part of the repository, while the `data/.gitignore` file is configured to exclude `*.parquet` files from being committed.
