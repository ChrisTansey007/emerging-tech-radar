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
- \`tests/\`: Contains unit and integration tests (e.g., using Pytest).
- \`uncertainty_handling/\`: Modules for managing and communicating prediction uncertainty.
- \`utils/\`: Placeholder for utility functions.

## Setup

1.  **Create a virtual environment (recommended):**
    \`\`\`bash
    python -m venv venv
    source venv/bin/activate  # On Windows use \`venv\Scripts\activate\`
    \`\`\`

2.  **Install dependencies:**
    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`
    *   **NLTK Resources:** The Natural Language Toolkit (`nltk`), a dependency, may require additional resources (like 'punkt' for tokenization and 'stopwords' for stop word lists). While the system attempts to download these automatically on first use if missing (you might see console messages during this one-time download), you can also download them manually via a Python interpreter:
        \`\`\`python
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        \`\`\`

3.  **Set Up API Keys (Optional for Mock Data):**
    - This system is designed to use API keys for live data collection, managed via a `.env` file at the project root (e.g., `EmergingTechRadar/.env` if your project is in a directory named `EmergingTechRadar`).
    - Create a file named `.env` in the **project root directory**.
    - Add your API keys to this `.env` file. Example format:
      \`\`\`env
      # .env - Place this file in the project root directory
      USPTO_API_KEY="YOUR_USPTO_API_KEY_PLACEHOLDER"
      CRUNCHBASE_API_KEY="YOUR_CRUNCHBASE_API_KEY_PLACEHOLDER"
      PUBMED_API_KEY="YOUR_PUBMED_API_KEY_PLACEHOLDER"
      \`\`\`
    - **Important:** The `.env` file is listed in the project's `.gitignore` and should *not* be committed to version control if it contains real secrets.
    - The `python-dotenv` library (installed via `requirements.txt`) automatically loads these variables when the application starts.
    - If API keys are not provided in `.env` or are left as placeholders, data collectors requiring them will use the default placeholder values defined in `innovation_system/config/settings.py`. This means they may not fetch live data or may be rate-limited. Currently, only the arXiv data collection is attempted live without requiring a dedicated API key configured through `.env` (as the `arxiv` library can be used without authentication for basic queries).

## Running the System

The main conceptual demonstration script is `main/run.py`. It can be run with default settings or customized using command-line arguments.
Refer to the "Data Handling and Persistence" section below for details on how data sources (live vs. mock) are currently managed.

**Note:** For full live data collection across all sources (patents, funding, comprehensive research), the relevant API keys need to be configured in the `.env` file, and the data collection calls in `main/run.py` (currently commented out) would need to be activated. The system primarily uses mock data for patents and funding in its current demonstration state.

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

### Data Handling and Persistence

The system manages data for its demonstration runs as follows:

*   **Live Data Collection:**
    *   **Research Papers (arXiv):** The `main/run.py` script attempts to fetch live data for research papers directly from arXiv based on the specified sectors (mapped to arXiv categories) and the chosen date range.
*   **Mock Data Generation:**
    *   **Patents (USPTO) & Funding (Crunchbase):** For patent and funding data, the system currently generates mock data. This mock data is designed to reflect the structure of real data (e.g., patent data mimics USPTO's PEDS API structure). The underlying collector classes (`PatentDataCollector`, `FundingDataCollector`) are set up to use API keys from the `.env` file, but the main script does not yet call them for live collection in the demo (their instantiation lines are commented out).
*   **Data Persistence:**
    *   All primary data used by the system—whether live-collected (arXiv) or mock-generated (patents, funding)—is saved into Parquet files within the `data/raw/` directory (e.g., `research_papers.parquet`, `patents.parquet`, `funding.parquet`).
        *   A separate SQLite database for monitoring pipeline status (`monitoring.sqlite`) is stored in the `data/` directory.
        *   The data source Parquet files (`data/raw/*.parquet`) and the monitoring database (`data/*.sqlite`) are configured to be ignored by Git (see `.gitignore` and `data/.gitignore`). The `data/raw/.gitkeep` file ensures the raw data directory structure is maintained in the repository.
*   **Loading Persisted Data:**
    *   On subsequent runs, if these Parquet files exist in `data/raw/`, the system will load data directly from them by default. This significantly speeds up startup and allows for consistent reruns with the same dataset.
*   **Overriding Persistence (`--force-collect`):**
    *   The `--force-collect` CLI flag, when used, bypasses loading from existing Parquet files.
    *   It will force the system to:
        *   Re-collect live data from arXiv.
        *   Re-generate mock data for patents and funding.
    *   The newly collected/generated data will then overwrite the existing Parquet files in `data/raw/`.
    *   This mechanism is useful for refreshing the data or testing the collection/generation processes.
