# Innovation Prediction System

This project implements a system for predicting technological innovation trends. It collects data from various sources (e.g., patents, funding, research papers), engineers features from this data, trains predictive models, and generates forecasts and other insights.

## Project Structure

The system is organized into several Python packages, each responsible for a specific part of the pipeline:

-   `config/`: Contains the central configuration file (`settings.py`) for all modules.
    -   Key file: `settings.py`
-   `data_collection/`: Modules for collecting raw data from various external sources.
    -   Key classes: `PatentDataCollector`, `FundingDataCollector`, `ResearchDataCollector` (in `collectors.py`)
-   `feature_engineering/`: Modules for transforming raw data into predictive features.
    -   Key class: `FeatureEngineer` (in `engineer.py`)
-   `model_development/`: Modules for training, validating, and managing predictive models.
    -   Key class: `InnovationPredictor` (in `predictor.py`)
-   `prediction/`: Modules for generating forecasts, identifying emerging technologies, and creating investment opportunities.
    -   Key class: `PredictionGenerator` (in `generator.py`)
-   `monitoring/`: Modules for system monitoring, including data pipeline health, data drift detection, and model performance evaluation.
    -   Key class: `SystemMonitor` (in `monitor.py`)
-   `uncertainty_handling/`: Modules for assessing and communicating the uncertainty associated with predictions.
    -   Key class: `UncertaintyManager` (in `manager.py`)
-   `main/`: Contains the main execution script (`run.py`) to orchestrate the system's pipeline.
    -   Key file: `run.py`
-   `utils/`: (Currently empty) Intended for utility functions shared across modules.
-   `tests/`: (To be developed) Intended for unit and integration tests.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory_name>/innovation_system
    # Ensure you are in the 'innovation_system' subdirectory for subsequent commands
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # For Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys (Crucial for Real Data Collection):**
    For the system to collect real data from external sources (patents, funding, research), you must provide valid API keys for services such as Crunchbase, USPTO (or other patent offices), and PubMed (NCBI E-utilities).

    **It is strongly recommended to use environment variables to store your API keys.** This is more secure and flexible than hardcoding them.

    **Examples:**

    *   **For Linux/macOS (bash/zsh):**
        Open your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`) and add:
        ```bash
        export CRUNCHBASE_API_KEY="YOUR_CRUNCHBASE_KEY"
        export USPTO_API_KEY="YOUR_USPTO_KEY_OR_AUTH_DETAILS"
        export PUBMED_API_KEY="YOUR_NCBI_PUBMED_KEY" # Optional, but recommended for higher rate limits
        ```
        Then, source the file (e.g., `source ~/.bashrc`) or open a new terminal.

    *   **For Windows (Command Prompt):**
        ```bash
        set CRUNCHBASE_API_KEY="YOUR_CRUNCHBASE_KEY"
        set USPTO_API_KEY="YOUR_USPTO_KEY_OR_AUTH_DETAILS"
        set PUBMED_API_KEY="YOUR_NCBI_PUBMED_KEY"
        ```
        (These need to be set each time a new command prompt is opened, or set system-wide via System Properties.)

    *   **For Windows (PowerShell):**
        ```powershell
        $env:CRUNCHBASE_API_KEY="YOUR_CRUNCHBASE_KEY"
        $env:USPTO_API_KEY="YOUR_USPTO_KEY_OR_AUTH_DETAILS"
        $env:PUBMED_API_KEY="YOUR_NCBI_PUBMED_KEY"
        ```
        (To make these persistent in PowerShell, add them to your PowerShell profile script.)

    **Accessing Keys in Python:**
    The data collector classes (and `config/settings.py` where API key environment variable names are defined) expect these environment variables to be set. Conceptually, they are accessed in Python like this (example from `collectors.py` if it were to directly use an API key):
    ```python
    import os
    api_key = os.environ.get('CRUNCHBASE_API_KEY')
    if not api_key:
        print("Crunchbase API key not found in environment variables!")
        # Handle missing key (e.g., use mock data, raise error)
    ```

    **Obtaining Keys:**
    Refer to the documentation of the respective API providers (Crunchbase, USPTO, NCBI E-utilities for PubMed) to obtain your API keys. Some services may require registration and adherence to usage policies.

    **Impact of Missing Keys:**
    If environment variables for API keys are not set or are invalid, the data collectors will likely fail to retrieve real data. They might fall back to using placeholder/mock data if designed to do so for demonstration, or they may raise errors. The current `main/run.py` script uses mock data and does not perform live API calls.

## Configuration

Detailed configurations for various aspects of the system are centralized in the `innovation_system/config/settings.py` file. This includes:

-   **Data Collection:** API endpoints (though actual keys are via env vars), specific search queries for different technologies, data source lookback periods, collection intervals.
-   **Feature Engineering:** Weights for combining features into composite indices, normalization methods, imputation strategies.
-   **Model Development:** Model types, hyperparameter grids for tuning, cross-validation settings, prediction horizons.
-   **Prediction Generation:** Thresholds for identifying emerging technologies, criteria weights for ranking investment opportunities.
-   **Monitoring:** Log file paths, health check frequencies, data drift thresholds, model performance alert levels.
-   **Uncertainty Handling:** Confidence level definitions, penalty factors for various uncertainty sources.

Users can inspect and modify `config/settings.py` to tailor the system's behavior, data sources, and analysis parameters for different scenarios or research questions. However, ensure that any changes to feature names or structures are consistently reflected in the relevant class implementations.

## Running the System

To run the main conceptual demonstration script:

1.  Navigate to the `innovation_system` directory (if you cloned `repository_url` into `repository_directory_name`, this would be `cd repository_directory_name/innovation_system`).
2.  Execute the main script:
    ```bash
    python main/run.py
    ```
    Alternatively, if you are in the project root directory (parent of `innovation_system`), you can run it as a module:
    ```bash
    python -m innovation_system.main.run
    ```
    Running from the `innovation_system` directory is generally simpler for this project structure.

The `main/run.py` script now orchestrates an integrated pipeline using the newly implemented classes. It demonstrates:
-   Instantiation of core components (collectors, engineer, predictor, generator, monitor, uncertainty manager).
-   Conceptual data processing using mock historical data for various technology sectors.
-   Feature engineering based on this mock data.
-   Model training for different sectors using the generated features.
-   Generation of predictive forecasts.
-   Identification of emerging technologies and investment opportunities based on these forecasts and mock current data.
-   Conceptual system monitoring checks (e.g., data drift baseline establishment).
-   An example of uncertainty assessment for a generated forecast.

**Note:** For the system to work with real data and provide meaningful insights, valid API keys must be configured as described in the "Setup" section, and the data collection components would need to be invoked with appropriate parameters. The current `run.py` primarily uses mock data for end-to-end pipeline demonstration.

## Output

The system generates several log files during its operation:

-   `innovation_main_run.log`: Main log for the execution of `main/run.py`, capturing the overall flow and key outputs from different pipeline stages. (Located in the directory where `run.py` is executed).
-   `system_monitor.log`: Logs from the `SystemMonitor` class, including pipeline health checks, data drift monitoring, and model performance evaluations. (Default location also in the execution directory, can be configured).
-   `uncertainty_manager.log`: Logs from the `UncertaintyManager` class, detailing the process of assessing prediction confidence. (Default location also in the execution directory).

**Console Output:**
When running `main/run.py`, you can expect console output indicating:
-   Start and end of the conceptual run.
-   Status messages from different phases (data collection mock, feature engineering mock, model training, prediction, etc.).
-   Sample outputs for generated forecasts, emerging technologies, and investment opportunities.
-   Summaries of model training and validation if verbose.
-   Log messages from various components (which are also directed to their respective files).

## Limitations and Known Issues

-   **Mock Data Focus:** The `main/run.py` script, in its current state, primarily uses mock/generated data to demonstrate the full pipeline flow. While the data collectors are implemented, `run.py` does not actively call them for live data collection. Full functionality with real-time data requires correctly configured API keys and uncommenting/adding live data collection calls in `run.py`.
-   **API Collector Robustness:** Some data collector functionalities (e.g., specific USPTO query mechanisms, comprehensive pagination for all APIs, advanced error handling for API limits) are conceptual. They may need further refinement, more sophisticated error handling, and thorough testing against live APIs for production use. Parsing logic might also need adjustments based on actual API response structures.
-   **Feature Engineering for Emergence:** The features used for the `identify_emerging_technologies` method in `run.py` are currently proxied from the mock data. A production system would require careful engineering of features that truly reflect technological emergence (e.g., rates of change, accelerations, novelty scores).
-   **Error Handling:** Error handling in some parts of the newly implemented code might be basic. Production deployment would benefit from more comprehensive error management, retries, and fault tolerance.
-   **Scalability:** The current implementation is designed as a conceptual framework. Scaling it to handle very large datasets or high-frequency updates would require further optimization, potentially different data storage solutions, and possibly distributed computing components.
-   **Testing:** Comprehensive unit and integration tests are yet to be developed.

This system provides a foundational framework for technology forecasting. Further development would focus on robust data integration, advanced modeling techniques, and rigorous validation.

```
