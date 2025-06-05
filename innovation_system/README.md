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
    - It is recommended to use environment variables to store your API keys for security and flexibility.
      For example, you can set them in your terminal:
      - For Linux/macOS:
        ```bash
        export CRUNCHBASE_API_KEY="YOUR_KEY"
        export PUBMED_API_KEY="YOUR_KEY"
        ```
      - For Windows (Command Prompt):
        ```bash
        set CRUNCHBASE_API_KEY="YOUR_KEY"
        set PUBMED_API_KEY="YOUR_KEY"
        ```
      - For Windows (PowerShell):
        ```powershell
        $env:CRUNCHBASE_API_KEY="YOUR_KEY"
        $env:PUBMED_API_KEY="YOUR_KEY"
        ```
    - The application code can then access these keys using `os.environ.get('API_KEY_NAME')`. For instance, `os.environ.get('CRUNCHBASE_API_KEY')`.
    - Alternatively, if you choose to use configuration files, ensure you update the placeholder values (e.g., "YOUR_CRUNCHBASE_KEY") in the relevant files within the `config/` directory or directly in the collector classes if they are not yet using external configs.

## Running the System

To run the main conceptual demonstration script:

\`\`\`bash
python main/run.py
\`\`\`

**Note:** The \`main/run.py\` script currently uses placeholder data and mock API calls for demonstration purposes.
Full functionality requires valid API keys and potentially more extensive historical data for model training.
The script also has temporary includes of classes and configurations which will be replaced by proper imports once the refactoring is complete.
To connect the system to real data sources, you will need to implement the actual API calls within the respective data collector classes (e.g., in `data_collection/funding_data_collector.py`), replacing the current mock/placeholder logic, in addition to providing valid API keys.
