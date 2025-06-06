# Emerging Tech Radar

This repository is dedicated to tracking and evaluating emerging technologies. The goal is to provide a comprehensive overview of the latest trends and innovations in the tech industry.

## 🚀 Getting Started in 5 Minutes

Follow these steps to get the Innovation Prediction System up and running quickly with mock data:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ChrisTansey007/emerging-tech-radar.git
    cd EmergingTechRadar
    ```

2.  **Create and Activate a Virtual Environment:**
    *   On macOS & Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    (From the project root directory `EmergingTechRadar/`)
    ```bash
    pip install -r innovation_system/requirements.txt
    ```
    *Note: Some libraries like NLTK might require additional data. The system attempts to download these automatically. See the detailed setup in the [Innovation Prediction System README](./innovation_system/README.md) if you encounter issues.*

4.  **API Key Setup (Using Defaults):**
    For this quick start, the system will use default placeholder API keys. Live data will only be fetched from arXiv. To enable other live data sources later, you'll need to create a `.env` file in the project root as described in the main "Setup" section of the [Innovation Prediction System README](./innovation_system/README.md).

5.  **Run a Sample Analysis (from project root):**
    This command will process data for AI and Quantum Computing for a short period, using mock data for patents/funding and attempting live arXiv data.
    ```bash
    python innovation_system/main/run.py --sectors "AI,Quantum Computing" --start-date 2023-01-01 --end-date 2023-01-07 --horizons "6" --force-collect
    ```

6.  **What You'll See:**
    This command processes data for the 'AI' and 'Quantum Computing' sectors for a short period in early 2023, using mock data for patents and funding, and attempting a live fetch for arXiv research papers. The `--force-collect` flag ensures data is (re)generated. You'll see progress messages in the console.

    Afterwards, check the following locations (relative to the project root directory `EmergingTechRadar/`):
    *   **Processed Data Files:**
        *   `innovation_system/data/raw/patents.parquet`
        *   `innovation_system/data/raw/funding.parquet`
        *   `innovation_system/data/raw/research_papers.parquet`
    *   **Monitoring Database:**
        *   `innovation_system/data/monitoring.sqlite`
    *   **Log File:**
        *   `system_monitor.log` (created in the project root directory `EmergingTechRadar/`)

    The Parquet files contain the data used for analysis, the SQLite database records the status of this run, and the log file provides detailed execution information.

## 🐳 Running with Docker (Containerized Deployment)

For a consistent and isolated environment, you can run the Innovation Prediction System using Docker and Docker Compose.

**Prerequisites:**

*   [Docker](https://docs.docker.com/get-docker/) installed and running.
*   [Docker Compose](https://docs.docker.com/compose/install/) installed (usually included with Docker Desktop).

**Setup:**

1.  **API Keys (`.env` file):**
    Ensure you have a `.env` file at the project root, as described in the "Getting Started" section or the detailed `innovation_system` README. The `docker-compose.yml` configuration will pass these environment variables (like `USPTO_API_KEY`, `CRUNCHBASE_API_KEY`, etc.) from your host's `.env` file into the container. If this file is missing or keys are placeholders, the system will use default placeholders internally.

**Running the System:**

1.  **Build and Run with Default Parameters:**
    (From the project root directory `EmergingTechRadar/`)
    ```bash
    docker-compose up --build
    ```
    This command builds the Docker image (if not already built or if changes were made) and starts the `innovation_system_app` service as defined in `docker-compose.yml`. The default command specified in `docker-compose.yml` will be executed (which includes sample CLI arguments like `--sectors "AI,Biotech"` and `--force-collect`).

    To run in detached mode (in the background), add the `-d` flag:
    ```bash
    docker-compose up --build -d
    ```

2.  **Running with Custom CLI Arguments:**
    To run the system with different parameters than the defaults in `docker-compose.yml`, use `docker-compose run`:
    ```bash
    # Example: Analyze only 'Quantum Computing' for a specific short period
    docker-compose run --rm innovation_system_app --sectors "Quantum Computing" --start-date 2023-02-01 --end-date 2023-02-05 --horizons "12" --force-collect
    ```
    The `--rm` flag automatically removes the container when it exits. The arguments listed after `innovation_system_app` will be passed to the `innovation_system/main/run.py` script inside the container, overriding the default `command` in `docker-compose.yml`.

3.  **Stopping the System (if running in detached mode):**
    ```bash
    docker-compose down
    ```

**Data Persistence:**

*   The `docker-compose.yml` configuration includes volume mounts:
    *   `./innovation_system/data` on your host machine is mapped to `/app/innovation_system/data` inside the container. This means any Parquet files or the `monitoring.sqlite` database generated by the application will persist on your host machine in `EmergingTechRadar/innovation_system/data/` even after the container stops.
    *   `./system_monitor.log` on your host is mapped to `/app/system_monitor.log` in the container, so the log file is also directly accessible on your host.

## Table of Contents

- [Introduction](#introduction)
- [Core Module: Innovation Prediction System](#core-module-innovation-prediction-system)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Emerging Tech Radar aims to identify and analyze technologies that are on the horizon, providing insights into their potential impact and adoption. This repository will serve as a central hub for discussions, research, and collaboration on these technologies.

## Core Module: Innovation Prediction System

The `Innovation Prediction System`, located in the `innovation_system/` directory, is the primary engine of the Emerging Tech Radar. It's designed to automate the collection, processing, and analysis of data from various sources (academic research, patents, funding news) to identify and forecast trends in technological innovation. This system aims to provide data-driven insights into what technologies are emerging, their momentum, and potential areas of impact.

A high-level visual overview of this module's workflow, from data collection through to prediction and monitoring, is available in the [System Workflow Diagram](./docs/flow.md).

### Key Features & Current Status:

*   **Status:** Actively under development.
*   **Core Functionality:**
    *   Automated data collection (currently live for arXiv research papers; mock data for patents and funding, with infrastructure ready for API key integration).
    *   Feature engineering, including statistical methods and basic Natural Language Processing (NLP) on research abstracts (e.g., abstract length, keyword counts).
    *   Conceptual modeling of innovation trends (details within the module).
*   **Data Management:**
    *   API keys are managed using a `.env` file at the project root (see `innovation_system/README.md` for setup). Currently uses placeholders for most keys.
    *   Collected/generated data (arXiv papers, mock patent/funding data) is persisted to Parquet files in the `innovation_system/data/raw/` directory to optimize subsequent runs. Detailed information about these data schemas is available in the [Data Schemas Document](./docs/data_schema.md).
*   **Execution & Control:**
    *   The system is run via a Command-Line Interface (CLI): `python innovation_system/main/run.py`.
    *   Key runtime arguments include specifying target `--sectors`, date ranges (`--start-date`, `--end-date`), prediction `--horizons`, and forcing data re-collection with `--force-collect`.
*   **Monitoring:**
    *   Basic pipeline run status (STARTED, COMPLETED) is tracked in an SQLite database located at `innovation_system/data/monitoring.sqlite`.

For more detailed setup instructions, usage examples, and technical information about this module, please refer to the [Innovation Prediction System README](./innovation_system/README.md).

## Continuous Integration (CI)

This project uses [GitHub Actions](https://github.com/features/actions) for Continuous Integration. On every push or pull request to key branches (e.g., `main`, `development`), a CI pipeline automatically performs the following checks:

*   **Linting:** Code style and quality are checked using [Flake8](https://flake8.pycqa.org/).
*   **Unit Tests:** Automated tests are run using [Pytest](https://pytest.org/).
*   **Code Coverage:** Test coverage is measured using [Coverage.py](https://coverage.readthedocs.io/) and the build will fail if coverage drops below a defined threshold (currently set as an example at 60%).

This helps to maintain code quality, ensure tests pass, and catch issues early in the development process.

[![Python CI Status](https://github.com/ChrisTansey007/emerging-tech-radar/actions/workflows/ci.yml/badge.svg)](https://github.com/ChrisTansey007/emerging-tech-radar/actions/workflows/ci.yml)

## Technologies
This section is currently under development and will be populated with information about tracked technologies soon.

This section will list the emerging technologies being tracked. Each technology will have its own page with detailed information, including:

- Overview
- Key Features
- Use Cases
- Potential Impact
- Current Status
- Resources

## Contributing

Contributions are welcome! If you have a technology you'd like to see added to the radar, or if you have insights to share, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
