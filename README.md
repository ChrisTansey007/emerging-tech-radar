# Emerging Tech Radar

This repository is dedicated to tracking and evaluating emerging technologies. The goal is to provide a comprehensive overview of the latest trends and innovations in the tech industry.

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

### Key Features & Current Status:

*   **Status:** Actively under development.
*   **Core Functionality:**
    *   Automated data collection (currently live for arXiv research papers; mock data for patents and funding, with infrastructure ready for API key integration).
    *   Feature engineering, including statistical methods and basic Natural Language Processing (NLP) on research abstracts (e.g., abstract length, keyword counts).
    *   Conceptual modeling of innovation trends (details within the module).
*   **Data Management:**
    *   API keys are managed using a `.env` file at the project root (see `innovation_system/README.md` for setup). Currently uses placeholders for most keys.
    *   Collected/generated data (arXiv papers, mock patent/funding data) is persisted to Parquet files in the `innovation_system/data/raw/` directory to optimize subsequent runs.
*   **Execution & Control:**
    *   The system is run via a Command-Line Interface (CLI): `python innovation_system/main/run.py`.
    *   Key runtime arguments include specifying target `--sectors`, date ranges (`--start-date`, `--end-date`), prediction `--horizons`, and forcing data re-collection with `--force-collect`.
*   **Monitoring:**
    *   Basic pipeline run status (STARTED, COMPLETED) is tracked in an SQLite database located at `innovation_system/data/monitoring.sqlite`.

For more detailed setup instructions, usage examples, and technical information about this module, please refer to the [Innovation Prediction System README](./innovation_system/README.md).

## Technologies

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

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
