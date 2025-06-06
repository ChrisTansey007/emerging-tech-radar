# Data Schemas

This document outlines the schemas for data files persisted by the Innovation Prediction System and the monitoring database.

## Persisted Data Files (Parquet)

These files are typically located in `innovation_system/data/raw/`.

### `research_papers.parquet`

Stores information about research papers, primarily from arXiv.

- `paper_id`: string (Unique identifier for the paper, e.g., arXiv ID)
- `title`: string (Title of the paper)
- `authors`: list[string] (List of author names)
- `abstract`: string (Abstract/summary of the paper)
- `categories`: list[string] (List of categories, e.g., arXiv subject classifications)
- `published_date`: string (Publication date in ISO format, e.g., "YYYY-MM-DDTHH:MM:SSZ" or "YYYY-MM-DD")
- `pdf_url`: string (URL to the PDF version of the paper)
- `source`: string (Origin of the data, e.g., "arXiv", "arXiv_mock")
- `citation_count`: integer (Number of citations; currently 0 for arXiv data)

### `patents.parquet`

Stores information about patents, currently using mock data based on USPTO PEDS API structure.

- `patent_id`: string (Unique identifier for the patent, e.g., "PG12345678")
  *(Note: `main/run.py` mock generation uses `patent_id`; `PatentDataCollector` might use `patentApplicationNumber` or similar for live data)*
- `title`: string (Title of the invention)
- `filing_date`: string (Filing date in ISO format "YYYY-MM-DD")
- `assignee`: string (Name of the assignee company/entity)
- `inventors`: list[string] (List of inventor names)
- `tech_class`: string or list[string] (Technical classification, e.g., CPC codes)
- `abstract`: string (Abstract of the patent)
  *(Note: `main/run.py` mock generation uses `abstract`; `PatentDataCollector` might use `abstractText`)*
- `citations`: integer (Number of citations; mock value)
- `source`: string (Origin of the data, e.g., "USPTO_mock")
- `sector_label`: string (Sector associated during mock data generation)

### `funding.parquet`

Stores information about funding rounds, currently using mock data.
*Note: The mock data generation for this file in `main/run.py` needs to be updated to reflect this raw funding round schema instead of pre-aggregated features. Currently, `main/run.py` saves a mock DataFrame that already has some pre-calculated features.*

- `company_uuid`: string (Unique identifier for the company; mock UUID)
- `company_name`: string (Name of the company receiving funding)
- `amount_usd`: float (Funding amount in USD)
- `currency`: string (Currency of the funding amount, e.g., "USD")
- `date`: string (Date of the funding round announcement, ISO format "YYYY-MM-DD")
- `stage`: string (Funding stage or investment type, e.g., "Series A", "Seed")
- `source`: string (Origin of the data, e.g., "Crunchbase_mock")
- `sector_label`: string (Sector associated during mock data generation)

## Monitoring Database

Located at `innovation_system/data/monitoring.sqlite`.

### Table: `pipeline_status`

Tracks the status of pipeline runs.

- `pipeline_name`: TEXT (PRIMARY KEY, Name of the pipeline or job, e.g., "main_run")
- `last_run_timestamp`: TEXT (Timestamp of the last status update in ISO format datetime)
- `status`: TEXT (Current status, e.g., "STARTED", "COMPLETED", "FAILED")
- `details`: TEXT (Additional information or error messages related to the run)
```
