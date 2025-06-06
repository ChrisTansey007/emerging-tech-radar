# System Workflow Diagram

This diagram provides a high-level overview of the Innovation Prediction System's architecture and data flow.

```mermaid
graph LR
    subgraph External World
        USPTO[(USPTO API)]
        Crunchbase[(Crunchbase API)]
        ArXivAPI[arXiv API]
    end

    subgraph Innovation Prediction System Module (`innovation_system/`)
        A[Data Collection (`collectors.py`)]

        subgraph Data Input Path
            ArXivAPI -- Live Data --> A
            USPTO -- Currently Mocked by `main/run.py` --> A_Patents_Mock(Mock Patent Data Generation)
            Crunchbase -- Currently Mocked by `main/run.py` --> A_Funding_Mock(Mock Funding Data Generation)
            A_Patents_Mock --> P[Data Persistence Layer <br> (`data/raw/*.parquet`)]
            A_Funding_Mock --> P
        end

        A -- Processed arXiv Data --> P

        P -- Loaded Data --> B[Feature Engineering (`engineer.py`)]
        B -- Engineered Features --> C[Model Training/Loading <br> (`predictor.py` - Mock Training currently)]
        C -- Trained Models --> D[Prediction Generation <br> (`generator.py`)]
        D -- Predictions/Insights --> E[System Outputs <br> (Console Summaries, Future Reports)]

        subgraph Monitoring & Logging
            M_DB[Monitoring Database <br> (`innovation_system/data/monitoring.sqlite`)]
            M_Log[Log File <br> (`system_monitor.log` at project root)]
            A -.-> M_Log # Log from collectors
            B -.-> M_Log # Log from feature eng.
            C -.-> M_Log # Log from predictor
            D -.-> M_Log # Log from generator
            F[Overall Run Orchestration <br> (`main/run.py`)] -- Updates status --> M_DB
            F -.-> M_Log # Log from main run
        end
    end

    style P fill:#lightgrey,stroke:#333,stroke-width:2px
    style M_DB fill:#lightyellow,stroke:#333,stroke-width:2px
    style M_Log fill:#lightyellow,stroke:#333,stroke-width:2px
    style E fill:#lightblue,stroke:#333,stroke-width:2px
```
