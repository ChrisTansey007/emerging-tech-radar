# Centralized Configuration for the Innovation Prediction System
from datetime import datetime, timedelta

# --- General System Configuration ---
general_config = {
    'log_level': 'INFO',  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    'main_log_file': 'logs/innovation_system_main.log', # Central log for main process
    'api_max_retries': 3,
    'api_retry_delay_seconds': 5,
    'random_seed': 42, # For reproducibility in stochastic processes
}

# --- Data Collection Configurations ---
patent_config = {
    'api_key_env_var': 'USPTO_API_KEY', # Placeholder for environment variable name
    'base_url': "https://developer.uspto.gov/ibd-api/v1", # Example
    'collection_frequency_hours': 24 * 7, # Weekly
    'tech_categories_queries': {
        'AI': '(("artificial intelligence" OR "machine learning" OR "deep learning") AND (algorithm OR model OR neural OR network))',
        'Blockchain': '(blockchain OR "distributed ledger" OR cryptocurrency OR "smart contract")',
        'QuantumComputing': '("quantum comput*" OR qubit OR "quantum algorithm")',
        'RenewableEnergy': '("solar power" OR "wind energy" OR "battery storage" OR "renewable fuel")',
        'AutonomousVehicles': '("self-driving car" OR "autonomous vehicle" OR lidar OR "adas")',
        'Biotechnology': '("gene editing" OR crispr OR immunotherapy OR "synthetic biology")',
        'ARVR': '("augmented reality" OR "virtual reality" OR "mixed reality" OR metaverse)',
        'Robotics': '(robotics OR "autonomous robot" OR cobot OR "humanoid robot")',
        '5G6G': '("5G technology" OR "6G wireless" OR "next generation network")',
        'CybersecurityTech': '("cybersecurity" OR "threat detection" OR "zero trust" OR "encryption technology")',
    },
    'lookback_days_dynamic': 30, # For regular runs, how far back to query
    'bulk_data_start_year': 2000, # For initial bulk data collection
    'data_quality_min_threshold': 0.80 # Min proportion of usable records in a batch
}

funding_config = {
    'crunchbase_api_key_env_var': 'CRUNCHBASE_API_KEY',
    'crunchbase_api_base_url': "https://api.crunchbase.com/api/v4",
    'categories_identifiers': [ # Mix of keywords and potential (illustrative) UUIDs
        "artificial intelligence", "blockchain", "quantum computing",
        "biotechnology", "renewable energy", "autonomous vehicles", "augmented reality",
        "robotics", "cybersecurity", "5g"
        # Example UUID: "c9d0a8ac-BF3d-4117-99a9-7880c89ca3ec" # AI
    ],
    'min_funding_amount_usd': 100000, # Minimum deal size to consider
    'collection_interval_hours': 24 * 7, # Weekly
    'lookback_days_announced_after': 30, # For regular runs
    'validation_rules': {
        'required_fields': ['funding_round_uuid', 'company_uuid', 'amount_usd', 'date', 'stage'],
        'amount_range_usd': (1000, 20 * 1e9)  # Min $1K, Max $20B
    }
}

research_config = {
    'arxiv_client_page_size': 200,
    'arxiv_client_delay_seconds': 3.1,
    'arxiv_client_num_retries': 3,
    'pubmed_api_key_env_var': 'PUBMED_API_KEY', # Optional NCBI API key
    'pubmed_base_url': "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    'arxiv_queries': [
        'cat:cs.AI OR cat:cs.LG', 'cat:quant-ph', 'ti:"CRISPR" OR abs:"gene editing"', 'cat:q-bio.BM',
        'cat:cs.RO', 'cat:cs.CV', 'cat:cs.CL', 'cat:cs.NE', 'cat:stat.ML',
        'cat:cs.AR AND (gpu OR accelerator OR hardware)', # Hardware for AI
        'cat:econ.GN AND (innovation OR "technological change")' # Economic aspects
    ],
    'pubmed_search_terms': [
        '("artificial intelligence" OR "machine learning") AND (drug discovery OR diagnosis OR "medical imaging")',
        'gene therapy AND (clinical trial OR efficacy OR "vector design")',
        'CRISPR Cas9 applications', 'immunotherapy advancements AND cancer',
        '("renewable energy" OR solar OR wind) AND ("storage" OR "grid integration")',
        '"quantum computing" AND (algorithms OR "drug development" OR "materials science")',
        '"autonomous vehicles" AND (safety OR "sensor fusion" OR "navigation system")',
        '"augmented reality" AND (healthcare OR industrial OR training)',
        'robotics AND (surgical OR logistics OR manufacturing)',
        '"cybersecurity" AND ("threat intelligence" OR "data privacy" OR "network security")'
    ],
    'collection_frequency_hours': 24, # Daily
    'lookback_days_dynamic': 30, # For regular runs
    'min_abstract_length_chars': 100, # For filtering papers
    'author_disambiguation_enabled': False # Placeholder
}

# --- Feature Engineering Configuration ---
feature_config = {
    'patent_feature_weights': { # For potential composite index creation
        'filing_rate_12m': 0.3,
        'citation_velocity_avg': 0.3,
        'tech_diversity_shannon': 0.2,
        'international_filing_ratio': 0.1,
        'avg_inventors_count': 0.1
    },
    'funding_feature_weights': {
        'funding_deals_velocity_3m': 0.3,
        'funding_amount_velocity_3m_usd_norm': 0.2, # Assume amount is normalized for weighting
        'avg_round_size_usd_norm': 0.2, # Assume amount is normalized
        'late_stage_ratio': 0.2,
        'funding_amount_gini': 0.1
    },
    'research_feature_weights': {
        'publication_rate_3m': 0.4,
        'avg_citation_external_norm': 0.3, # Assume citations are normalized
        'research_category_diversity_shannon': 0.2,
        'avg_authors_per_paper': 0.1
    },
    'normalization_method_for_indices': 'min_max', # Method for normalizing features before index creation
    'imputation_strategy_for_training': 'median' # For main model training data prep
}

# --- Model Development Configuration ---
model_config = {
    'target_variable_name': 'y_growth_6m_actual', # Example, define your target
    'sector_id_column_name': 'sector_id',
    'random_state_global': general_config['random_seed'],
    'model_blueprints': { # Define which models to try
        'random_forest': {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, None], 'min_samples_leaf': [1, 3, 5]},
        'gradient_boosting': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5]}
    },
    'ensemble_weights_prediction': {'random_forest': 0.6, 'gradient_boosting': 0.4},
    'time_series_cv_splits': 5,
    'min_samples_for_cv_split': 10, # Min samples per series in a CV training set
    'min_samples_per_sector_train': 20, # Min samples for a sector to be trained
    'prediction_horizons_months': [6, 12, 18, 24], # For which future periods to forecast
    'save_trained_models_path': 'trained_models/', # Directory to save models
    'scaler_type': 'StandardScaler', # 'StandardScaler' or 'MinMaxScaler'
}

# --- Prediction Generation Configuration ---
prediction_config = {
    'default_prediction_horizon_months': 12, # If a single forecast point is needed
    'emergence_indicators_weights': {
        # Features here are assumed to be pre-calculated and named, possibly normalized.
        # Example: 'patent_filing_rate_yoy_change_norm' (Year-over-Year change, normalized)
        'patent_filing_rate_change_norm': 0.3,
        'funding_velocity_change_norm': 0.25,
        'research_publication_growth_norm': 0.25,
        'tech_diversity_increase_norm': 0.2
    },
    'emergence_score_threshold_percentile': 75, # Top 25% of scores considered emerging
    'investment_ranking_criteria_weights': {
        'forecasted_growth_rate': 0.35,
        'prediction_quality_score': 0.15,
        'emergence_score_norm': 0.20,
        'market_size_potential_usd_norm': 0.20,
        'risk_metric_inverse_norm': 0.10
    },
    'investment_min_score_threshold': 0.60, # Min attractiveness score to flag
    'prediction_confidence_intervals_std_dev_factor': 0.1, # For CI estimation fallback
}

# --- Monitoring Configuration ---
monitoring_config = {
    'log_file': 'logs/system_monitor.log',
    'db_connection_string_env_var': 'MONITOR_DB_CONNECTION_STRING', # Placeholder
    'pipeline_health_checks': {
        'patents_data_pipeline': {'max_staleness_days': 2, 'expected_status': 'SUCCESS'},
        'funding_data_pipeline': {'max_staleness_days': 7, 'expected_status': 'SUCCESS'},
        'research_data_pipeline': {'max_staleness_days': 3, 'expected_status': 'SUCCESS'}
    },
    'data_drift_monitoring_threshold_pct_change': 0.20, # For mean change
    'model_performance_thresholds': { # Mirrored from model_config for clarity in monitoring context
        'mae_max': 0.20,
        'direction_accuracy_min': 0.60
    },
    'retraining_suggestion_reason_performance': "Performance degradation on recent data.",
    'retraining_suggestion_reason_drift': "Significant data drift detected in key features.",
    'auto_retraining_enabled': False # Feature flag
}

# --- Uncertainty Handling Configuration ---
uncertainty_config = {
    'confidence_threshold_labels': {'high': 0.75, 'medium': 0.55, 'low': 0.0}, # From prediction_config
    'min_data_completeness_target': 0.70, # Min required completeness for a source (0-1)
    'data_completeness_penalty_factor': 0.20, # How much to penalize for incompleteness
    'min_research_coverage_target': 0.50, # Min required research coverage (0-1)
    'research_coverage_penalty_factor': 0.15,
    'conflict_penalty_factor': 0.25, # Penalty for conflicting signals
    'custom_factor_default_impact': -0.05, # Default impact for unquantified custom factors
    'low_confidence_disclaimer': "Significant uncertainties exist; interpret with extreme caution and seek expert consultation.",
    'medium_confidence_disclaimer': "Some uncertainties identified; consider these in your assessment and monitor developments.",
    'high_confidence_disclaimer': "While confidence is relatively high, predictions are inherently uncertain and subject to change."
}

# Ensure consistency for shared threshold labels
uncertainty_config['confidence_threshold_labels'] = prediction_config['confidence_thresholds_labels']

# Final check: ensure all necessary sub-configs from classes are covered.
# The classes will need to be updated to take their specific config dict as an argument.

```
