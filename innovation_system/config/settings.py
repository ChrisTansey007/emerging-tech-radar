from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API Keys loaded from environment variables with defaults
USPTO_API_KEY = os.getenv("USPTO_API_KEY", "USPTO_DEFAULT_PLACEHOLDER")
CRUNCHBASE_API_KEY = os.getenv("CRUNCHBASE_API_KEY", "CRUNCHBASE_DEFAULT_PLACEHOLDER")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "PUBMED_DEFAULT_PLACEHOLDER")


patent_config = {
    'collection_frequency': 'daily',
    'tech_categories': [
        'artificial intelligence', 'blockchain', 'quantum computing',
        'biotechnology', 'renewable energy', 'autonomous vehicles'
    ],
    'lookback_days': 30,
    'data_quality_threshold': 0.85
}

funding_config = {
    'categories': [
        'artificial intelligence', 'machine learning', 'blockchain',
        'fintech', 'biotech', 'clean energy', 'autonomous vehicles'
    ],
    'min_amount_usd': 100000,
    'collection_interval': 'weekly',
    'validation_rules': {
        'required_fields': ['company_uuid', 'amount_usd', 'date'],
        'amount_range': (1000, 10000000000)
    }
}

research_config = {
    'arxiv_categories': [
        'cs.AI', 'cs.LG', 'cs.CV', 'cs.RO', 'q-bio.GN', 'quant-ph'
    ],
    'pubmed_terms': [
        'gene therapy', 'CRISPR', 'immunotherapy', 'personalized medicine', 'mRNA vaccine'
    ],
    'collection_frequency': 'daily',
    'min_abstract_length': 100,
    'author_disambiguation': True
}

feature_config = {
    'patent_weights': {
        'filing_rate': 0.25, 'citation_velocity': 0.20, 'tech_diversity': 0.15,
        'international_ratio': 0.15, 'inventor_network': 0.25
    },
    'funding_weights': {
        'velocity': 0.30, 'round_size': 0.25,
        'stage_distribution': 0.20, 'concentration': 0.25
    },
    'research_weights': {
        'publication_rate': 0.25, 'citation_quality': 0.25,
        'collaboration': 0.20, 'novelty': 0.30
    },
    'normalization_method': 'z_score',
    'missing_value_strategy': 'median_imputation',
    'emerging_tech_keywords': [
        'artificial intelligence', 'machine learning', 'deep learning',
        'quantum computing', 'quantum supremacy',
        'blockchain', 'cryptocurrency', 'smart contract', 'nft',
        'biotechnology', 'gene editing', 'crispr', 'synthetic biology',
        'renewable energy', 'solar power', 'wind power', 'battery storage', ' sostenible', ' जलवायु',
        'nanotechnology', 'carbon nanotubes'
    ],
}

model_config = {
    'prediction_horizons': [6, 12, 18, 24, 36],
    'validation_method': 'time_series_split',
    'ensemble_method': 'weighted_average',
    'performance_threshold': {
        'mae': 0.15, 'direction_accuracy': 0.65
    },
    'retraining_frequency': 'monthly',
    'feature_importance_tracking': True
}

prediction_config = {
    'update_frequency': 'weekly',
    'confidence_thresholds': {
        'high': 0.8, 'medium': 0.6, 'low': 0.4
    },
    'emergence_indicators': {
        'patent_filing_acceleration': 0.3, 'funding_velocity_increase': 0.25,
        'research_publication_growth': 0.25, 'cross_disciplinary_collaboration': 0.1,
        'keyword_novelty_score': 0.1
    },
    'investment_ranking_criteria': {
        'growth_potential': 0.4,
        'confidence_score': 0.3,
        'market_size': 0.2,
        'risk_adjusted_return': 0.1,
        # New additions:
        'high_growth_threshold': 0.10,
        'low_growth_threshold': 0.01,
        'large_market_threshold_usd': 500000000,
        'min_market_size_threshold_usd': 100000000,
        'action_invest_score_thresh': 0.5, # Threshold for "Consider Investment" action score
        'action_invest_growth_thresh': 0.05, # Min growth for "Consider Investment"
        'action_monitor_score_thresh': 0.3, # Threshold for "Monitor Closely" action score
    },
    'emergence_analysis_thresholds': {
        'score_high_confidence': 0.70,
        'score_medium_confidence': 0.40,
        'timeline_fast_threshold': 0.70,
        'timeline_medium_threshold': 0.40
    },
    'emergence_risk_factors_map': {
        'low_funding_signal': "Funding/Commercialization Lag",
        'low_research_signal': "Weakening Research Base",
        'low_patent_signal': "Slowing IP Generation",
        'highly_concentrated_activity': "Dependence on Few Players",
        'nascent_market': "Market Adoption Uncertainty",
        'default_risk': "General market and execution risks"
    }
}

monitoring_config = {
    'log_file': 'system_monitor.log',
    'health_check_frequency': 'daily',
    'data_drift_check_frequency': 'weekly',
    'model_performance_evaluation_frequency': 'monthly',
    'alert_email_recipient': 'admin@example.com',
    'auto_retraining_enabled': True
}

uncertainty_config = {
    'confidence_thresholds': prediction_config['confidence_thresholds'], # Reuse
    'min_data_completeness_for_high_confidence': 0.80,
    'min_paper_coverage_for_high_confidence': 0.60,
    'default_probability_conflicting_signals': 0.5
}
