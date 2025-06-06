import pytest
import pandas as pd
import numpy as np # Added for the new test
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import os

# Import necessary classes from the innovation_system
from innovation_system.data_collection import collectors
from innovation_system.feature_engineering import engineer
from innovation_system.model_development import predictor as predictor_module # For new test
from innovation_system.prediction import generator as prediction_generator_module # For new test
from innovation_system.config import settings

# Try to import BaseEstimator, provide a dummy if not available
try:
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import GridSearchCV # For predictor's train method mock
except ImportError:
    class BaseEstimator:
        def get_params(self, deep=True): return {}
        def set_params(self, **params): return self
        def fit(self, X, y=None): self.is_fitted_ = True; return self
        def predict(self, X): return np.zeros(X.shape[0])

    # Dummy GridSearchCV if sklearn is not available
    class GridSearchCV:
        def __init__(self, estimator, param_grid, cv=None, scoring=None, refit=True):
            self.estimator = estimator
            self.param_grid = param_grid
            self.best_estimator_ = estimator # Simplistic mock
            self.best_score_ = 0.0 # Simplistic mock

        def fit(self, X, y=None):
            # Fit the first set of params for simplicity
            params = {k: v[0] for k, v in self.param_grid.items()}
            self.best_estimator_.set_params(**params).fit(X,y)
            self.is_fitted_ = True
            return self


# --- Mock API Responses for Collectors (Simplified from unit tests) ---
SAMPLE_USPTO_API_RESPONSE = {
    "results": [{
        "patentApplicationNumber": "12345678",
        "inventionTitleText": "Test Patent 1",
        "filingDate": "2023-01-15",
        "patentGrantIdentification": {"grantDate": "2024-01-01"},
        "inventorNameArrayText": [{"inventorNameText": "Inventor A"}],
        "assigneeEntityNameText": "Test Assignee Inc.",
        "uspcClassification": [{"classificationSymbolText": "G06F"}],
        "citation": [{"text": "Cited Patent 1"}]
    }]
}
SAMPLE_CRUNCHBASE_API_RESPONSE = {
    "entities": [{
        "uuid": "round-uuid-1", "properties": {
            "money_raised": {"value_usd": 100000, "currency": "USD"},
            "announced_on": "2023-02-01",
            "funded_organization_identifier": {"uuid": "org-uuid-1", "value": "org-uuid-1", "name": "Innovatech"},
            "investment_type": "seed"
        }
    }], "count": 1
}

class MockArxivResultIntegration:
    def __init__(self, entry_id, title, authors, summary, categories, published, pdf_url, comment=None, doi=None, journal_ref=None, primary_category=None, updated=None):
        self.entry_id = entry_id
        self.title = title
        self.authors = [MockArxivAuthor(name) for name in authors]
        self.summary = summary
        self.categories = categories
        self.published = published
        self.pdf_url = pdf_url
        self.comment = comment
        self.doi = doi
        self.journal_ref = journal_ref
        self.primary_category = primary_category
        self.updated = updated if updated else published

class MockArxivAuthor:
    def __init__(self, name):
        self.name = name

SAMPLE_ARXIV_API_RESULTS = [
    MockArxivResultIntegration(
        "http://arxiv.org/abs/2301.00001v1", "Arxiv Paper 1", ["AuthX"], "Abstract 1", ["cs.AI"],
        datetime.now(timezone.utc) - timedelta(days=5), "http://arxiv.org/pdf/2301.00001v1.pdf"
    )
]
SAMPLE_PUBMED_API_RESPONSE_INTEGRATION = [{
    'paper_id': 'pmid1', 'title': 'PubMed Paper 1', 'authors': ['AuthY'],
    'abstract': 'Abstract 2', 'published_date': (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
    'source': 'PubMed', 'url': 'https://pubmed.ncbi.nlm.nih.gov/pmid1/'
}]


@pytest.fixture
def feature_engineer_instance():
    return engineer.FeatureEngineer(config=settings.feature_config)

@patch('os.makedirs', MagicMock(return_value=None))
@patch('pandas.DataFrame.to_parquet', MagicMock(return_value=None))
def test_data_collection_to_feature_engineering_flow(feature_engineer_instance):
    with patch('innovation_system.data_collection.collectors.CRUNCHBASE_API_KEY', 'mock_cb_key'), \
         patch('innovation_system.data_collection.collectors.PUBMED_API_KEY', 'mock_pubmed_key'):
        patent_collector = collectors.PatentDataCollector()
        funding_collector = collectors.FundingDataCollector()
        research_collector = collectors.ResearchDataCollector()

    collected_patents_df = pd.DataFrame()
    collected_funding_df = pd.DataFrame()
    collected_research_df = pd.DataFrame()

    with patch('requests.get') as mock_requests_get, \
         patch('requests.post') as mock_requests_post, \
         patch('arxiv.Client') as mock_arxiv_client_constructor, \
         patch('time.sleep', MagicMock(return_value=None)):

        mock_uspto_response = MagicMock()
        mock_uspto_response.status_code = 200
        mock_uspto_response.json.return_value = SAMPLE_USPTO_API_RESPONSE

        mock_crunchbase_response = MagicMock()
        mock_crunchbase_response.status_code = 200
        mock_crunchbase_response.json.return_value = SAMPLE_CRUNCHBASE_API_RESPONSE
        mock_requests_post.return_value = mock_crunchbase_response

        mock_arxiv_instance = MagicMock()
        mock_arxiv_instance.results.return_value = iter(SAMPLE_ARXIV_API_RESULTS)
        mock_arxiv_client_constructor.return_value = mock_arxiv_instance

        mock_pubmed_esearch_response = MagicMock()
        mock_pubmed_esearch_response.status_code = 200
        mock_pubmed_esearch_response.json.return_value = {"esearchresult": {"idlist": ["pmid1", "pmid2"]}}

        def requests_get_side_effect(url, **kwargs):
            if "pds.uspto.gov/api/search" in url:
                return mock_uspto_response
            if "eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi" in url:
                return mock_pubmed_esearch_response
            return MagicMock(status_code=404)
        mock_requests_get.side_effect = requests_get_side_effect

        with patch.object(research_collector, '_fetch_pubmed_details', return_value=SAMPLE_PUBMED_API_RESPONSE_INTEGRATION):
            start_date = datetime(2023,1,1)
            end_date = datetime(2023,1,31)
            days_back_research = 30

            collected_patents_df = patent_collector.collect_uspto_patents(start_date, end_date, "G01")
            collected_funding_df = funding_collector.collect_funding_rounds(start_date.strftime('%Y-%m-%d'), ["some_funding_cat_uuid"])

            arxiv_papers_list = research_collector.collect_arxiv_papers(categories=["cs.AI"], days_back=days_back_research)
            pubmed_papers_list = research_collector.collect_pubmed_papers(search_terms=["cancer"], days_back=days_back_research)

            df_arxiv = pd.DataFrame(arxiv_papers_list if arxiv_papers_list else [])
            df_pubmed = pd.DataFrame(pubmed_papers_list if pubmed_papers_list else [])
            if not df_arxiv.empty or not df_pubmed.empty:
                collected_research_df = pd.concat([df_arxiv, df_pubmed], ignore_index=True)
            else:
                collected_research_df = pd.DataFrame()

    assert not collected_patents_df.empty, "Patent collection returned empty DataFrame"
    assert 'patent_id' in collected_patents_df.columns
    assert not collected_funding_df.empty, "Funding collection returned empty DataFrame"
    assert 'company_uuid' in collected_funding_df.columns
    assert not collected_research_df.empty, "Research collection returned empty DataFrame"
    assert 'paper_id' in collected_research_df.columns

    collected_patents_df['filing_date'] = pd.to_datetime(collected_patents_df['filing_date'])
    collected_funding_df['date'] = pd.to_datetime(collected_funding_df['date'])
    if not collected_research_df.empty:
        collected_research_df['published_date'] = pd.to_datetime(collected_research_df['published_date'])

    patent_features = feature_engineer_instance.create_patent_features(collected_patents_df)
    funding_features = feature_engineer_instance.create_funding_features(collected_funding_df)
    research_features = feature_engineer_instance.create_research_features(collected_research_df) if not collected_research_df.empty else pd.DataFrame()

    assert not patent_features.empty, "Patent features are empty"
    assert isinstance(patent_features, pd.DataFrame)
    assert 'filing_rate_3m' in patent_features.columns
    assert not funding_features.empty, "Funding features are empty"
    assert isinstance(funding_features, pd.DataFrame)
    assert 'funding_deals_velocity_3m' in funding_features.columns
    if not collected_research_df.empty:
        assert not research_features.empty, "Research features are empty"
        assert isinstance(research_features, pd.DataFrame)
        assert 'publication_rate_3m' in research_features.columns
    else:
        assert research_features.empty, "Research features should be empty if no research data"


# --- New Test: Features to Prediction Flow ---

class MockEstimatorForIntegration(BaseEstimator):
    def __init__(self, some_param=1):
        self.some_param = some_param
        self.feature_importances_ = []
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_importances_ = np.random.rand(X.shape[1])
            self.feature_names_in_ = X.columns.to_list()
        elif isinstance(X, np.ndarray):
            self.feature_importances_ = np.random.rand(X.shape[1])
            self.feature_names_in_ = [f"feat_{i}" for i in range(X.shape[1])] # Assign default names
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if self.feature_names_in_ is not None and hasattr(X, 'columns'):
            # Ensure X for predict has same features as fit, if feature_names_in_ is set
            # This might fail if X doesn't have all feature_names_in_ (e.g. after lag drop)
            # For integration test, assume X will have the necessary columns for simplicity.
            X_ordered = X[self.feature_names_in_]
        else:
            X_ordered = X
        return np.random.rand(X_ordered.shape[0]) * 0.1

    def get_params(self, deep=True):
        return {"some_param": self.some_param}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

@pytest.fixture
def sample_engineered_features():
    date_rng = pd.date_range(start='2022-01-01', end='2023-01-01', freq='MS')
    n_samples = len(date_rng)

    patent_feats = pd.DataFrame({
        'filing_rate_3m': np.random.rand(n_samples) * 5,
        'tech_diversity_shannon': np.random.rand(n_samples) * 1.5
    }, index=date_rng)

    funding_feats = pd.DataFrame({
        'funding_deals_velocity_3m': np.random.rand(n_samples) * 3,
        'avg_round_size_usd': np.random.rand(n_samples) * 1e6
    }, index=date_rng)

    research_feats = pd.DataFrame({
        'publication_rate_3m': np.random.rand(n_samples) * 10,
        'avg_keyword_count': np.random.rand(n_samples) * 2
    }, index=date_rng)

    targets = pd.DataFrame({
        'target_growth_6m': np.random.rand(n_samples) * 0.2 - 0.05
    }, index=date_rng)

    return patent_feats, funding_feats, research_feats, targets

# Need mock_prediction_config for PredictionGenerator
# Assuming it's available from test_generator.py or conftest.py
# If not, we need to define it here or mock PredictionGenerator's config access.
# For now, let's mock the config access in PredictionGenerator if needed, or define a simple one.
@pytest.fixture
def minimal_mock_prediction_config():
    return {
        'confidence_thresholds': {'high': 0.8, 'medium': 0.6, 'low': 0.4},
        # Add other keys if PredictionGenerator's __init__ or methods directly access them
        # and they are not part of what's being tested by *this* integration test.
        'emergence_indicators': {}, # Empty for this flow if not used
        'emergence_analysis_thresholds': {},
        'emergence_risk_factors_map': {},
        'investment_ranking_criteria': {}
    }


@patch('sklearn.model_selection.GridSearchCV', new_callable=MagicMock) # Use MagicMock for new_callable
@patch('time.sleep', return_value=None)
def test_features_to_prediction_flow(mock_time_sleep, mock_grid_search_cv, sample_engineered_features, minimal_mock_prediction_config):
    patent_feats, funding_feats, research_feats, targets_df = sample_engineered_features

    innovation_predictor = predictor_module.InnovationPredictor(random_state=42)

    X_prepared, y_prepared = innovation_predictor.prepare_training_data(
        patent_feats, funding_feats, research_feats, targets_df
    )

    assert not X_prepared.empty, "X_prepared should not be empty"
    assert not y_prepared.empty, "y_prepared should not be empty"
    assert len(X_prepared) == len(y_prepared)
    assert isinstance(X_prepared.index, pd.DatetimeIndex)

    if not X_prepared.empty:
        X_prepared['sector_label'] = 'general_tech'

    mock_gs_instance = mock_grid_search_cv.return_value # Get the MagicMock instance
    # Fit the estimator to set feature_names_in_
    estimator_for_gs = MockEstimatorForIntegration()
    if not X_prepared.empty:
         # Ensure estimator is "fitted" to set feature_names_in_
        estimator_for_gs.fit(X_prepared.drop(columns=['sector_label']), y_prepared)
    mock_gs_instance.best_estimator_ = estimator_for_gs
    mock_gs_instance.best_score_ = -0.05

    original_blueprints = innovation_predictor.models_blueprints
    innovation_predictor.models_blueprints = { 'mock_rf': MockEstimatorForIntegration() }
    original_get_param_grid = innovation_predictor._get_param_grid
    innovation_predictor._get_param_grid = lambda name: {'some_param': [1]}

    if not X_prepared.empty:
        innovation_predictor.train_sector_models(X_prepared, y_prepared, sectors_column='sector_label')

    innovation_predictor.models_blueprints = original_blueprints
    innovation_predictor._get_param_grid = original_get_param_grid

    if not X_prepared.empty:
        assert 'general_tech' in innovation_predictor.sector_models
        assert 'mock_rf' in innovation_predictor.sector_models['general_tech']
        trained_mock_model = innovation_predictor.sector_models['general_tech']['mock_rf']
        assert isinstance(trained_mock_model, MockEstimatorForIntegration)
        if hasattr(trained_mock_model, 'feature_names_in_') and trained_mock_model.feature_names_in_ is not None:
            expected_feature_names = X_prepared.drop(columns=['sector_label']).columns.tolist()
            assert trained_mock_model.feature_names_in_ == expected_feature_names
    else:
        assert not innovation_predictor.sector_models

    ensemble_weights_for_pg = {'mock_rf': 1.0}
    pg = prediction_generator_module.PredictionGenerator(
        trained_models=innovation_predictor.sector_models,
        ensemble_weights=ensemble_weights_for_pg,
        prediction_cfg=minimal_mock_prediction_config
    )

    current_features_df_for_pg = pd.DataFrame()
    if not X_prepared.empty:
        last_prepared_features_series = X_prepared.drop(columns=['sector_label']).iloc[-1:]
        current_features_df_for_pg = last_prepared_features_series.copy()
        current_features_df_for_pg['sector_label'] = 'general_tech'

    forecast_horizons = [6, 12]

    # Mock the sub-methods of PredictionGenerator for this integration test
    # Use fixed return values for predictability
    fixed_ensemble_pred_val = 0.05
    fixed_individual_preds = {'mock_rf': fixed_ensemble_pred_val} # Aligns with ensemble_weights_for_pg
    fixed_conf_intervals = {'68%': [0.04,0.06], '95%': [0.03,0.07]} # Symmetrical for simplicity
    fixed_quality_score = 0.8

    with patch.object(pg, '_ensemble_predict_single_sector', return_value=(fixed_ensemble_pred_val, fixed_individual_preds)) as mock_pg_ensemble, \
         patch.object(pg, '_calculate_confidence_intervals_bootstrap', return_value=fixed_conf_intervals) as mock_pg_ci, \
         patch.object(pg, '_assess_prediction_quality', return_value=fixed_quality_score) as mock_pg_quality:

        forecasts = pg.generate_sector_forecasts(
            current_features_all_sectors_df=current_features_df_for_pg,
            sector_column_name='sector_label',
            horizons=forecast_horizons
        )

    if not X_prepared.empty and not current_features_df_for_pg.empty:
        assert 'general_tech' in forecasts
        assert len(forecasts['general_tech']) == len(forecast_horizons)
        for h in forecast_horizons:
            horizon_key = f'{h}m_growth_rate'
            assert horizon_key in forecasts['general_tech']
            assert forecasts['general_tech'][horizon_key]['prediction'] == fixed_ensemble_pred_val
            assert forecasts['general_tech'][horizon_key]['confidence_intervals'] == fixed_conf_intervals
            assert forecasts['general_tech'][horizon_key]['quality_score'] == fixed_quality_score

        assert mock_pg_ensemble.call_count == len(forecast_horizons)
        assert mock_pg_ci.call_count == len(forecast_horizons)
        assert mock_pg_quality.call_count == len(forecast_horizons)

    elif X_prepared.empty:
        assert not forecasts, "Forecasts should be empty if X_prepared was empty"
    elif current_features_df_for_pg.empty :
         assert not forecasts, "Forecasts should be empty if current_features_df_for_pg is empty"

```
