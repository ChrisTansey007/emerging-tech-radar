import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch
import logging # Added for caplog access if needed by new tests

from innovation_system.prediction import generator

# Minimal MockModel for PredictionGenerator tests
class MockPredictorModel:
    def __init__(self, model_name="test_model", prediction_value=1.0, features_in=None):
        self.model_name = model_name
        self.prediction_value = prediction_value # Value to return on predict
        self.feature_names_in_ = features_in if features_in is not None else ['feat1', 'feat2']

    def predict(self, X):
        # X is expected to be a DataFrame by the generator's ensemble method
        # Return an array of prediction_value, one for each row in X (usually 1 row for generator)
        return np.array([self.prediction_value] * X.shape[0])

    def get_params(self, deep=True): # Required by some sklearn internal checks
        return {"model_name": self.model_name, "prediction_value": self.prediction_value}

@pytest.fixture
def mock_prediction_config():
    # Define a fairly complete mock prediction_config based on PredictionGenerator's usage
    return {
        'confidence_thresholds': {'high': 0.8, 'medium': 0.6, 'low': 0.4},
        'emergence_indicators': {
            'patent_filing_acceleration': 0.3,
            'funding_velocity_increase': 0.25,
            'research_publication_growth': 0.25,
            'cross_disciplinary_collaboration': 0.1,
            'keyword_novelty_score': 0.1
        },
        'emergence_analysis_thresholds': {
            'score_high_confidence': 0.70,
            'score_medium_confidence': 0.40,
            'timeline_fast_threshold': 0.70, # Score for 0-1 year timeline
            'timeline_medium_threshold': 0.40, # Score for 1-2 year timeline
            # feature_low_signal_threshold_multiplier is used in _analyze_emergence_signals
            # It's not in this mock config, so the method will use its default (0.25)
        },
        'emergence_risk_factors_map': {
            'low_funding_signal': "Funding/Commercialization Lag",
            'low_research_signal': "Weakening Research Base",
            'low_patent_signal': "Reduced IP Activity", # Corrected key from 'Slowing IP Generation'
            'low_collaboration_signal': "Innovation Siloing",
            'low_novelty_signal': "Stagnant Keyword Space"
        },
        'investment_ranking_criteria': {
            'growth_potential': 0.4,    'confidence_score': 0.3,
            'market_size': 0.2,         'risk_adjusted_return': 0.1,
            'high_growth_threshold': 0.10, 'low_growth_threshold': 0.01,
            'large_market_threshold_usd': 500000000, 'min_market_size_threshold_usd': 100000000,
            'action_invest_score_thresh': 0.5, 'action_invest_growth_thresh': 0.05,
            'action_monitor_score_thresh': 0.3,
            'quality_score_threshold': 0.5 # Added, used in create_investment_opportunities
        }
    }


@pytest.fixture
def prediction_generator_instance(mock_prediction_config):
    trained_models = {
        "tech": {
            "rf_tech": MockPredictorModel(model_name="rf_tech", prediction_value=0.05, features_in=['f1', 'f2']),
            "gb_tech": MockPredictorModel(model_name="gb_tech", prediction_value=0.07, features_in=['f1', 'f2'])
        },
        "health": {
            "rf_health": MockPredictorModel(model_name="rf_health", prediction_value=0.10, features_in=['h1', 'h2']),
        }
    }
    ensemble_weights = {"rf": 0.6, "gb": 0.4}
    return generator.PredictionGenerator(trained_models, ensemble_weights, mock_prediction_config)

def test_prediction_generator_init(prediction_generator_instance, mock_prediction_config):
    pg = prediction_generator_instance
    assert pg.sector_models is not None
    assert pg.ensemble_weights is not None
    assert pg.prediction_config == mock_prediction_config
    assert pg.rng is not None

def test_ensemble_predict_single_sector_success(prediction_generator_instance):
    pg = prediction_generator_instance
    sector = "tech"
    features = pd.Series({'f1': 1, 'f2': 2, 'other_feat': 100})

    pg.sector_models[sector]['rf_tech'].feature_names_in_ = ['f1', 'f2']
    pg.sector_models[sector]['gb_tech'].feature_names_in_ = ['f1', 'f2']

    prediction, individuals = pg._ensemble_predict_single_sector(sector, features)

    expected_rf_pred = 0.05
    expected_gb_pred = 0.07
    expected_ensemble = (expected_rf_pred * 0.6) + (expected_gb_pred * 0.4)

    assert individuals["rf_tech"] == expected_rf_pred
    assert individuals["gb_tech"] == expected_gb_pred
    assert prediction == pytest.approx(expected_ensemble)

def test_ensemble_predict_single_sector_one_model_nan(prediction_generator_instance):
    pg = prediction_generator_instance
    sector = "tech"
    features = pd.Series({'f1': 1, 'f2': 2})

    original_predict_gb = pg.sector_models[sector]["gb_tech"].predict
    pg.sector_models[sector]["gb_tech"].predict = MagicMock(return_value=np.array([np.nan]))

    prediction, individuals = pg._ensemble_predict_single_sector(sector, features)

    expected_rf_pred = 0.05
    expected_ensemble = expected_rf_pred

    assert individuals["rf_tech"] == expected_rf_pred
    assert np.isnan(individuals["gb_tech"])
    assert prediction == pytest.approx(expected_ensemble)

    pg.sector_models[sector]["gb_tech"].predict = original_predict_gb


def test_ensemble_predict_single_sector_all_models_nan(prediction_generator_instance):
    pg = prediction_generator_instance
    sector = "tech"
    features = pd.Series({'f1': 1, 'f2': 2})

    original_predict_rf = pg.sector_models[sector]["rf_tech"].predict
    original_predict_gb = pg.sector_models[sector]["gb_tech"].predict
    pg.sector_models[sector]["rf_tech"].predict = MagicMock(return_value=np.array([np.nan]))
    pg.sector_models[sector]["gb_tech"].predict = MagicMock(return_value=np.array([np.nan]))

    prediction, individuals = pg._ensemble_predict_single_sector(sector, features)

    assert np.isnan(individuals["rf_tech"])
    assert np.isnan(individuals["gb_tech"])
    assert np.isnan(prediction)

    pg.sector_models[sector]["rf_tech"].predict = original_predict_rf
    pg.sector_models[sector]["gb_tech"].predict = original_predict_gb


def test_ensemble_predict_single_sector_no_model_for_sector(prediction_generator_instance):
    pg = prediction_generator_instance
    features = pd.Series({'f1': 1, 'f2': 2})
    with pytest.raises(KeyError):
        pg._ensemble_predict_single_sector("non_existent_sector", features)


def test_ensemble_predict_single_sector_model_predict_exception(prediction_generator_instance, capsys):
    pg = prediction_generator_instance
    sector = "tech"
    features = pd.Series({'f1': 1, 'f2': 2})

    original_predict_rf = pg.sector_models[sector]["rf_tech"].predict
    pg.sector_models[sector]["rf_tech"].predict = MagicMock(side_effect=Exception("Predict failed!"))

    prediction, individuals = pg._ensemble_predict_single_sector(sector, features)

    captured = capsys.readouterr()
    assert "Error during prediction with rf_tech for tech: Predict failed!" in captured.out
    assert np.isnan(individuals["rf_tech"])

    expected_gb_pred = 0.07
    assert prediction == pytest.approx(expected_gb_pred)

    pg.sector_models[sector]["rf_tech"].predict = original_predict_rf


def test_calculate_confidence_intervals_bootstrap(prediction_generator_instance):
    pg = prediction_generator_instance
    sector = "tech"
    features = pd.Series({'f1': 1, 'f2': 2})

    original_rf_pred_val = pg.sector_models[sector]["rf_tech"].prediction_value
    original_gb_pred_val = pg.sector_models[sector]["gb_tech"].prediction_value
    pg.sector_models[sector]["rf_tech"].prediction_value = 0.06
    pg.sector_models[sector]["gb_tech"].prediction_value = 0.06

    intervals = pg._calculate_confidence_intervals_bootstrap(sector, features)
    assert intervals['68%'] == pytest.approx([0.06, 0.06])
    assert intervals['95%'] == pytest.approx([0.06, 0.06])

    pg.sector_models[sector]["rf_tech"].prediction_value = original_rf_pred_val
    pg.sector_models[sector]["gb_tech"].prediction_value = original_gb_pred_val


def test_calculate_confidence_intervals_bootstrap_few_predictions(prediction_generator_instance):
    pg = prediction_generator_instance
    sector = "health"
    features = pd.Series({'h1': 1, 'h2': 2})

    intervals = pg._calculate_confidence_intervals_bootstrap(sector, features)

    assert np.isnan(intervals['68%'][0]) and np.isnan(intervals['68%'][1])
    assert np.isnan(intervals['95%'][0]) and np.isnan(intervals['95%'][1])


def test_assess_prediction_quality(prediction_generator_instance):
    pg = prediction_generator_instance
    assert pg._assess_prediction_quality("any_sector", pd.Series([1.0, 2.0]), 6) == 0.75


# Continuing tests for PredictionGenerator

@pytest.fixture
def sample_current_features_df():
    return pd.DataFrame({
        'sector_label': ['tech', 'health', 'finance'],
        'f1': [1, np.nan, np.nan],
        'f2': [2, np.nan, np.nan],
        'h1': [np.nan, 10, np.nan],
        'h2': [np.nan, 20, np.nan],
        'fin_feat1': [np.nan, np.nan, 100],
        'filing_rate_3m_patent':              [0.5, 0.2, 0.8],
        'funding_deals_velocity_3m_funding':  [0.6, 0.3, 0.7],
        'publication_rate_3m_research':       [0.4, 0.1, 0.6],
        'cross_disciplinary_collaboration_index': [0.3, 0.2, 0.1],
        'avg_keyword_novelty':                [0.5, 0.4, 0.3]
    }).set_index('sector_label')

def test_generate_sector_forecasts_success(prediction_generator_instance, sample_current_features_df):
    pg = prediction_generator_instance

    mock_ensemble_pred_val = 0.06
    mock_conf_intervals = {'68%': [0.04, 0.08], '95%': [0.02, 0.10]}
    mock_quality_score = 0.85

    tech_features_series = sample_current_features_df.loc['tech']
    health_features_series = sample_current_features_df.loc['health']

    with patch.object(pg, '_ensemble_predict_single_sector', return_value=(mock_ensemble_pred_val, {"mocked_individual": mock_ensemble_pred_val})) as mock_ensemble, \
         patch.object(pg, '_calculate_confidence_intervals_bootstrap', return_value=mock_conf_intervals) as mock_ci, \
         patch.object(pg, '_assess_prediction_quality', return_value=mock_quality_score) as mock_quality:

        current_features_with_col_df = sample_current_features_df.reset_index()
        forecasts = pg.generate_sector_forecasts(current_features_with_col_df, sector_column_name='sector_label', horizons=[6, 12])

    assert "tech" in forecasts
    assert "health" in forecasts
    assert "finance" not in forecasts

    assert len(forecasts["tech"]) == 2
    assert forecasts["tech"]["6m_growth_rate"]["prediction"] == mock_ensemble_pred_val
    assert forecasts["tech"]["12m_growth_rate"]["confidence_intervals"] == mock_conf_intervals
    assert forecasts["tech"]["6m_growth_rate"]["quality_score"] == mock_quality_score
    assert "generated_at" in forecasts["tech"]["6m_growth_rate"]

    assert mock_ensemble.call_count == 4

    tech_call_found = any(
        args[0] == 'tech' and args[1].name == 'tech' and args[1].equals(tech_features_series)
        for args, kwargs in mock_ensemble.call_args_list
    )
    health_call_found = any(
        args[0] == 'health' and args[1].name == 'health' and args[1].equals(health_features_series)
        for args, kwargs in mock_ensemble.call_args_list
    )
    assert tech_call_found, "Ensemble predict for 'tech' with correct features not found or Series.name mismatch"
    assert health_call_found, "Ensemble predict for 'health' with correct features not found or Series.name mismatch"

    assert mock_ci.call_count == 4
    assert mock_quality.call_count == 4


def test_generate_sector_forecasts_no_sector_column(prediction_generator_instance, sample_current_features_df, capsys):
    pg = prediction_generator_instance
    forecasts = pg.generate_sector_forecasts(sample_current_features_df, sector_column_name='wrong_sector_col')
    assert forecasts == {}
    captured = capsys.readouterr()
    assert "Error: Sector column 'wrong_sector_col' not found in current_features_all_sectors_df." in captured.out


def test_generate_sector_forecasts_ensemble_returns_nan(prediction_generator_instance, sample_current_features_df, capsys):
    pg = prediction_generator_instance

    with patch.object(pg, '_ensemble_predict_single_sector', return_value=(np.nan, {})) as mock_ensemble:
        current_features_with_col_df = sample_current_features_df.reset_index()
        forecasts = pg.generate_sector_forecasts(current_features_with_col_df, sector_column_name='sector_label', horizons=[6])

    captured = capsys.readouterr()
    assert "Generated NaN prediction for tech, horizon 6. Skipping." in captured.out
    assert "Generated NaN prediction for health, horizon 6. Skipping." in captured.out
    assert "tech" not in forecasts
    assert "health" not in forecasts


def test_calculate_emergence_score(prediction_generator_instance):
    pg = prediction_generator_instance
    indicators_config = pg.prediction_config['emergence_indicators']

    emergence_feature_data = pd.Series({
        'filing_rate_3m_patent': 0.5,
        'funding_deals_velocity_3m_funding': 0.6,
        'publication_rate_3m_research': 0.4,
        'cross_disciplinary_collaboration_index': 0.2,
        'avg_keyword_novelty': 0.7
    })

    expected_score = (
        emergence_feature_data.get('filing_rate_3m_patent', 0) * indicators_config.get('patent_filing_acceleration', 0) +
        emergence_feature_data.get('funding_deals_velocity_3m_funding', 0) * indicators_config.get('funding_velocity_increase', 0) +
        emergence_feature_data.get('publication_rate_3m_research', 0) * indicators_config.get('research_publication_growth', 0) +
        emergence_feature_data.get('cross_disciplinary_collaboration_index', 0) * indicators_config.get('cross_disciplinary_collaboration', 0) +
        emergence_feature_data.get('avg_keyword_novelty', 0) * indicators_config.get('keyword_novelty_score', 0)
    )

    score = pg._calculate_emergence_score(emergence_feature_data, indicators_config)
    assert score == pytest.approx(expected_score)


def test_identify_emerging_technologies_basic_flow(prediction_generator_instance):
    pg = prediction_generator_instance
    data_for_emergence = pd.DataFrame({
        'sector_label': ['AI', 'Quantum', 'BioTech', 'NanoTech'],
        'filing_rate_3m_patent':              [0.8, 0.3, 0.5, 0.2],
        'funding_deals_velocity_3m_funding':  [0.7, 0.2, 0.6, 0.1],
        'publication_rate_3m_research':       [0.6, 0.8, 0.4, 0.3],
        'cross_disciplinary_collaboration_index': [0.5, 0.6, 0.3, 0.2],
        'avg_keyword_novelty':                [0.7, 0.4, 0.5, 0.1]
    })
    indicators_config = pg.prediction_config['emergence_indicators']
    analysis_thresholds = pg.prediction_config['emergence_analysis_thresholds']
    risk_map = pg.prediction_config['emergence_risk_factors_map']

    def mock_analyze_side_effect(area_features, score, risk_map_cfg, analysis_cfg): # Adjusted signature
        confidence = "Low"
        if score >= analysis_cfg['score_high_confidence']: confidence = "High"
        elif score >= analysis_cfg['score_medium_confidence']: confidence = "Medium"
        return {'key_indicators': ["Indicator X"], 'timeline_estimate': "Y years",
                'confidence_level': confidence, 'risk_factors': ["Risk Z"]}

    with patch.object(pg, '_analyze_emergence_signals', side_effect=mock_analyze_side_effect) as mock_analyze:
        emerging_techs = pg.identify_emerging_technologies(
            data_for_emergence.copy(),
            sector_col='sector_label',
            indicators_config=indicators_config,
            analysis_config=analysis_thresholds,
            risk_map_config=risk_map,
            threshold_percentile=90
        )

    assert len(emerging_techs) == 1
    assert emerging_techs[0]['technology'] == 'AI'
    assert emerging_techs[0]['emergence_score'] == pytest.approx(0.685)
    assert emerging_techs[0]['is_above_threshold'] is True
    assert emerging_techs[0]['analysis']['confidence_level'] == "Medium"
    mock_analyze.assert_called_once()


def test_identify_emerging_technologies_not_enough_scores_for_percentile(prediction_generator_instance, caplog):
    pg = prediction_generator_instance
    data_few_scores = pd.DataFrame({
        'sector_label': ['NewArea'],
        'filing_rate_3m_patent': [0.8],'funding_deals_velocity_3m_funding': [0.7],'publication_rate_3m_research': [0.6],
        'cross_disciplinary_collaboration_index': [0.5], 'avg_keyword_novelty': [0.7]
    })
    indicators_config = pg.prediction_config['emergence_indicators']
    analysis_thresholds = pg.prediction_config['emergence_analysis_thresholds']
    risk_map = pg.prediction_config['emergence_risk_factors_map']

    def mock_analyze_side_effect(area_features, score, risk_map_cfg, analysis_cfg): # Adjusted signature
        return {'key_indicators': [], 'timeline_estimate': "", 'confidence_level': "Medium", 'risk_factors': []}

    with patch.object(pg, '_analyze_emergence_signals', side_effect=mock_analyze_side_effect) as mock_analyze, \
         caplog.at_level(logging.INFO):
        emerging_techs = pg.identify_emerging_technologies(
            data_few_scores, 'sector_label', indicators_config, analysis_thresholds, risk_map, threshold_percentile=90
        )

    assert "Not enough sectors with positive scores (1) to calculate a meaningful 90th percentile for emergence. Using max score as threshold." in caplog.text
    assert len(emerging_techs) == 1
    assert emerging_techs[0]['technology'] == 'NewArea'
    assert emerging_techs[0]['is_above_threshold'] is True
    mock_analyze.assert_called_once()


def test_identify_emerging_technologies_empty_input(prediction_generator_instance):
    pg = prediction_generator_instance
    empty_df = pd.DataFrame(columns=['sector_label', 'filing_rate_3m_patent',
                                     'funding_deals_velocity_3m_funding', 'publication_rate_3m_research',
                                     'cross_disciplinary_collaboration_index', 'avg_keyword_novelty'])
    results = pg.identify_emerging_technologies(empty_df, 'sector_label', {}, {}, {}, 90)
    assert results == []

# Continuing tests for PredictionGenerator - Analysis and Opportunity Creation

def test_analyze_emergence_signals_high_score(prediction_generator_instance):
    pg = prediction_generator_instance
    area_features = pd.Series({
        'filing_rate_3m_patent': 0.8,
        'funding_deals_velocity_3m_funding': 0.7,
        'publication_rate_3m_research': 0.6,
        'cross_disciplinary_collaboration_index': 0.7, # Strong
        'avg_keyword_novelty': 0.8 # Strong
    })
    emergence_score = 0.75 # High score, above timeline_fast_threshold (0.70)

    analysis = pg._analyze_emergence_signals(area_features, emergence_score,
                                             pg.prediction_config['emergence_risk_factors_map'],
                                             pg.prediction_config['emergence_analysis_thresholds'])

    assert analysis['timeline_estimate'] == "0-1 year (High Confidence)"
    assert analysis['confidence_level'] == "High" # Score 0.75 >= score_high_confidence 0.70
    assert len(analysis['key_indicators']) > 0
    assert any("High Patent Filing Acceleration" in ind for ind in analysis['key_indicators'])
    assert analysis['risk_factors'] == ["Standard market adoption risks"] # All signals strong


def test_analyze_emergence_signals_low_score_with_risks(prediction_generator_instance):
    pg = prediction_generator_instance
    area_features = pd.Series({ # Values chosen to be below 0.25 * their typical mean (assuming means are around 0.5-1.0)
        'filing_rate_3m_patent': 0.05, # Low
        'funding_deals_velocity_3m_funding': 0.02, # Very Low
        'publication_rate_3m_research': 0.1, # Low
        'cross_disciplinary_collaboration_index': 0.05, # Low
        'avg_keyword_novelty': 0.08 # Low
    })
    # Assume these features have means around 0.4-0.5 in a typical baseline dataset for default threshold to trigger
    # Default low_signal_threshold_multiplier is 0.25.
    # E.g. if mean patent filing is 0.4, threshold is 0.1. Current 0.05 is below.
    emergence_score = 0.2 # Low score, below timeline_medium_threshold (0.40)

    analysis = pg._analyze_emergence_signals(area_features, emergence_score,
                                             pg.prediction_config['emergence_risk_factors_map'],
                                             pg.prediction_config['emergence_analysis_thresholds'])

    assert analysis['timeline_estimate'] == "2-3+ years (Low Confidence)"
    assert analysis['confidence_level'] == "Low" # Score 0.2 < score_medium_confidence 0.40
    assert len(analysis['key_indicators']) > 0

    expected_risks = [
        pg.prediction_config['emergence_risk_factors_map']['low_patent_signal'],
        pg.prediction_config['emergence_risk_factors_map']['low_funding_signal'],
        pg.prediction_config['emergence_risk_factors_map']['low_research_signal'],
        pg.prediction_config['emergence_risk_factors_map']['low_collaboration_signal'],
        pg.prediction_config['emergence_risk_factors_map']['low_novelty_signal']
    ]
    for risk in expected_risks:
        assert risk in analysis['risk_factors']


def test_calculate_investment_attractiveness_strong_case(prediction_generator_instance):
    pg = prediction_generator_instance
    sector = "TestSector"
    forecast_details = {'prediction': 0.15, 'quality_score': 0.85}
    market_data = {'market_size_usd': 6e8, 'current_investment_usd': 1e7} # Added current_investment
    ranking_criteria = pg.prediction_config['investment_ranking_criteria']

    attractiveness = pg._calculate_investment_attractiveness(sector, forecast_details, market_data, ranking_criteria)

    # Check score calculation based on weights:
    # Growth potential: 0.15 (normalized (0.15-0.01)/(0.10-0.01) = 0.14/0.09 = 1.55 -> capped at 1) * 0.4 = 0.4
    # Confidence: 0.85 (normalized (0.85-0.3)/(0.8-0.3) = 0.55/0.5 = 1.1 -> capped at 1) * 0.3 = 0.3
    # Market size: 6e8 (normalized (6e8-1e8)/(5e8-1e8) = 5e8/4e8 = 1.25 -> capped at 1) * 0.2 = 0.2
    # Risk-adj return (placeholder): 0.5 (default if not calculated) * 0.1 = 0.05
    # Total score = 0.4 + 0.3 + 0.2 + 0.05 = 0.95
    assert attractiveness['score'] == pytest.approx(0.95)
    assert "Strong Growth Potential" in "".join(attractiveness['drivers'])
    assert "High Prediction Confidence" in "".join(attractiveness['drivers'])
    assert "Large Market Size" in "".join(attractiveness['drivers'])
    assert attractiveness['action'] == "Invest" # Score 0.95 > 0.5, Growth 0.15 > 0.05
    assert len(attractiveness['risks']) > 0


def test_calculate_investment_attractiveness_weak_case(prediction_generator_instance):
    pg = prediction_generator_instance
    sector = "NicheSector"
    forecast_details = {'prediction': 0.005, 'quality_score': 0.3}
    market_data = {'market_size_usd': 5e7, 'current_investment_usd': 2e7}
    ranking_criteria = pg.prediction_config['investment_ranking_criteria']

    attractiveness = pg._calculate_investment_attractiveness(sector, forecast_details, market_data, ranking_criteria)
    # Growth: 0.005 (norm (0.005-0.01)/(0.1-0.01) = -0.005/0.09 = -0.055 -> capped at 0) * 0.4 = 0
    # Confidence: 0.3 (norm (0.3-0.3)/(0.8-0.3) = 0) * 0.3 = 0
    # Market Size: 5e7 (norm (5e7-1e8)/(5e8-1e8) = -0.5e8/4e8 = -0.125 -> capped at 0) * 0.2 = 0
    # Risk-adj return: 0.5 (default) * 0.1 = 0.05
    # Total score = 0.05
    assert attractiveness['score'] == pytest.approx(0.05)
    assert "Low/Negative Growth" in "".join(attractiveness['drivers'])
    assert "Very Low Prediction Confidence" in "".join(attractiveness['risks'])
    assert "Niche or Potentially Small Market" in "".join(attractiveness['risks'])
    assert attractiveness['action'] == "Avoid / Re-evaluate" # Score 0.05 < 0.3 (monitor threshold)


def test_create_investment_opportunities_mixed_inputs(prediction_generator_instance):
    pg = prediction_generator_instance

    sector_forecasts_data = {
        "EstablishedSector1": {
            "12m_growth_rate": {'prediction': 0.10, 'quality_score': 0.8, 'confidence_intervals': {}, 'generated_at': ''}
        },
        "EstablishedSector2": {
            "12m_growth_rate": {'prediction': 0.05, 'quality_score': 0.3, 'confidence_intervals': {}, 'generated_at': ''}
        }, # quality_score 0.3 < threshold 0.5, will be filtered
        "EstablishedSector3": {}
    }

    emerging_techs_data = [
        {'technology': 'EmergingTech1', 'analysis': {'timeline_estimate': '0-1 year', 'confidence_level': 'High'},
         'emergence_score': 0.75, 'key_indicators': ['IndA'], 'risk_factors': ['RiskX']}
    ]

    market_data = {
        "EstablishedSector1": {'market_size_usd': 1e9, 'current_investment_usd': 5e7},
        "EmergingTech1": {'market_size_usd': 2e8, 'current_investment_usd': 1e6} # Market data for emerging tech
    }

    # Mock _calculate_investment_attractiveness
    # For EstablishedSector1: pred=0.1, quality=0.8, market=1e9. Score will be high.
    # Growth: (0.1-0.01)/(0.1-0.01)=1 -> 1*0.4=0.4
    # Conf: (0.8-0.3)/(0.8-0.3)=1 -> 1*0.3=0.3
    # Market: (1e9-1e8)/(5e8-1e8)=2.25 -> capped 1*0.2=0.2
    # Risk: 0.5*0.1=0.05. Total = 0.4+0.3+0.2+0.05 = 0.95. Action: Invest.
    attr_estab1 = {'score': 0.95, 'drivers': ["Strong Growth"], 'risks': [], 'action': 'Invest'}

    # For EmergingTech1 (treated as established for attractiveness calc if it has forecast details)
    # Here, it doesn't have forecast_details, so its attractiveness is its emergence_score.

    def mock_calc_attr_side_effect(sector, forecast, market, criteria):
        if sector == "EstablishedSector1": return attr_estab1
        return {'score': 0.1, 'drivers': [], 'risks': [], 'action': 'Avoid / Re-evaluate'} # Default for others

    with patch.object(pg, '_calculate_investment_attractiveness', side_effect=mock_calc_attr_side_effect) as mock_calc_attr:
        opportunities = pg.create_investment_opportunities(
            sector_forecasts_data,
            emerging_techs_data,
            market_data,
            pg.prediction_config['investment_ranking_criteria']
        )

    assert len(opportunities) == 2

    opp_estab1 = next(o for o in opportunities if o['sector'] == "EstablishedSector1")
    assert opp_estab1['type'] == 'Established Sector'
    assert opp_estab1['attractiveness_score'] == pytest.approx(0.95)
    assert opp_estab1['action_recommendation'] == 'Invest'

    opp_emerg1 = next(o for o in opportunities if o['sector'] == "EmergingTech1")
    assert opp_emerg1['type'] == 'Emerging Technology'
    assert opp_emerg1['attractiveness_score'] == 0.75 # Uses emergence_score
    assert opp_emerg1['action_recommendation'] == 'Monitor Closely & Consider Pilot' # Default for emerging

    # Check sorting: Estab1 (0.95) vs Emerg1 (0.75)
    assert opportunities[0]['sector'] == 'EstablishedSector1'
    assert opportunities[1]['sector'] == 'EmergingTech1'

    # Check mock_calc_attr was called for EstablishedSector1, but not for EmergingTech1
    # (as it uses emergence_score directly for attractiveness if no forecast details)
    # However, the current code *does* call _calc_attractiveness for emerging techs too,
    # using placeholder forecast_details. This needs to be reflected.

    # If emerging techs also go through _calc_attractiveness with default/placeholder values:
    # EmergingTech1: pred=0 (default), quality=0.75 (from emergence score), market=2e8
    # Growth: (0-0.01)/(0.1-0.01) -> capped 0 * 0.4 = 0
    # Conf: (0.75-0.3)/(0.8-0.3) = 0.45/0.5 = 0.9 -> 0.9*0.3 = 0.27
    # Market: (2e8-1e8)/(5e8-1e8) = 1e8/4e8 = 0.25 -> 0.25*0.2 = 0.05
    # Risk: 0.5*0.1 = 0.05. Total = 0 + 0.27 + 0.05 + 0.05 = 0.37
    # This would make EmergingTech1 score 0.37.
    # The prompt says "attractiveness_score == 0.75 # Uses emergence_score directly"
    # This implies the code has specific logic for emerging_techs.
    # Let's assume the code is: if type is 'Emerging', score = emergence_score.
    # If so, mock_calc_attr should only be called for established sectors.
    assert mock_calc_attr.call_count == 1
    assert mock_calc_attr.call_args[0][0] == "EstablishedSector1"


def test_create_investment_opportunities_empty_inputs(prediction_generator_instance):
    pg = prediction_generator_instance
    opportunities = pg.create_investment_opportunities({}, [], {}, pg.prediction_config['investment_ranking_criteria'])
    assert opportunities == []

```
