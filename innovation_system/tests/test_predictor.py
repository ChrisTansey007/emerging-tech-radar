import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from innovation_system.model_development.predictor import InnovationPredictor # Keep this

# Imports for new pytest-style tests
import pytest
from unittest.mock import patch, MagicMock # Ensure MagicMock is imported
# Ensure all necessary sklearn imports are here, handle ImportError for dummy versions if needed
try:
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.base import BaseEstimator
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError:
    # Dummy classes for environments where scikit-learn might not be fully installed
    class BaseEstimator:
        def get_params(self, deep=True): return {}
        def set_params(self, **params): return self
        def fit(self, X, y=None): self.is_fitted_ = True; return self
        def predict(self, X): return np.zeros(X.shape[0])

    class GridSearchCV:
        def __init__(self, estimator, param_grid, cv=None, scoring=None, refit=True, n_jobs=None, verbose=0): # Added more params
            self.estimator = estimator
            self.param_grid = param_grid
            self.cv = cv
            self.scoring = scoring
            self.refit = refit
            self.n_jobs = n_jobs
            self.verbose = verbose
            self.best_estimator_ = estimator
            self.best_score_ = 0.0

        def fit(self, X, y=None):
            if self.param_grid and isinstance(self.param_grid, dict):
                 # Get the first set of parameters from the grid
                first_params = {k: v[0] for k, v in self.param_grid.items() if isinstance(v, list) and v}
                self.best_estimator_.set_params(**first_params)
            self.best_estimator_.fit(X,y) # Fit the underlying estimator
            self.is_fitted_ = True # Common attribute checked by sklearn utils
            return self

    # Dummy for TimeSeriesSplit if needed by code under test, though predictor.py uses it directly.
    # For now, assume direct use doesn't need a dummy here if the test doesn't construct it.

# --- Pytest Fixture for Data Preparation Test Data ---
@pytest.fixture
def data_prep_setup():
    """Provides the setup data previously in TestPredictorDataPreparation.setUp()"""
    predictor_instance = InnovationPredictor(random_state=42)
    dates1 = pd.to_datetime([datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)])
    dates2 = pd.to_datetime([datetime(2023, 1, 3) + timedelta(days=i) for i in range(5)])

    patent_df_raw = pd.DataFrame({
        'date': dates1, 'filings': np.random.rand(5), 'citations': np.random.randint(1, 10, 5)
    })
    funding_df_raw = pd.DataFrame({
        'date': dates2, 'deals': np.random.rand(5), 'amount': np.random.randint(100, 1000, 5)
    })
    research_df_raw = pd.DataFrame({
        'published_date': dates1, 'papers': np.random.rand(5)
    })
    targets_df_raw = pd.DataFrame({
        'date': dates1, 'target_growth_6m': np.random.randn(5)
    })

    return {
        "predictor": predictor_instance,
        "patent_df_raw": patent_df_raw,
        "funding_df_raw": funding_df_raw,
        "research_df_raw": research_df_raw,
        "targets_df_raw": targets_df_raw
    }

# --- Refactored Data Preparation Tests (pytest style) ---

def test_ensure_datetime_index_with_column(data_prep_setup):
    predictor = data_prep_setup["predictor"]
    patent_df_raw = data_prep_setup["patent_df_raw"]
    df = predictor._ensure_datetime_index(patent_df_raw.copy(), 'patent_df_raw')
    assert isinstance(df.index, pd.DatetimeIndex)
    assert 'date' not in df.columns

def test_ensure_datetime_index_with_existing_index(data_prep_setup):
    predictor = data_prep_setup["predictor"]
    patent_df_raw = data_prep_setup["patent_df_raw"]
    df_indexed = patent_df_raw.set_index(pd.to_datetime(patent_df_raw['date']))
    df_processed = predictor._ensure_datetime_index(df_indexed.copy(), 'df_indexed')
    assert isinstance(df_processed.index, pd.DatetimeIndex)
    assert len(df_indexed.columns) == len(df_processed.columns)

def test_ensure_datetime_index_no_column_no_datetimeindex(data_prep_setup):
    predictor = data_prep_setup["predictor"]
    df_no_date = pd.DataFrame({'data': [1,2,3]})
    df_processed = predictor._ensure_datetime_index(df_no_date.copy(), 'df_no_date')
    assert df_processed.empty
    assert isinstance(df_processed.index, pd.DatetimeIndex)

def test_temporal_alignment_empty_inputs(data_prep_setup):
    predictor = data_prep_setup["predictor"]
    aligned_output = predictor._temporal_alignment(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if isinstance(aligned_output, tuple):
        aligned_df, aligned_series = aligned_output
        assert aligned_df.empty
        assert aligned_series.empty
    else:
        assert aligned_output.empty

def test_temporal_alignment_features_only(data_prep_setup):
    predictor = data_prep_setup["predictor"]
    patent_df_raw = data_prep_setup["patent_df_raw"]
    funding_df_raw = data_prep_setup["funding_df_raw"]

    patent_df = predictor._ensure_datetime_index(patent_df_raw.copy(), 'patent_df_raw')
    funding_df = predictor._ensure_datetime_index(funding_df_raw.copy(), 'funding_df_raw')

    aligned_output = predictor._temporal_alignment(patent_df, funding_df, pd.DataFrame(), pd.DataFrame())
    aligned_df = aligned_output
    if isinstance(aligned_output, tuple):
        aligned_df, _ = aligned_output

    assert not aligned_df.empty
    assert 'filings_patent' in aligned_df.columns
    assert 'amount_funding' in aligned_df.columns
    assert not aligned_df.isnull().any().any(), "NaNs should be handled by ffill/bfill"

def test_temporal_alignment_with_targets(data_prep_setup):
    predictor = data_prep_setup["predictor"]
    patent_df_raw = data_prep_setup["patent_df_raw"]
    funding_df_raw = data_prep_setup["funding_df_raw"]
    targets_df_raw = data_prep_setup["targets_df_raw"]

    patent_df = predictor._ensure_datetime_index(patent_df_raw.copy(), 'patent_df_raw')
    funding_df = predictor._ensure_datetime_index(funding_df_raw.copy(), 'funding_df_raw')
    targets_df = predictor._ensure_datetime_index(targets_df_raw.copy(), 'targets_df_raw')

    aligned_df = predictor._temporal_alignment(patent_df, funding_df, pd.DataFrame(), targets_df)

    assert not aligned_df.empty
    assert 'filings_patent' in aligned_df.columns
    target_col_name = 'target_growth_6m' if 'target_growth_6m' in targets_df.columns else targets_df.columns[0]
    assert target_col_name in aligned_df.columns
    assert all(idx in targets_df.index for idx in aligned_df.index)

    check_cols = [col for col in ['filings_patent', 'amount_funding', target_col_name] if col in aligned_df.columns]
    if check_cols:
        assert not aligned_df[check_cols].isnull().any().any(), "NaNs should be handled"

def test_temporal_alignment_fill_behavior(data_prep_setup):
    predictor = data_prep_setup["predictor"]
    df1 = pd.DataFrame({'value_df1': [1, np.nan, 3]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    df2 = pd.DataFrame({'value_df2': [np.nan, 5, np.nan]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    targets = pd.DataFrame({'target_growth_6m': [0,0,0]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))

    aligned = predictor._temporal_alignment(df1, df2, pd.DataFrame(), targets)
    assert aligned['value_df1_patent'].tolist() == [1.0, 1.0, 3.0]
    assert aligned['value_df2_funding'].tolist() == [5.0, 5.0, 5.0]

def test_prepare_training_data_simple_run(data_prep_setup):
    predictor = data_prep_setup["predictor"]
    patent_df = predictor._ensure_datetime_index(data_prep_setup["patent_df_raw"].copy(), 'patent_df_raw')
    funding_df = predictor._ensure_datetime_index(data_prep_setup["funding_df_raw"].copy(), 'funding_df_raw')
    research_df = predictor._ensure_datetime_index(data_prep_setup["research_df_raw"].copy(), 'research_df_raw')
    targets_df = predictor._ensure_datetime_index(data_prep_setup["targets_df_raw"].copy(), 'targets_df_raw')

    X, y = predictor.prepare_training_data(patent_df, funding_df, research_df, targets_df)

    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)

    original_feature_cols = list(data_prep_setup["patent_df_raw"].columns) + \
                            list(data_prep_setup["funding_df_raw"].columns) + \
                            list(data_prep_setup["research_df_raw"].columns)
    if 'date' in original_feature_cols: original_feature_cols.remove('date')
    if 'published_date' in original_feature_cols: original_feature_cols.remove('published_date')

    for orig_col in original_feature_cols:
        assert not any(orig_col == x_col for x_col in X.columns), f"Original column {orig_col} should not be in X directly"

    assert any(col.endswith(('_patent', '_funding', '_research')) or '_lag' in col for col in X.columns)
    assert not X.isnull().any().any(), "X should have no NaNs after processing"

def test_prepare_training_data_empty_inputs(data_prep_setup):
    predictor = data_prep_setup["predictor"]
    X, y = predictor.prepare_training_data(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert X.empty, "X should be empty for all empty inputs"
    assert y.empty, "y should be empty for all empty inputs"

def test_prepare_training_data_empty_features_with_targets(data_prep_setup):
    predictor = data_prep_setup["predictor"]
    targets_df = predictor._ensure_datetime_index(data_prep_setup["targets_df_raw"].copy(), 'targets_df_raw')
    X, y = predictor.prepare_training_data(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), targets_df)
    assert X.empty
    assert y.empty

def test_prepare_training_data_normalization_and_lagging(data_prep_setup):
    predictor = data_prep_setup["predictor"]
    dates = pd.to_datetime([datetime(2023,1,1) + timedelta(days=i) for i in range(20)])
    patent_data = pd.DataFrame({'date': dates, 'filings': np.arange(20, dtype=float)})
    patent_df = predictor._ensure_datetime_index(patent_data, 'patent_data')

    targets_data_values = np.arange(20,40, dtype=float)
    targets_df = pd.DataFrame({'target_growth_6m': targets_data_values}, index=dates)

    X, y = predictor.prepare_training_data(patent_df, pd.DataFrame(), pd.DataFrame(), targets_df)

    assert not X.empty
    assert any('_lag1' in col for col in X.columns)
    assert any('_lag3' in col for col in X.columns)
    assert any('_lag6' in col for col in X.columns)
    assert any('_lag12' in col for col in X.columns)

    assert not X.isnull().any().any()
    assert len(X) == len(y)
    pd.testing.assert_index_equal(X.index, y.index)
    assert len(X) < 20


# --- Existing Pytest Style Tests (Should remain unchanged below this line) ---

@pytest.fixture
def predictor(): # This fixture was already defined for other pytest tests
    """Pytest fixture for InnovationPredictor."""
    return InnovationPredictor(random_state=42)

def test_create_lagged_features_valid_df(predictor):
    data = {'feature1': np.arange(10), 'feature2': np.arange(10, 20)}
    idx = pd.date_range('2023-01-01', periods=10, freq='D')
    df = pd.DataFrame(data, index=idx)

    lagged_df = predictor._create_lagged_features(df, ['feature1', 'feature2'], lags=[1, 2])

    assert 'feature1_lag1' in lagged_df.columns
    assert 'feature2_lag2' in lagged_df.columns
    assert pd.isna(lagged_df['feature1_lag1'].iloc[0])
    assert lagged_df['feature1_lag1'].iloc[1] == df['feature1'].iloc[0]
    assert pd.isna(lagged_df['feature2_lag2'].iloc[1])
    assert lagged_df['feature2_lag2'].iloc[2] == df['feature2'].iloc[0]
    assert len(df) == len(lagged_df)

def test_create_lagged_features_empty_df(predictor):
    empty_df = pd.DataFrame()
    lagged_df = predictor._create_lagged_features(empty_df, ['feature1'], lags=[1])
    assert lagged_df.empty

def test_create_lagged_features_no_datetimeindex(predictor, capsys):
    df_no_dt_index = pd.DataFrame({'feature1': [1, 2, 3]})
    lagged_df = predictor._create_lagged_features(df_no_dt_index, ['feature1'], lags=[1])
    assert 'feature1_lag1' in lagged_df.columns
    captured = capsys.readouterr()
    assert "Warning: Dataframe does not have a DatetimeIndex in _create_lagged_features" in captured.out


def test_normalize_features_z_score(predictor):
    data = {'feature1': np.array([1, 2, 3, 4, 5], dtype=float), 'feature2': np.array([10, 20, 30, 40, 50], dtype=float)}
    df = pd.DataFrame(data)

    original_scaler = predictor.scaler
    predictor.scaler = InnovationPredictor(random_state=42).scaler

    with patch.object(predictor.scaler, 'fit_transform', wraps=predictor.scaler.fit_transform) as mock_fit_transform:
        normalized_df = predictor._normalize_features(df.copy(), method='z_score')
        mock_fit_transform.assert_called_once()

    assert normalized_df.shape == df.shape
    assert np.allclose(normalized_df.mean(), 0.0)
    assert np.allclose(normalized_df.std(ddof=0), 1.0)

    predictor.scaler = original_scaler

def test_normalize_features_min_max(predictor):
    data = {'feature1': np.array([1, 2, 3, 4, 5], dtype=float), 'feature2': np.array([10, 20, 30, 40, 50], dtype=float)}
    df = pd.DataFrame(data)

    normalized_df_actual = predictor._normalize_features(df.copy(), method='min_max')
    assert np.allclose(normalized_df_actual.min(), 0.0)
    assert np.allclose(normalized_df_actual.max(), 1.0)


def test_normalize_features_empty_df(predictor):
    empty_df = pd.DataFrame()
    normalized_df = predictor._normalize_features(empty_df.copy(), method='z_score')
    assert normalized_df.empty

def test_normalize_features_unknown_method(predictor):
    data = {'feature1': [1.0, 2.0, 3.0]}
    df = pd.DataFrame(data)
    normalized_df = predictor._normalize_features(df.copy(), method='unknown_method')
    pd.testing.assert_frame_equal(normalized_df, df)


def test_get_param_grid(predictor):
    rf_grid = predictor._get_param_grid('random_forest')
    assert 'n_estimators' in rf_grid
    assert 'max_depth' in rf_grid

    gb_grid = predictor._get_param_grid('gradient_boosting')
    assert 'learning_rate' in gb_grid

    empty_grid = predictor._get_param_grid('unknown_model')
    assert empty_grid == {}

@pytest.fixture
def sample_training_data():
    n_samples = 100
    n_features = 5
    X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'feat_{i}' for i in range(n_features)])
    X['sector_label'] = np.random.choice(['tech', 'health'], size=n_samples)
    y = pd.Series(np.random.rand(n_samples) * 10, name="target_growth_6m")

    X.index = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    y.index = X.index
    return X, y

class MockModel(BaseEstimator): # Renamed from MockModel to avoid clash if file imported elsewhere
    def __init__(self, some_param=1):
        self.some_param = some_param
        self.feature_importances_ = []

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.n_features_in_ = X.shape[1]
            self.feature_importances_ = np.random.rand(X.shape[1])
        elif isinstance(X, np.ndarray):
            self.n_features_in_ = X.shape[1]
            self.feature_importances_ = np.random.rand(X.shape[1])
        else:
            self.n_features_in_ = 0
            self.feature_importances_ = []
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.zeros(X.shape[0] if hasattr(X, 'shape') else len(X))

    def get_params(self, deep=True):
        return {"some_param": self.some_param}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

@patch('sklearn.model_selection.GridSearchCV')
def test_train_sector_models_successful_training(mock_grid_search_cv, predictor, sample_training_data):
    X_train, y_train = sample_training_data

    mock_gs_instance = MagicMock(spec=GridSearchCV)
    mock_model_instance = MockModel()
    num_features_for_model = X_train.drop(columns=['sector_label']).shape[1]
    mock_model_instance.fit(np.random.rand(10, num_features_for_model))
    mock_gs_instance.best_estimator_ = mock_model_instance
    mock_gs_instance.best_score_ = -0.5
    mock_grid_search_cv.return_value = mock_gs_instance

    original_blueprints = predictor.models_blueprints
    predictor.models_blueprints = {
        'mock_model_1': MockModel(),
        'mock_model_2': MockModel()
    }
    original_get_param_grid = predictor._get_param_grid
    def mock_get_param_grid_simple(model_name):
        return {'some_param': [1, 2]}
    predictor._get_param_grid = mock_get_param_grid_simple

    trained_models = predictor.train_sector_models(X_train.copy(), y_train.copy(), sectors_column='sector_label')

    unique_sectors = X_train['sector_label'].unique()
    assert len(trained_models) == len(unique_sectors)

    for sector in unique_sectors:
        assert sector in predictor.sector_models
        assert len(predictor.sector_models[sector]) == len(predictor.models_blueprints)
        for model_name in predictor.models_blueprints.keys():
            assert model_name in predictor.sector_models[sector]
            assert isinstance(predictor.sector_models[sector][model_name], MockModel)
            assert sector in predictor.feature_importances
            assert model_name in predictor.feature_importances[sector]
            assert len(predictor.feature_importances[sector][model_name]) == X_train.shape[1] - 1

    expected_gs_calls = len(unique_sectors) * len(predictor.models_blueprints)
    assert mock_grid_search_cv.call_count == expected_gs_calls
    assert mock_gs_instance.fit.call_count == expected_gs_calls

    predictor.models_blueprints = original_blueprints
    predictor._get_param_grid = original_get_param_grid


def test_train_sector_models_empty_data(predictor, capsys):
    empty_X = pd.DataFrame(columns=['feat_1', 'sector_label'])
    empty_y = pd.Series([], dtype=float)
    result = predictor.train_sector_models(empty_X, empty_y, sectors_column='sector_label')
    assert result == {}
    captured = capsys.readouterr()
    assert "Training data is empty. Skipping model training." in captured.out

def test_train_sector_models_insufficient_data_for_sector(predictor, sample_training_data, capsys):
    X_train, y_train = sample_training_data
    X_modified = X_train.copy()
    y_modified = y_train.copy()

    tech_indices = X_modified[X_modified['sector_label'] == 'tech'].index
    if len(tech_indices) > 5:
        niche_indices_to_assign = np.random.choice(tech_indices, size=5, replace=False)
        X_modified.loc[niche_indices_to_assign, 'sector_label'] = 'niche_sector'
    else:
        X_modified.loc[tech_indices, 'sector_label'] = 'niche_sector'

    if 'health' in X_modified['sector_label'].unique():
        if len(X_modified[X_modified['sector_label'] == 'health']) < 20:
            pass
    else:
        pass

    with patch('sklearn.model_selection.GridSearchCV') as mock_gs:
        mock_gs_instance = MagicMock()
        fitted_mock_estimator = MockModel().fit(X_train.drop(columns=['sector_label']), y_train)
        mock_gs_instance.best_estimator_ = fitted_mock_estimator
        mock_gs.return_value = mock_gs_instance
        predictor.train_sector_models(X_modified, y_modified, sectors_column='sector_label')

    captured = capsys.readouterr()
    assert "Insufficient data for niche_sector (5 samples). Minimum 20 required. Skipping." in captured.out
    assert 'niche_sector' not in predictor.sector_models


@patch('sklearn.model_selection.GridSearchCV')
def test_train_sector_models_training_exception(mock_grid_search_cv, predictor, sample_training_data, capsys):
    X_train, y_train = sample_training_data

    mock_gs_instance = MagicMock(spec=GridSearchCV)
    mock_gs_instance.fit.side_effect = Exception("Training failed miserably")
    mock_grid_search_cv.return_value = mock_gs_instance

    original_blueprints = predictor.models_blueprints
    predictor.models_blueprints = {'mock_model_error': MockModel()}
    original_get_param_grid = predictor._get_param_grid
    predictor._get_param_grid = lambda name: {'some_param': [1]}

    unique_sectors = X_train['sector_label'].unique()
    test_sector = unique_sectors[0]
    X_one_sector = X_train[X_train['sector_label'] == test_sector].copy()
    y_one_sector = y_train[y_train.index.isin(X_one_sector.index)].copy()

    predictor.train_sector_models(X_one_sector, y_one_sector, sectors_column='sector_label')

    captured = capsys.readouterr()
    assert f"Error training mock_model_error for sector {test_sector}: Training failed miserably" in captured.out
    assert test_sector in predictor.sector_models
    assert 'mock_model_error' not in predictor.sector_models.get(test_sector, {})

    predictor.models_blueprints = original_blueprints
    predictor._get_param_grid = original_get_param_grid


def test_validate_models_successful_validation(predictor, sample_training_data):
    X_test, y_test = sample_training_data

    unique_sectors = X_test['sector_label'].unique()
    mock_predictions_map = {}

    for sector in unique_sectors:
        predictor.sector_models[sector] = {}
        sector_mask_test = X_test['sector_label'] == sector
        X_sector_test_df = X_test[sector_mask_test].drop(columns=['sector_label'])
        num_samples_sector = X_sector_test_df.shape[0]

        for model_name_key in InnovationPredictor().models_blueprints.keys():
            model_instance = MockModel()
            constant_pred_value = 1.0 if 'random_forest' in model_name_key else 2.0
            mock_predict_output = np.full(num_samples_sector, constant_pred_value)

            if sector not in mock_predictions_map: mock_predictions_map[sector] = {}
            mock_predictions_map[sector][model_name_key] = mock_predict_output

            model_instance.predict = MagicMock(return_value=mock_predict_output)
            predictor.sector_models[sector][model_name_key] = model_instance

    validation_results = predictor.validate_models(X_test.copy(), y_test.copy(), sectors_column='sector_label')

    assert len(validation_results) == len(unique_sectors)
    for sector in unique_sectors:
        assert sector in validation_results
        y_sector_test_actual = y_test[X_test['sector_label'] == sector]

        for model_name, results in validation_results[sector].items():
            assert model_name in predictor.sector_models[sector]
            actual_model_instance = predictor.sector_models[sector][model_name]
            actual_model_instance.predict.assert_called_once()
            expected_preds_for_metric_calc = mock_predictions_map[sector][model_name]
            expected_mae = mean_absolute_error(y_sector_test_actual, expected_preds_for_metric_calc)
            expected_rmse = np.sqrt(mean_squared_error(y_sector_test_actual, expected_preds_for_metric_calc))
            y_actual_series = y_sector_test_actual.fillna(0)
            preds_series = pd.Series(expected_preds_for_metric_calc, index=y_actual_series.index).fillna(0)
            expected_dir_acc = np.mean(np.sign(y_actual_series) == np.sign(preds_series))

            assert results['mae'] == pytest.approx(expected_mae)
            assert results['rmse'] == pytest.approx(expected_rmse)
            assert results['direction_accuracy'] == pytest.approx(expected_dir_acc)
            assert len(results['predictions_sample']) <= 3
            assert len(results['actuals_sample']) <= 3


def test_validate_models_empty_test_data(predictor, capsys):
    empty_X = pd.DataFrame(columns=['feat_1', 'sector_label'])
    empty_y = pd.Series([], dtype=float)
    results = predictor.validate_models(empty_X, empty_y, sectors_column='sector_label')
    assert results == {}
    captured = capsys.readouterr()
    assert "Test data is empty. Skipping model validation." in captured.out


def test_validate_models_no_trained_model_for_sector(predictor, sample_training_data, capsys):
    X_test, y_test = sample_training_data
    predictor.sector_models = {}
    results = predictor.validate_models(X_test.copy(), y_test.copy(), sectors_column='sector_label')
    captured = capsys.readouterr()
    for sector_in_data in X_test['sector_label'].unique():
         assert f"No trained models for sector '{sector_in_data}'. Skipping validation." in captured.out
    assert results == {}


@patch.object(MockModel, 'predict', side_effect=Exception("Prediction failed badly"))
def test_validate_models_prediction_exception(mock_predict_method_on_class, predictor, sample_training_data, capsys):
    X_test, y_test = sample_training_data
    test_sector = X_test['sector_label'].unique()[0]
    model_name_to_test = list(InnovationPredictor().models_blueprints.keys())[0]
    failing_model_instance = MockModel()
    predictor.sector_models = { test_sector: { model_name_to_test: failing_model_instance } }
    X_one_sector = X_test[X_test['sector_label'] == test_sector].copy()
    y_one_sector = y_test[y_test.index.isin(X_one_sector.index)].copy()

    results = predictor.validate_models(X_one_sector, y_one_sector, sectors_column='sector_label')

    captured = capsys.readouterr()
    assert f"Error validating {model_name_to_test} for sector {test_sector}: Prediction failed badly" in captured.out
    assert test_sector in results
    assert model_name_to_test in results[test_sector]
    assert 'error' in results[test_sector][model_name_to_test]
    assert results[test_sector][model_name_to_test]['error'] == "Prediction failed badly"

# Note: The if __name__ == '__main__': unittest.main(...) block is removed as it's not used by pytest.
```
