import pytest
import sqlite3
import os
import logging
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, ANY

import pandas as pd # Ensure pandas is imported
import numpy as np  # Ensure numpy is imported

from innovation_system.monitoring import monitor
# Assuming model_config and monitoring_config structures are available for testing
# We can define simplified mock configs for monitor tests.

MOCK_MODEL_CONFIG = {
    'performance_threshold': {'mae': 0.15, 'direction_accuracy': 0.65}
}

MOCK_MONITORING_CONFIG = {
    'log_file': 'system_monitor_test.log', # Example
    'auto_retraining_enabled': True,
    # Define default max_delay_days for each pipeline, matching monitor.py if possible
    'pipeline_max_delay_days': {
        'patents_pipeline': 2,
        'funding_pipeline': 7,
        'research_pipeline': 3
    }
}

@pytest.fixture
def mock_configs():
    return MOCK_MODEL_CONFIG, MOCK_MONITORING_CONFIG

@pytest.fixture
def db_in_memory():
    # Use an in-memory SQLite database for most tests to avoid file I/O
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()

@pytest.fixture
def system_monitor_init_only(mock_configs):
    # For testing __init__ without full DB setup, or when DB is mocked separately
    model_cfg, monitor_cfg = mock_configs
    with patch('sqlite3.connect') as mock_connect, \
         patch('os.makedirs') as mock_makedirs:

        mock_conn_instance = MagicMock(spec=sqlite3.Connection)
        mock_cursor = MagicMock(spec=sqlite3.Cursor)
        mock_conn_instance.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn_instance

        # Patch logger to avoid file I/O for log file during this specific init test
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger_instance = MagicMock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger_instance

            sm = monitor.SystemMonitor(
                trained_models_dict={},
                model_cfg=model_cfg,
                monitor_cfg=monitor_cfg,
                db_path="data/test_monitoring.sqlite"
            )
    return sm, mock_connect, mock_makedirs, mock_conn_instance


@pytest.fixture
def system_monitor_with_db(mock_configs): # Removed db_in_memory from params, SystemMonitor creates its own
    model_cfg, monitor_cfg = mock_configs

    # Patch logger to avoid actual file creation for log file
    with patch('logging.getLogger') as mock_get_logger, \
         patch('os.makedirs') as mock_makedirs: # Mock makedirs for db path if not :memory:
        mock_logger_instance = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger_instance

        sm = monitor.SystemMonitor(
            trained_models_dict={},
            model_cfg=model_cfg,
            monitor_cfg=monitor_cfg,
            db_path=":memory:"
        )
    return sm


def test_system_monitor_init(system_monitor_init_only):
    sm, mock_connect, mock_makedirs, mock_conn_instance = system_monitor_init_only

    assert sm.model_config == MOCK_MODEL_CONFIG
    assert sm.monitoring_config == MOCK_MONITORING_CONFIG
    assert sm.db_path == "data/test_monitoring.sqlite"

    if sm.db_path != ":memory:":
        db_dir = os.path.dirname(sm.db_path)
        if db_dir:
             mock_makedirs.assert_called_once_with(db_dir, exist_ok=True)

    mock_connect.assert_called_once_with("data/test_monitoring.sqlite")
    # _initialize_db creates a cursor, executes one SQL statement, and commits.
    mock_conn_instance.cursor.assert_called_once()
    mock_conn_instance.cursor().execute.assert_called_once_with(ANY) # Simplified: check it was called
    mock_conn_instance.commit.assert_called_once()


def test_initialize_db_table_creation(mock_configs):
    model_cfg, monitor_cfg = mock_configs
    # Let SystemMonitor create its own in-memory DB and initialize it
    with patch('logging.getLogger'): # Avoid log file IO
        sm_temp = monitor.SystemMonitor({}, model_cfg, monitor_cfg, db_path=":memory:")

    try:
        cursor_sm = sm_temp.conn.cursor()
        cursor_sm.execute("SELECT * FROM pipeline_status")
        # Check columns (optional, but good for ensuring schema)
        names = [description[0] for description in cursor_sm.description]
        assert 'pipeline_name' in names
        assert 'last_run_timestamp' in names
        assert 'status' in names
        assert 'details' in names
    except sqlite3.OperationalError as e:
        pytest.fail(f"Table pipeline_status not created or query failed: {e}")
    finally:
        sm_temp.conn.close()


def test_initialize_db_error_logging(caplog, mock_configs):
    model_cfg, monitor_cfg = mock_configs
    with patch('sqlite3.connect') as mock_connect, \
         patch('logging.getLogger'): # Avoid log file IO
        mock_conn = MagicMock(spec=sqlite3.Connection)
        # Make cursor() itself raise the error, or execute() on the cursor object
        mock_conn.cursor.side_effect = sqlite3.Error("Test DB error on cursor")
        mock_connect.return_value = mock_conn

        with caplog.at_level(logging.ERROR):
            # SystemMonitor's __init__ calls _initialize_db
            sm = monitor.SystemMonitor({}, model_cfg, monitor_cfg, db_path="dummy.db")

        assert "Error initializing database table: Test DB error on cursor" in caplog.text


def test_update_and_get_pipeline_status(system_monitor_with_db):
    sm = system_monitor_with_db
    pipeline_name = "test_pipeline"
    status = "SUCCESS"
    details = "All good"

    sm.update_pipeline_status(pipeline_name, status, details)

    retrieved_status_info = sm._get_pipeline_status_from_db(pipeline_name)

    assert retrieved_status_info is not None
    assert retrieved_status_info['status'] == status
    assert retrieved_status_info['details'] == details
    assert retrieved_status_info['last_run_timestamp'] is not None

    new_status = "FAILURE"
    new_details = "Something broke"
    sm.update_pipeline_status(pipeline_name, new_status, new_details)
    retrieved_new_status_info = sm._get_pipeline_status_from_db(pipeline_name)
    assert retrieved_new_status_info is not None
    assert retrieved_new_status_info['status'] == new_status
    assert retrieved_new_status_info['details'] == new_details


def test_get_pipeline_status_not_found(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    with caplog.at_level(logging.INFO): # monitor.py logs at INFO if not found
        status_info = sm._get_pipeline_status_from_db("non_existent_pipeline")

    assert status_info is None # Method returns None if not found
    assert "No status found in DB for pipeline: non_existent_pipeline" in caplog.text


def test_pipeline_status_db_error_on_get(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    with patch.object(sm.conn, 'cursor') as mock_cursor_method:
        mock_cursor_instance = MagicMock(spec=sqlite3.Cursor)
        mock_cursor_instance.execute.side_effect = sqlite3.Error("DB query failed")
        mock_cursor_method.return_value = mock_cursor_instance

        with caplog.at_level(logging.ERROR):
            status_info = sm._get_pipeline_status_from_db("any_pipeline")

    assert status_info is None # Returns None on DB error
    assert "Error querying pipeline status for any_pipeline: DB query failed" in caplog.text


def test_pipeline_status_db_error_on_update(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    with patch.object(sm.conn, 'cursor') as mock_cursor_method:
        mock_cursor_instance = MagicMock(spec=sqlite3.Cursor)
        mock_cursor_instance.execute.side_effect = sqlite3.Error("DB update failed")
        mock_cursor_method.return_value = mock_cursor_instance

        with caplog.at_level(logging.ERROR):
            sm.update_pipeline_status("any_pipeline", "TRY_UPDATE", "details")

    assert "Error updating pipeline status for 'any_pipeline': DB update failed" in caplog.text


def test_check_data_pipeline_health_all_healthy(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    now_iso = datetime.now().isoformat()

    def mock_status_side_effect(pipeline_name):
        return {'last_run_timestamp': now_iso, 'status': 'SUCCESS', 'details': ''}

    with patch.object(sm, '_get_pipeline_status_from_db', side_effect=mock_status_side_effect), \
         caplog.at_level(logging.INFO):
        is_healthy = sm.check_data_pipeline_health()

    assert is_healthy is True
    assert "Data pipelines healthy." in caplog.text


def test_check_data_pipeline_health_one_delayed(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    now = datetime.now()
    # patents_pipeline max_delay_days is 2 in MOCK_MONITORING_CONFIG
    delayed_ts = (now - timedelta(days=3)).isoformat()
    ok_ts = now.isoformat()

    def mock_status_side_effect(pipeline_name):
        if pipeline_name == "patents_pipeline":
            return {'last_run_timestamp': delayed_ts, 'status': 'SUCCESS', 'details': ''}
        else:
            return {'last_run_timestamp': ok_ts, 'status': 'SUCCESS', 'details': ''}

    with patch.object(sm, '_get_pipeline_status_from_db', side_effect=mock_status_side_effect), \
         caplog.at_level(logging.WARNING):
        is_healthy = sm.check_data_pipeline_health()

    assert is_healthy is False
    assert "Pipeline 'patents_pipeline' delayed." in caplog.text
    assert "One or more data pipelines have issues." in caplog.text


def test_check_data_pipeline_health_one_failed_status(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    now_iso = datetime.now().isoformat()

    def mock_status_side_effect(pipeline_name):
        if pipeline_name == "funding_pipeline":
            return {'last_run_timestamp': now_iso, 'status': 'FAILED', 'details': 'Error X'}
        else:
            return {'last_run_timestamp': now_iso, 'status': 'SUCCESS', 'details': ''}

    with patch.object(sm, '_get_pipeline_status_from_db', side_effect=mock_status_side_effect), \
         caplog.at_level(logging.WARNING):
        is_healthy = sm.check_data_pipeline_health()

    assert is_healthy is False
    assert "Pipeline 'funding_pipeline' status: FAILED. Details: Error X" in caplog.text
    assert "One or more data pipelines have issues." in caplog.text

def test_check_data_pipeline_health_unparseable_timestamp(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    ok_ts = datetime.now().isoformat()

    def mock_status_side_effect(pipeline_name):
        if pipeline_name == "patents_pipeline":
            return {'last_run_timestamp': "not-a-timestamp", 'status': 'SUCCESS', 'details': ''}
        else:
            return {'last_run_timestamp': ok_ts, 'status': 'SUCCESS', 'details': ''}

    with patch.object(sm, '_get_pipeline_status_from_db', side_effect=mock_status_side_effect), \
         caplog.at_level(logging.WARNING): # Changed to WARNING as errors are logged as warnings
        is_healthy = sm.check_data_pipeline_health()

    assert is_healthy is False
    assert "Could not parse timestamp 'not-a-timestamp' for pipeline 'patents_pipeline'." in caplog.text
    # Since last_run_dt becomes None, it's considered delayed.
    assert "Pipeline 'patents_pipeline' delayed." in caplog.text


# MockModel for monitor tests
class MockModelForMonitor:
    def __init__(self, model_name="test_model"):
        self.model_name = model_name
        self.feature_names_in_ = ['feat_1', 'feat_2']

    def predict(self, X):
        if not isinstance(X, pd.DataFrame): # Basic check if X is not DataFrame
            X = pd.DataFrame(X, columns=self.feature_names_in_ if hasattr(self, 'feature_names_in_') else None)
        return X.sum(axis=1) * 0.1

    def get_params(self, deep=True):
        return {"model_name": self.model_name}

@pytest.fixture
def sample_features_df():
    # Features names should match those in SystemMonitor.data_drift_thresholds default keys
    return pd.DataFrame({
        'patent_filing_rate_3m_patent': np.random.rand(100) * 10,
        'funding_amount_velocity_3m_usd_funding': np.random.rand(100) * 1000,
        'publication_rate_3m_research': np.random.rand(100) * 5,
        'other_feature_not_in_thresholds': np.random.rand(100) # Example of a feature not tracked for drift
    })

def test_set_data_drift_baseline(system_monitor_with_db, sample_features_df, caplog):
    sm = system_monitor_with_db
    # Ensure data_drift_thresholds is set on sm before calling set_data_drift_baseline
    # It's typically set in __init__ from monitoring_config or defaults.
    # For this test, let's ensure it's present as expected by the method.
    if not hasattr(sm, 'data_drift_thresholds') or not sm.data_drift_thresholds:
        sm.data_drift_thresholds = {
            'patent_filing_rate_3m_patent': 0.20,
            'funding_amount_velocity_3m_usd_funding': 0.25,
            'publication_rate_3m_research': 0.15
        }

    sm.set_data_drift_baseline(sample_features_df)

    assert len(sm.data_drift_baselines) == len(sm.data_drift_thresholds)
    for feature_name in sm.data_drift_thresholds.keys():
        assert feature_name in sm.data_drift_baselines
        assert 'mean' in sm.data_drift_baselines[feature_name]
        assert 'std' in sm.data_drift_baselines[feature_name]
        assert sm.data_drift_baselines[feature_name]['mean'] == pytest.approx(sample_features_df[feature_name].mean())

    original_thresholds = sm.data_drift_thresholds.copy()
    sm.data_drift_thresholds['missing_feature_in_df'] = 0.1 # This feature is not in sample_features_df
    with caplog.at_level(logging.WARNING):
        sm.set_data_drift_baseline(sample_features_df)
    assert "Feature 'missing_feature_in_df' for baseline not in historical data. Skipping." in caplog.text
    assert 'missing_feature_in_df' not in sm.data_drift_baselines
    sm.data_drift_thresholds = original_thresholds # Restore


def test_monitor_data_drift_no_drift(system_monitor_with_db, sample_features_df, caplog):
    sm = system_monitor_with_db
    if not hasattr(sm, 'data_drift_thresholds') or not sm.data_drift_thresholds:
        sm.data_drift_thresholds = { # Ensure it's set for the test
            'patent_filing_rate_3m_patent': 0.20,
            'funding_amount_velocity_3m_usd_funding': 0.25,
            'publication_rate_3m_research': 0.15
        }
    sm.set_data_drift_baseline(sample_features_df)

    with caplog.at_level(logging.INFO): # Drift logs at INFO if no drift, WARNING if drift
        drifted, summary = sm.monitor_data_drift(sample_features_df.copy())

    assert drifted is False
    assert "No significant data drift detected." in caplog.text
    for feature_name in sm.data_drift_baselines.keys():
        if feature_name in summary:
            assert summary[feature_name]['status'] == 'STABLE'


def test_monitor_data_drift_with_drift(system_monitor_with_db, sample_features_df, caplog):
    sm = system_monitor_with_db
    if not hasattr(sm, 'data_drift_thresholds') or not sm.data_drift_thresholds:
        sm.data_drift_thresholds = {
            'patent_filing_rate_3m_patent': 0.20, # Threshold for drift
            'funding_amount_velocity_3m_usd_funding': 0.25,
            'publication_rate_3m_research': 0.15
        }
    sm.set_data_drift_baseline(sample_features_df)

    current_df_drifted = sample_features_df.copy()
    feature_to_drift = 'patent_filing_rate_3m_patent'

    # Make current mean significantly different (e.g., > threshold * std from baseline mean)
    # The drift logic in monitor.py is: abs(current_mean - baseline_mean) > threshold * baseline_std
    # Let baseline_std be non-zero for this test. If std is 0, drift logic might change.
    if sm.data_drift_baselines[feature_to_drift]['std'] == 0:
         current_df_drifted[feature_to_drift] = current_df_drifted[feature_to_drift] + \
                                                sm.data_drift_thresholds[feature_to_drift] + 0.1 # Ensure change if std is 0
    else:
        drift_magnitude = (sm.data_drift_thresholds[feature_to_drift] + 0.1) * sm.data_drift_baselines[feature_to_drift]['std']
        current_df_drifted[feature_to_drift] = current_df_drifted[feature_to_drift] + drift_magnitude


    with caplog.at_level(logging.WARNING): # Drift logs at WARNING
        drifted, summary = sm.monitor_data_drift(current_df_drifted)

    assert drifted is True
    # Log message format depends on implementation, e.g. "Data drift for 'feature_name': mean change X.XX, std change Y.YY"
    assert f"Data drift detected for feature '{feature_to_drift}'." in caplog.text # Adjust if log msg is different
    assert summary[feature_to_drift]['status'] == 'DRIFT'
    # assert summary[feature_to_drift]['change_in_means'] == pytest.approx(drift_magnitude, abs=0.01) # If this is reported


def test_monitor_data_drift_feature_missing_in_current(system_monitor_with_db, sample_features_df, caplog):
    sm = system_monitor_with_db
    if not hasattr(sm, 'data_drift_thresholds') or not sm.data_drift_thresholds:
         sm.data_drift_thresholds = {'patent_filing_rate_3m_patent': 0.20} # Ensure this key exists
    sm.set_data_drift_baseline(sample_features_df) # Baseline includes 'patent_filing_rate_3m_patent'

    current_df_missing_feature = sample_features_df.drop(columns=['patent_filing_rate_3m_patent'])
    with caplog.at_level(logging.WARNING): # Log for missing feature is WARNING
        drifted, summary = sm.monitor_data_drift(current_df_missing_feature)

    assert "Feature 'patent_filing_rate_3m_patent' for drift check not in current data. Skipping." in caplog.text
    assert drifted is False # Assuming other features (if any being tracked) are stable


def test_evaluate_model_performance_ok(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    sector = "tech"
    mock_rf_model = MockModelForMonitor("RF_tech")
    sm.trained_models_dict = {sector: {"RF_tech": mock_rf_model}}

    X_recent = pd.DataFrame({'feat_1': [1,2,3], 'feat_2': [4,5,6]})
    y_recent = pd.Series([0.5, 0.7, 0.9]) # Actuals: sum*0.1 -> (5*0.1=0.5), (7*0.1=0.7), (9*0.1=0.9)
                                        # MockModelForMonitor predicts these exact values. MAE will be 0.

    with caplog.at_level(logging.INFO): # OK logs at INFO
        is_ok = sm.evaluate_model_performance_on_recent_data(sector, X_recent, y_recent)

    assert is_ok is True
    assert f"Performance for '{sector}' OK on recent data." in caplog.text
    # Check MAE and Dir Acc logs for the model
    assert "RF_tech in tech - MAE: 0.000" in caplog.text
    assert "RF_tech in tech - Direction Accuracy: 1.000" in caplog.text


def test_evaluate_model_performance_degraded_mae(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    sector = "health"
    bad_model = MockModelForMonitor("BadMAE_health")
    sm.trained_models_dict = {sector: {"BadMAE_health": bad_model}}

    X_recent = pd.DataFrame({'feat_1': [1]*10, 'feat_2': [1]*10}) # Predicts 0.2 for all
    y_recent = pd.Series([1.0]*10) # Actual is 1.0. MAE = |0.2-1.0| = 0.8
                                  # Threshold is 0.15, so this should fail.

    with caplog.at_level(logging.WARNING): # Degradation logs at WARNING
        is_ok = sm.evaluate_model_performance_on_recent_data(sector, X_recent, y_recent)

    assert is_ok is False
    assert f"Performance degradation: BadMAE_health in health. MAE: 0.800" in caplog.text # Check formatting
    assert f"Overall performance for '{sector}' DEGRADED." in caplog.text # Check overall summary log


def test_evaluate_model_performance_no_model_for_sector(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    with caplog.at_level(logging.ERROR): # No model is an ERROR for eval
        is_ok = sm.evaluate_model_performance_on_recent_data("unknown_sector", pd.DataFrame({'feat_1':[]}), pd.Series([], dtype=float))
    assert is_ok is False
    assert "No models available for sector unknown_sector for performance evaluation." in caplog.text

def test_trigger_retraining(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    with caplog.at_level(logging.INFO), patch('builtins.print') as mock_print: # Assuming print is placeholder
        sm.trigger_retraining("tech_sector")

    assert "Retraining triggered for sector: tech_sector due to performance/drift issues." in caplog.text
    mock_print.assert_called_once_with("ACTION: Implement retraining logic for sector tech_sector")


def test_run_scheduled_maintenance_pipeline_unhealthy(system_monitor_with_db, caplog):
    sm = system_monitor_with_db
    with patch.object(sm, 'check_data_pipeline_health', return_value=False), \
         caplog.at_level(logging.ERROR): # Abort logs at ERROR
        sm.run_scheduled_maintenance({}, {})

    assert "Aborting scheduled maintenance due to data pipeline health issues." in caplog.text


def test_run_scheduled_maintenance_drift_and_retrain(system_monitor_with_db, sample_features_df, caplog):
    sm = system_monitor_with_db # auto_retraining_enabled is True by MOCK_MONITORING_CONFIG
    sector = "test_sector" # A single sector for simplicity

    # Setup: baseline, a model for the sector
    if not hasattr(sm, 'data_drift_thresholds') or not sm.data_drift_thresholds: # Ensure thresholds exist
        sm.data_drift_thresholds = {'patent_filing_rate_3m_patent': 0.20}
    sm.set_data_drift_baseline(sample_features_df)
    sm.trained_models_dict = {sector: {"SomeModel": MockModelForMonitor("SomeModel")}}

    # Mock underlying checks: Drift IS detected, Performance IS degraded
    with patch.object(sm, 'check_data_pipeline_health', return_value=True), \
         patch.object(sm, 'monitor_data_drift', return_value=(True, {"patent_filing_rate_3m_patent": {"status": "DRIFT"}})) as mock_drift_check, \
         patch.object(sm, 'evaluate_model_performance_on_recent_data', return_value=False) as mock_perf_eval, \
         patch.object(sm, 'trigger_retraining') as mock_retrain_trigger, \
         caplog.at_level(logging.INFO): # Overall process logs start/end at INFO

        current_features_map = {sector: sample_features_df.copy()} # Current data has drift
        # Eval data: X_eval for MockModelForMonitor, y_eval
        X_eval_dummy = pd.DataFrame({'feat_1': [1,2], 'feat_2': [3,4]})
        y_eval_dummy = pd.Series([0.1, 0.2]) # Doesn't matter what values, perf_eval is mocked
        recent_eval_map = {sector: (X_eval_dummy, y_eval_dummy)}

        sm.run_scheduled_maintenance(current_features_map, recent_eval_map)

    mock_drift_check.assert_called_once_with(sample_features_df.copy()) # Check it was called with the right data
    assert f"Data drift detected for sector {sector}." in caplog.text # Log from run_scheduled_maintenance

    mock_perf_eval.assert_called_once_with(sector, X_eval_dummy, y_eval_dummy)
    # Retraining should be triggered because perf is False and auto_retraining_enabled is True
    mock_retrain_trigger.assert_called_once_with(sector)
    assert "Scheduled maintenance run completed." in caplog.text # Final log

```
