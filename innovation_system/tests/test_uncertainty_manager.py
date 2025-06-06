import pytest
import logging
import numpy as np # For np.nan
import pandas as pd # For pd.isna, though not used in this initial set

from innovation_system.uncertainty_handling import manager

@pytest.fixture
def mock_uncertainty_config():
    return {
        'confidence_thresholds': {'high': 0.75, 'medium': 0.50, 'low': 0.25}, # Custom for test
        'min_data_completeness_for_high_confidence': 0.8,
        'min_paper_coverage_for_high_confidence': 0.6,
        # Add other relevant keys if methods directly use them
    }

@pytest.fixture
def uncertainty_manager_instance(mock_uncertainty_config):
    return manager.UncertaintyManager(mock_uncertainty_config)

def test_uncertainty_manager_init(uncertainty_manager_instance, mock_uncertainty_config):
    um = uncertainty_manager_instance
    assert um.config == mock_uncertainty_config
    assert um.confidence_thresholds == mock_uncertainty_config['confidence_thresholds']

def test_assess_data_completeness_impact_high(uncertainty_manager_instance, caplog):
    um = uncertainty_manager_instance
    metric = 0.9 # High completeness
    # Ensure no logs at INFO or higher for this case
    with caplog.at_level(logging.INFO):
        adj, msg = um.assess_data_completeness_impact("TestSource", metric, "TestSector")

    assert adj == 0.0
    assert msg == "" # No specific message for high completeness
    assert not caplog.records # No logs expected for high completeness

def test_assess_data_completeness_impact_medium(uncertainty_manager_instance, caplog):
    um = uncertainty_manager_instance
    metric = 0.7
    # Method logs at INFO for medium caution
    with caplog.at_level(logging.INFO):
        adj, msg = um.assess_data_completeness_impact("TestSource", metric, "TestSector")

    assert adj == -0.10
    assert "Caution (TestSector): TestSource data incomplete (70.0%). Confidence reduced." in msg
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "INFO" # Check log level


def test_assess_data_completeness_impact_low(uncertainty_manager_instance, caplog):
    um = uncertainty_manager_instance
    metric = 0.3
    # Method logs at WARNING for low completeness
    with caplog.at_level(logging.WARNING):
        adj, msg = um.assess_data_completeness_impact("TestSource", metric, "TestSector")

    assert adj == -0.20
    assert "Warning (TestSector): TestSource data very incomplete (30.0%). Confidence significantly reduced." in msg
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"


def test_handle_conflicting_signals_consistent(uncertainty_manager_instance, caplog):
    um = uncertainty_manager_instance
    signals = {"model1": 0.8, "model2": 0.7, "data_source_A": 0.6}
    with caplog.at_level(logging.INFO): # Method logs consistency at INFO
        adj, msg, scenario_info = um.handle_conflicting_signals("TestSector", signals)

    assert adj == 0.0
    assert "Signals for 'TestSector' generally consistent." in msg
    assert scenario_info['status'] == 'consistent'
    assert scenario_info['scenarios_suggested'] is False
    assert len(caplog.records) == 1 # One INFO log for consistency


def test_handle_conflicting_signals_conflicting(uncertainty_manager_instance, caplog):
    um = uncertainty_manager_instance
    signals = {"model1": 0.9, "model2": -0.7, "data_source_A": 0.8, "data_source_B": -0.6}
    with caplog.at_level(logging.WARNING): # Conflict logs at WARNING
        adj, msg, scenario_info = um.handle_conflicting_signals("TestSector", signals)

    assert adj == -0.15
    assert "Conflicting signals for 'TestSector'." in msg
    # Using string containment for set representation as order can vary
    assert "Positive signals from: {'model1', 'data_source_A'}." in msg or \
           "Positive signals from: {'data_source_A', 'model1'}." in msg
    assert "Negative signals from: {'model2', 'data_source_B'}." in msg or \
           "Negative signals from: {'data_source_B', 'model2'}." in msg
    assert scenario_info['status'] == 'conflicting'
    assert scenario_info['scenarios_suggested'] is True
    assert len(caplog.records) == 1


def test_handle_conflicting_signals_no_strong_signals(uncertainty_manager_instance, caplog): # Added caplog
    um = uncertainty_manager_instance
    signals = {"model1": 0.2, "model2": -0.3}
    with caplog.at_level(logging.INFO): # Consistency (due to no strong signals) logs at INFO
        adj, msg, scenario_info = um.handle_conflicting_signals("TestSector", signals)

    assert adj == 0.0
    assert "Signals for 'TestSector' generally consistent." in msg
    assert scenario_info['status'] == 'consistent'
    assert len(caplog.records) == 1 # One INFO log


def test_acknowledge_research_pipeline_gap_high_coverage(uncertainty_manager_instance, caplog):
    um = uncertainty_manager_instance
    coverage = 0.9
    with caplog.at_level(logging.INFO):
        adj, msg = um.acknowledge_research_pipeline_gap("TestSector", coverage)

    assert adj == 0.0
    assert msg == "" # No specific message for high coverage
    assert not caplog.records # No logs expected for high coverage


def test_acknowledge_research_pipeline_gap_medium_coverage(uncertainty_manager_instance, caplog):
    um = uncertainty_manager_instance
    coverage = 0.5
    with caplog.at_level(logging.INFO): # Medium logs at INFO
        adj, msg = um.acknowledge_research_pipeline_gap("TestSector", coverage)

    assert adj == -0.05
    assert "Caution (TestSector): Moderate research visibility gap (50.0%). Caution with research insights." in msg
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "INFO"


def test_acknowledge_research_pipeline_gap_low_coverage(uncertainty_manager_instance, caplog):
    um = uncertainty_manager_instance
    coverage = 0.2
    with caplog.at_level(logging.WARNING): # Low logs at WARNING
        adj, msg = um.acknowledge_research_pipeline_gap("TestSector", coverage)

    assert adj == -0.15
    assert "Warning (TestSector): Research visibility gap - sparse papers (20.0%). Low research insight reliability." in msg
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"

# Continuing tests for UncertaintyManager

def test_format_final_prediction_output_high_confidence(uncertainty_manager_instance, caplog):
    um = uncertainty_manager_instance # Uses thresholds: H:0.75, M:0.50, L:0.25
    pred_value = 0.88
    base_confidence = 0.70
    adjustments = [0.0, 0.10, -0.02] # Net +0.08. Final = 0.70 + 0.08 = 0.78 (High)
    intervals = {'68%': [0.80, 0.95], '95%': [0.75, 1.05123]}

    with caplog.at_level(logging.INFO): # Method logs final output at INFO
        output = um.format_final_prediction_output("TestSector", pred_value, base_confidence, adjustments, intervals)

    assert output['prediction_value'] == pred_value
    assert output['final_confidence_score'] == pytest.approx(0.78)
    assert output['confidence_label'] == "High"
    assert "High confidence. Subject to unforeseen events." in output['disclaimer']
    assert output['confidence_interval_str'] == "[0.750, 1.051] (95% CI)" # Prefers 95%
    assert f"Final Output for TestSector - Pred: {pred_value:.3f}, Conf: High (0.78), Interval: [0.750, 1.051] (95% CI)" in caplog.text
    assert len(caplog.records) == 1

def test_format_final_prediction_output_medium_confidence(uncertainty_manager_instance, caplog): # Added caplog
    um = uncertainty_manager_instance # H:0.75, M:0.50, L:0.25
    pred_value = 0.55
    base_confidence = 0.60
    adjustments = [-0.05, -0.03] # Net -0.08. Final = 0.60 - 0.08 = 0.52 (Medium)
    intervals_68_only = {'68%': [0.40, 0.70]} # Only 68% CI provided

    with caplog.at_level(logging.INFO):
        output = um.format_final_prediction_output("TestSector", pred_value, base_confidence, adjustments, intervals_68_only)

    assert output['final_confidence_score'] == pytest.approx(0.52)
    assert output['confidence_label'] == "Medium"
    assert "Medium confidence. Corroborate with other sources." in output['disclaimer']
    assert output['confidence_interval_str'] == "[0.400, 0.700] (68% CI)"
    assert f"Final Output for TestSector - Pred: {pred_value:.3f}, Conf: Medium (0.52), Interval: [0.400, 0.700] (68% CI)" in caplog.text
    assert len(caplog.records) == 1


def test_format_final_prediction_output_low_confidence_no_interval(uncertainty_manager_instance, caplog): # Added caplog
    um = uncertainty_manager_instance # H:0.75, M:0.50, L:0.25
    pred_value = 0.12
    base_confidence = 0.40
    adjustments = [-0.20] # Final = 0.20 (Low)

    with caplog.at_level(logging.INFO):
        output = um.format_final_prediction_output("TestSector", pred_value, base_confidence, adjustments, None) # No intervals

    assert output['final_confidence_score'] == pytest.approx(0.20)
    assert output['confidence_label'] == "Low"
    assert "Low confidence. Interpret with extreme caution." in output['disclaimer']
    assert output['confidence_interval_str'] == "N/A"
    assert f"Final Output for TestSector - Pred: {pred_value:.3f}, Conf: Low (0.20), Interval: N/A" in caplog.text
    assert len(caplog.records) == 1


def test_format_final_prediction_output_confidence_clipping(uncertainty_manager_instance):
    um = uncertainty_manager_instance
    # Test clipping at 1.0
    output_high = um.format_final_prediction_output("TestSector", 1.0, 0.9, [0.3], None) # 0.9 + 0.3 = 1.2 -> 1.0
    assert output_high['final_confidence_score'] == 1.0
    assert output_high['confidence_label'] == "High"

    # Test clipping at 0.0
    output_low = um.format_final_prediction_output("TestSector", 0.1, 0.1, [-0.5], None) # 0.1 - 0.5 = -0.4 -> 0.0
    assert output_low['final_confidence_score'] == 0.0
    assert output_low['confidence_label'] == "Low"


def test_format_final_prediction_output_nan_interval(uncertainty_manager_instance):
    um = uncertainty_manager_instance
    intervals_with_nan = {'95%': [np.nan, 0.9]}
    output = um.format_final_prediction_output("TestSector", 0.5, 0.7, [], intervals_with_nan)
    assert output['confidence_interval_str'] == "N/A"

    intervals_with_nan_2 = {'95%': [0.1, np.nan]}
    output2 = um.format_final_prediction_output("TestSector", 0.5, 0.7, [], intervals_with_nan_2)
    assert output2['confidence_interval_str'] == "N/A"


def test_recommend_alternative_sources_or_methods(uncertainty_manager_instance, caplog):
    um = uncertainty_manager_instance
    source = "Magic8Ball"
    sector = "FutureTech"
    issue = "Consistently vague predictions"

    with caplog.at_level(logging.WARNING): # Method logs at WARNING
        message = um.recommend_alternative_sources_or_methods(source, sector, issue)

    expected_log_msg_part1 = f"Issue with '{source}' for '{sector}': {issue}."
    expected_log_msg_part2 = "Consider: [Expert Interviews], [Market Reports], [Analogous Sector Analysis], [Scenario Planning]."

    assert expected_log_msg_part1 in message
    assert expected_log_msg_part2 in message

    full_expected_message_in_log = f"{expected_log_msg_part1} {expected_log_msg_part2}"
    found_log = any(full_expected_message_in_log == record.message for record in caplog.records if record.levelname == "WARNING")
    assert found_log, "Expected recommendation warning not found in logs or format mismatch."
    assert len(caplog.records) == 1

```
