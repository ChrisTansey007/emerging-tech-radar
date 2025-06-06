import os
import pytest
from dotenv import load_dotenv
import importlib

# Import the settings module. This will execute settings.py.
# This assumes that the 'innovation_system' package is installed (e.g., editable mode)
# or that PYTHONPATH is set up correctly so that 'config.settings' can be found.
from innovation_system.config import settings

@pytest.fixture(autouse=True)
def manage_environment_and_reload_settings(monkeypatch):
    """
    Fixture to clear relevant environment variables before each test
    and ensure settings are reloaded to reflect changes.
    It also restores original environment variables after the test.
    Using autouse=True to apply this to all tests in this module.
    """
    env_vars_to_manage = ["USPTO_API_KEY", "CRUNCHBASE_API_KEY", "PUBMED_API_KEY"]
    original_values = {var: os.environ.get(var) for var in env_vars_to_manage}

    for var in env_vars_to_manage:
        monkeypatch.delenv(var, raising=False)

    # Reload settings initially to reflect cleared env vars for default tests
    importlib.reload(settings)

    yield monkeypatch # Test runs here, can use monkeypatch to set new env vars

    # Restore original environment variables
    for var, value in original_values.items():
        if value is not None:
            monkeypatch.setenv(var, value)
        else:
            monkeypatch.delenv(var, raising=False) # Ensure it's unset if it was originally unset

    # Reload settings again to clean up state for subsequent test files (if any)
    importlib.reload(settings)


def test_api_keys_loaded_from_env(manage_environment_and_reload_settings):
    monkeypatch = manage_environment_and_reload_settings # Get the monkeypatch fixture

    # Set mock environment variables
    monkeypatch.setenv("USPTO_API_KEY", "test_uspto_key")
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "test_crunchbase_key")
    monkeypatch.setenv("PUBMED_API_KEY", "test_pubmed_key")

    # Reload settings to pick up the new environment variables
    importlib.reload(settings)

    assert settings.USPTO_API_KEY == "test_uspto_key"
    assert settings.CRUNCHBASE_API_KEY == "test_crunchbase_key"
    assert settings.PUBMED_API_KEY == "test_pubmed_key"

def test_api_keys_default_placeholders():
    # Env vars are cleared by the autouse fixture, settings reloaded
    assert settings.USPTO_API_KEY == "USPTO_DEFAULT_PLACEHOLDER"
    assert settings.CRUNCHBASE_API_KEY == "CRUNCHBASE_DEFAULT_PLACEHOLDER"
    assert settings.PUBMED_API_KEY == "PUBMED_DEFAULT_PLACEHOLDER"

def test_patent_config_structure():
    assert isinstance(settings.patent_config, dict)
    assert 'collection_frequency' in settings.patent_config
    assert 'tech_categories' in settings.patent_config
    assert isinstance(settings.patent_config['tech_categories'], list)
    assert 'lookback_days' in settings.patent_config
    assert isinstance(settings.patent_config['lookback_days'], int)

def test_funding_config_structure():
    assert isinstance(settings.funding_config, dict)
    assert 'categories' in settings.funding_config
    assert isinstance(settings.funding_config['categories'], list)
    assert 'min_amount_usd' in settings.funding_config
    assert isinstance(settings.funding_config['min_amount_usd'], int)

def test_research_config_structure():
    assert isinstance(settings.research_config, dict)
    assert 'arxiv_categories' in settings.research_config
    assert isinstance(settings.research_config['arxiv_categories'], list)
    assert 'pubmed_terms' in settings.research_config

def test_feature_config_structure():
    assert isinstance(settings.feature_config, dict)
    assert 'patent_weights' in settings.feature_config
    assert isinstance(settings.feature_config['patent_weights'], dict)
    assert 'normalization_method' in settings.feature_config

def test_model_config_structure():
    assert isinstance(settings.model_config, dict)
    assert 'prediction_horizons' in settings.model_config
    assert isinstance(settings.model_config['prediction_horizons'], list)
    assert 'validation_method' in settings.model_config

def test_prediction_config_structure():
    assert isinstance(settings.prediction_config, dict)
    assert 'update_frequency' in settings.prediction_config
    assert 'confidence_thresholds' in settings.prediction_config
    assert isinstance(settings.prediction_config['confidence_thresholds'], dict)

def test_monitoring_config_structure():
    assert isinstance(settings.monitoring_config, dict)
    assert 'log_file' in settings.monitoring_config
    assert 'alert_email_recipient' in settings.monitoring_config

def test_uncertainty_config_structure():
    assert isinstance(settings.uncertainty_config, dict)
    assert 'confidence_thresholds' in settings.uncertainty_config
    assert settings.uncertainty_config['confidence_thresholds'] == settings.prediction_config['confidence_thresholds']

def test_dotenv_loading_if_file_exists(manage_environment_and_reload_settings, tmp_path):
    monkeypatch = manage_environment_and_reload_settings

    dotenv_content = "USPTO_API_KEY=env_file_uspto_key\nCRUNCHBASE_API_KEY=env_file_crunchbase_key"
    env_file_path = tmp_path / ".env"
    env_file_path.write_text(dotenv_content)

    load_dotenv(dotenv_path=env_file_path, override=True)

    importlib.reload(settings)

    assert settings.USPTO_API_KEY == "env_file_uspto_key"
    assert settings.CRUNCHBASE_API_KEY == "env_file_crunchbase_key"
    assert settings.PUBMED_API_KEY == "PUBMED_DEFAULT_PLACEHOLDER"
