import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from innovation_system.feature_engineering import engineer
from innovation_system.config import settings # To access feature_config for comparison
import nltk # For nltk.downloader.DownloadError in test

# Mock NLTK download to prevent actual downloads during tests if resources are missing
# This should be at the top to patch before 'engineer' module is fully loaded if _ensure_nltk_resources is at module level
# However, engineer.py already calls _ensure_nltk_resources() at import time.
# So, for existing test runs, it would have already attempted to download.
# A more robust way is to ensure test environment has these or to structure code for easier mocking.
# For now, we assume the resources are there or the initial download in engineer.py was handled.

@pytest.fixture
def default_feature_engineer():
    return engineer.FeatureEngineer()

@pytest.fixture
def custom_feature_engineer():
    custom_config = {
        'emerging_tech_keywords': ['custom_keyword1', 'custom_keyword2'],
        'normalization_method': 'min_max', # Example custom setting
        # Add other config keys if FeatureEngineer uses them directly from self.feature_config
    }
    # Ensure all keys from settings.feature_config are present if the class expects them
    full_custom_config = settings.feature_config.copy()
    full_custom_config.update(custom_config)
    return engineer.FeatureEngineer(config=full_custom_config)

def test_feature_engineer_init_default(default_feature_engineer):
    assert default_feature_engineer.feature_config == settings.feature_config
    assert isinstance(default_feature_engineer.scaler, engineer.StandardScaler)

def test_feature_engineer_init_custom_config(custom_feature_engineer):
    assert 'custom_keyword1' in custom_feature_engineer.feature_config['emerging_tech_keywords']
    assert custom_feature_engineer.feature_config['normalization_method'] == 'min_max'
    # Depending on how FeatureEngineer handles scaler initialization based on config:
    # If 'normalization_method' in config dictates scaler type, this test needs adjustment.
    # Assuming FeatureEngineer always initializes a scaler, and might use 'normalization_method' later.
    # For now, engineer.py shows it always initializes StandardScaler.
    assert isinstance(custom_feature_engineer.scaler, engineer.StandardScaler)

# --- Tests for create_patent_features ---
def test_create_patent_features_empty_df(default_feature_engineer):
    empty_df = pd.DataFrame(columns=['filing_date', 'tech_class', 'inventors', 'citations', 'source', 'forward_citations'])
    features_df = default_feature_engineer.create_patent_features(empty_df)
    assert features_df.empty

def test_create_patent_features_structure(default_feature_engineer):
    # Test with minimal data to ensure all feature keys are present
    data = {
        'filing_date': [pd.Timestamp('2023-01-01') - timedelta(days=30)], # Use a fixed date for consistency
        'tech_class': ['G06F'],
        'inventors': [['Inventor A']],
        'citations': [5],
        'source': ['USPTO'], # Added to match expected columns in _calculate_international_ratio if used
        'forward_citations': [2]
    }
    patents_df = pd.DataFrame(data)

    with patch('pandas.Timestamp.now', return_value=pd.Timestamp('2023-01-01')) as mock_now, \
         patch.object(engineer.FeatureEngineer, '_calculate_event_rate', return_value=1.0) as mock_event_rate, \
         patch.object(engineer.FeatureEngineer, '_calculate_tech_diversity', return_value=0.5) as mock_tech_diversity, \
         patch.object(engineer.FeatureEngineer, '_calculate_citation_velocity', return_value=0.2) as mock_citation_vel, \
         patch.object(engineer.FeatureEngineer, '_calculate_international_ratio', return_value=0.1) as mock_intl_ratio:

        features_df = default_feature_engineer.create_patent_features(patents_df)

    assert not features_df.empty
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 1

    expected_cols = [
        'filing_rate_3m', 'filing_rate_6m', 'filing_rate_12m',
        'tech_diversity_shannon', 'unique_inventor_count',
        'citation_velocity_avg', 'forward_citation_rate_avg',
        'international_filing_ratio'
    ]
    for col in expected_cols:
        assert col in features_df.columns

    assert mock_event_rate.call_count == 3
    mock_tech_diversity.assert_called_once()
    mock_citation_vel.assert_called_once()
    mock_intl_ratio.assert_called_once()
    assert features_df['unique_inventor_count'].iloc[0] == 1
    assert features_df['forward_citation_rate_avg'].iloc[0] == 2.0


# --- Tests for create_funding_features ---
def test_create_funding_features_empty_df(default_feature_engineer):
    empty_df = pd.DataFrame(columns=['date', 'amount_usd', 'stage'])
    features_df = default_feature_engineer.create_funding_features(empty_df)
    assert features_df.empty

def test_create_funding_features_structure(default_feature_engineer):
    data = {
        'date': [pd.Timestamp('2023-01-01') - timedelta(days=60)], # Use a fixed date
        'amount_usd': [100000],
        'stage': ['series_a']
    }
    funding_df = pd.DataFrame(data)

    with patch('pandas.Timestamp.now', return_value=pd.Timestamp('2023-01-01')) as mock_now, \
         patch.object(engineer.FeatureEngineer, '_calculate_event_rate', return_value=0.5) as mock_event_rate, \
         patch.object(engineer.FeatureEngineer, '_calculate_sum_rate', return_value=50000) as mock_sum_rate, \
         patch.object(engineer.FeatureEngineer, '_calculate_gini_coefficient', return_value=0.3) as mock_gini:

        features_df = default_feature_engineer.create_funding_features(funding_df)

    assert not features_df.empty
    assert len(features_df) == 1
    expected_cols = [
        'funding_deals_velocity_3m', 'funding_deals_velocity_6m',
        'funding_amount_velocity_3m_usd', 'funding_amount_velocity_6m_usd',
        'avg_round_size_usd', 'median_round_size_usd', 'funding_amount_gini',
        'seed_ratio', 'series_a_ratio', 'late_stage_ratio'
    ]
    for col in expected_cols:
        assert col in features_df.columns

    assert mock_event_rate.call_count == 2
    assert mock_sum_rate.call_count == 2
    mock_gini.assert_called_once()
    assert features_df['avg_round_size_usd'].iloc[0] == 100000
    assert features_df['series_a_ratio'].iloc[0] == 1.0


# --- Tests for create_research_features ---
def test_create_research_features_empty_df(default_feature_engineer):
    empty_df = pd.DataFrame(columns=['published_date', 'citation_count', 'authors', 'categories', 'abstract'])
    features_df = default_feature_engineer.create_research_features(empty_df)
    assert features_df.empty

@patch('nltk.word_tokenize', return_value=['test', 'keyword'])
@patch('nltk.corpus.stopwords.words', return_value=['a', 'the'])
def test_create_research_features_structure(mock_stopwords, mock_tokenize, default_feature_engineer):
    data = {
        'published_date': [pd.Timestamp('2023-01-01') - timedelta(days=90)], # Use a fixed date
        'citation_count': [10],
        'authors': [['Author X', 'Author Y']],
        'categories': [['cs.AI']],
        'abstract': ['This is a test abstract with a test keyword.']
    }
    papers_df = pd.DataFrame(data)

    with patch('pandas.Timestamp.now', return_value=pd.Timestamp('2023-01-01')) as mock_now, \
         patch.object(engineer.FeatureEngineer, '_calculate_event_rate', return_value=0.3) as mock_event_rate, \
         patch.object(engineer.FeatureEngineer, '_calculate_tech_diversity', return_value=0.1) as mock_tech_diversity:

        original_config = default_feature_engineer.feature_config.copy() # Ensure deep copy if nested
        # Create a mutable copy for modification within the test
        test_specific_config = original_config.copy()
        test_specific_config['emerging_tech_keywords'] = ['keyword']
        default_feature_engineer.feature_config = test_specific_config

        features_df = default_feature_engineer.create_research_features(papers_df)

        default_feature_engineer.feature_config = original_config

    assert not features_df.empty
    assert len(features_df) == 1
    expected_cols = [
        'publication_rate_3m', 'publication_rate_6m',
        'avg_citation_count', 'avg_authors_per_paper', 'category_diversity_shannon',
        'avg_abstract_length', 'avg_keyword_count', 'sum_keyword_count'
    ]
    for col in expected_cols:
        assert col in features_df.columns

    assert mock_event_rate.call_count == 2
    mock_tech_diversity.assert_called_once()

    assert features_df['avg_citation_count'].iloc[0] == 10
    assert features_df['avg_authors_per_paper'].iloc[0] == 2
    assert features_df['avg_abstract_length'].iloc[0] == len('This is a test abstract with a test keyword.')
    assert features_df['avg_keyword_count'].iloc[0] == 1.0
    assert features_df['sum_keyword_count'].iloc[0] == 1.0


# --- Tests for Helper Methods ---

@patch('nltk.download')
@patch('nltk.data.find')
def test_ensure_nltk_resources_downloads_if_missing(mock_find, mock_download):
    mock_find.side_effect = nltk.downloader.DownloadError()
    engineer._ensure_nltk_resources()
    assert mock_download.call_count == 2
    mock_download.assert_any_call('stopwords', quiet=True)
    mock_download.assert_any_call('punkt', quiet=True)

@patch('nltk.download')
@patch('nltk.data.find', return_value=True)
def test_ensure_nltk_resources_no_download_if_present(mock_find, mock_download):
    engineer._ensure_nltk_resources()
    mock_download.assert_not_called()


def test_calculate_event_rate(default_feature_engineer):
    with patch('pandas.Timestamp.now', return_value=pd.Timestamp('2023-12-01')) as mock_now:
        test_df_frozen = pd.DataFrame({
            'event_date': pd.to_datetime([ # Ensure datetime objects
                '2023-11-20',
                '2023-10-15',
                '2023-07-01'
            ])
        })
        rate_3m = default_feature_engineer._calculate_event_rate(test_df_frozen, 'event_date', months=3)
        assert rate_3m == pytest.approx(2/3)

        rate_1m = default_feature_engineer._calculate_event_rate(test_df_frozen, 'event_date', months=1)
        assert rate_1m == pytest.approx(1/1)


def test_calculate_event_rate_empty_df(default_feature_engineer):
    empty_df = pd.DataFrame({'event_date': pd.to_datetime([])})
    rate = default_feature_engineer._calculate_event_rate(empty_df, 'event_date', months=3)
    assert rate == 0

def test_calculate_event_rate_missing_column(default_feature_engineer):
    test_df = pd.DataFrame({'some_other_date': pd.to_datetime([])}) # Ensure datetime type for consistency
    rate = default_feature_engineer._calculate_event_rate(test_df, 'event_date', months=3)
    assert rate == 0

# Add more tests for other helpers: _calculate_sum_rate, _calculate_tech_diversity, etc.

# Continuing tests for Helper Methods in FeatureEngineer

@pytest.fixture
def sample_patents_df():
    # Used for testing multiple helper methods
    now = pd.Timestamp('2023-12-01').tz_localize(None) # Fixed date for consistent tests
    return pd.DataFrame({
        'filing_date': pd.to_datetime([
            now - timedelta(days=30),
            now - timedelta(days=60),
            now - timedelta(days=90),
            now - timedelta(days=120),
            now - timedelta(days=200)
        ]),
        'tech_class': ['G06F', 'H04L', 'G06F', 'A61K', 'H04L'],
        'inventors': [['A', 'B'], ['C'], ['A', 'D', 'E'], ['F'], ['G', 'H']],
        'citations': [10, 5, 12, 3, 8],
        'forward_citations': [2, 1, 3, 0, 2],
        'source': ['USPTO', 'EPO', 'USPTO', 'WIPO', 'USPTO']
    })

@pytest.fixture
def sample_funding_df():
    now = pd.Timestamp('2023-12-01').tz_localize(None) # Fixed date
    return pd.DataFrame({
        'date': pd.to_datetime([
            now - timedelta(days=15),
            now - timedelta(days=45),
            now - timedelta(days=75),
            now - timedelta(days=150)
        ]),
        'amount_usd': [100000, 500000, 200000, 1000000],
        'stage': ['seed', 'series_a', 'seed', 'series_b']
    })

@pytest.fixture
def sample_papers_df():
    now = pd.Timestamp('2023-12-01').tz_localize(None) # Fixed date
    return pd.DataFrame({
        'published_date': pd.to_datetime([
            now - timedelta(days=20),
            now - timedelta(days=50),
            now - timedelta(days=80)
        ]),
        'citation_count': [5, 15, 3],
        'authors': [['X', 'Y'], ['Y', 'Z', 'W'], ['X']],
        'categories': [['cs.AI', 'cs.LG'], ['cs.LG', 'stat.ML'], ['cs.AI']],
        'abstract': [
            "This is about artificial intelligence and machine learning.",
            "Deep learning models for natural language processing.",
            "Another paper on AI applications."
        ]
    })


def test_calculate_sum_rate(default_feature_engineer, sample_funding_df):
    fixed_now = pd.Timestamp('2023-12-01') # Match sample_funding_df's 'now'
    with patch('pandas.Timestamp.now', return_value=fixed_now) as mock_now:

        sum_rate_3m = default_feature_engineer._calculate_sum_rate(sample_funding_df, 'date', 'amount_usd', months=3)
        # Dates in sample_funding_df relative to 2023-12-01:
        # 2023-11-16 (15 days ago) -> 100k
        # 2023-10-17 (45 days ago) -> 500k
        # 2023-09-17 (75 days ago) -> 200k
        # All three are within 3 months of 2023-12-01.
        # 2023-07-04 (150 days ago) -> 1000k (outside 3 months)
        expected_sum_3m = 100000 + 500000 + 200000
        assert sum_rate_3m == pytest.approx(expected_sum_3m / 3)

        sum_rate_1m = default_feature_engineer._calculate_sum_rate(sample_funding_df, 'date', 'amount_usd', months=1)
        # Only 2023-11-16 (100k) is within 1 month of 2023-12-01
        expected_sum_1m = 100000
        assert sum_rate_1m == pytest.approx(expected_sum_1m / 1)


def test_calculate_sum_rate_empty_df(default_feature_engineer):
    empty_df = pd.DataFrame({'date': pd.to_datetime([]), 'value': []})
    rate = default_feature_engineer._calculate_sum_rate(empty_df, 'date', 'value', months=3)
    assert rate == 0

def test_calculate_tech_diversity_patents(default_feature_engineer, sample_patents_df):
    p_g06f = 2/5; p_h04l = 2/5; p_a61k = 1/5
    expected_diversity = -(p_g06f * np.log2(p_g06f) + p_h04l * np.log2(p_h04l) + p_a61k * np.log2(p_a61k))
    diversity = default_feature_engineer._calculate_tech_diversity(sample_patents_df, 'tech_class')
    assert diversity == pytest.approx(expected_diversity)

def test_calculate_tech_diversity_papers_exploded(default_feature_engineer, sample_papers_df):
    # Exploded categories: cs.AI (2), cs.LG (2), stat.ML (1). Total 5.
    p_ai = 2/5; p_lg = 2/5; p_ml = 1/5
    expected_diversity = -(p_ai * np.log2(p_ai) + p_lg * np.log2(p_lg) + p_ml * np.log2(p_ml))
    diversity = default_feature_engineer._calculate_tech_diversity(sample_papers_df, 'categories', explode_list=True)
    assert diversity == pytest.approx(expected_diversity)

def test_calculate_tech_diversity_empty(default_feature_engineer):
    empty_df = pd.DataFrame({'category': []})
    assert default_feature_engineer._calculate_tech_diversity(empty_df, 'category') == 0
    nan_df = pd.DataFrame({'category': [np.nan, None, []]}) # Added empty list for explode_list robustness
    assert default_feature_engineer._calculate_tech_diversity(nan_df, 'category', explode_list=True) == 0


def test_calculate_citation_velocity(default_feature_engineer, sample_patents_df):
    mock_now_ts = pd.Timestamp('2023-12-01') # Match sample_patents_df's 'now'

    with patch('pandas.Timestamp.now', return_value=mock_now_ts):
        df_copy = sample_patents_df.copy()
        # Ensure filing_date is tz-naive for consistent subtraction with mock_now_ts (which is naive)
        df_copy['filing_date'] = pd.to_datetime(df_copy['filing_date']).dt.tz_localize(None)

        expected_rates = []
        for idx, row in df_copy.iterrows():
            # Age in months. Ensure positive and non-zero.
            age_in_days = (mock_now_ts - row['filing_date']).days
            months_since = age_in_days / 30.44  # Approximate days in a month
            months_since = max(months_since, 1) # As per implementation detail (age_months = max(1, age_months))
            expected_rates.append(row['citations'] / months_since)

        expected_avg_velocity = np.mean(expected_rates)
        # Pass the original sample_patents_df which has naive datetimes by fixture design
        velocity = default_feature_engineer._calculate_citation_velocity(sample_patents_df)
        assert velocity == pytest.approx(expected_avg_velocity)

def test_calculate_citation_velocity_empty(default_feature_engineer):
    empty_df = pd.DataFrame({'filing_date': pd.to_datetime([]), 'citations': []})
    assert default_feature_engineer._calculate_citation_velocity(empty_df) == 0


def test_calculate_international_ratio(default_feature_engineer, sample_patents_df):
    ratio = default_feature_engineer._calculate_international_ratio(sample_patents_df, 'source')
    assert ratio == pytest.approx(0.4) # EPO (1) + WIPO (1) / Total (5)

def test_calculate_international_ratio_all_uspto(default_feature_engineer):
    all_us_df = pd.DataFrame({'source': ['USPTO', 'US', 'USPTO']})
    ratio = default_feature_engineer._calculate_international_ratio(all_us_df, 'source')
    assert ratio == 0.0

def test_calculate_international_ratio_empty(default_feature_engineer):
    empty_df = pd.DataFrame({'source': []})
    assert default_feature_engineer._calculate_international_ratio(empty_df, 'source') == 0


def test_calculate_gini_coefficient(default_feature_engineer):
    series_equal = pd.Series([5, 5, 5, 5, 5])
    assert default_feature_engineer._calculate_gini_coefficient(series_equal) == pytest.approx(0.0)

    series_unequal = pd.Series([1, 2, 3, 6, 10])
    assert default_feature_engineer._calculate_gini_coefficient(series_unequal) == pytest.approx(0.4)

    series_with_zero = pd.Series([0, 0, 0, 10])
    assert default_feature_engineer._calculate_gini_coefficient(series_with_zero) == pytest.approx(0.75)

def test_calculate_gini_coefficient_empty_or_single(default_feature_engineer):
    assert default_feature_engineer._calculate_gini_coefficient(pd.Series([], dtype=float)) == 0 # ensure dtype for empty
    assert default_feature_engineer._calculate_gini_coefficient(pd.Series([10])) == 0
    assert default_feature_engineer._calculate_gini_coefficient(pd.Series([np.nan, np.nan], dtype=float)) == 0


@patch('nltk.word_tokenize')
@patch('nltk.corpus.stopwords.words', return_value=['is', 'a', 'the', 'for', 'on', 'and', 'this', 'about', '.']) # Expanded stopwords
def test_research_keyword_counting(mock_stopwords, mock_tokenize, default_feature_engineer, sample_papers_df):

    def side_effect_tokenize(text):
        # Simple split and lower, actual tokenization is more complex
        return [token.strip(".").lower() for token in text.split()]
    mock_tokenize.side_effect = side_effect_tokenize

    # Config with single-token keywords, matching sample_papers_df content
    custom_config_dict = default_feature_engineer.feature_config.copy()
    custom_config_dict['emerging_tech_keywords'] = ['intelligence', 'learning', 'ai', 'language'] # Added 'language'

    # Create a new engineer instance with this specific config for this test
    engineer_custom_nlp = engineer.FeatureEngineer(config=custom_config_dict)


    # Mock other calculations to isolate keyword counting impact
    with patch.object(engineer.FeatureEngineer, '_calculate_event_rate', return_value=0.0), \
         patch.object(engineer.FeatureEngineer, '_calculate_tech_diversity', return_value=0.0), \
         patch('pandas.Timestamp.now', return_value=pd.Timestamp('2023-12-01')): # Match sample data 'now'

        # Pass a copy of sample_papers_df to avoid side effects if the method modifies it
        features_df_single_kw = engineer_custom_nlp.create_research_features(sample_papers_df.copy())

    # Expected processed tokens and keyword counts:
    # Abs 1: "This is about artificial intelligence and machine learning."
    #        Tokens (mocked): ["this", "is", "about", "artificial", "intelligence", "and", "machine", "learning"]
    #        Post-stopwords: ["artificial", "intelligence", "machine", "learning"]
    #        Keywords: intelligence (1), learning (1) => Total 2
    # Abs 2: "Deep learning models for natural language processing."
    #        Tokens: ["deep", "learning", "models", "for", "natural", "language", "processing"]
    #        Post-stopwords: ["deep", "learning", "models", "natural", "language", "processing"]
    #        Keywords: learning (1), language (1) => Total 2
    # Abs 3: "Another paper on AI applications."
    #        Tokens: ["another", "paper", "on", "ai", "applications"]
    #        Post-stopwords: ["another", "paper", "ai", "applications"]
    #        Keywords: ai (1) => Total 1

    expected_sum_keyword_count = 2 + 2 + 1 # = 5
    expected_avg_keyword_count = expected_sum_keyword_count / 3

    assert features_df_single_kw['sum_keyword_count'].iloc[0] == pytest.approx(expected_sum_keyword_count)
    assert features_df_single_kw['avg_keyword_count'].iloc[0] == pytest.approx(expected_avg_keyword_count)


    empty_abstract_df = pd.DataFrame({
        'published_date': [pd.Timestamp('2023-12-01')], 'citation_count': [0],
        'authors': [['A']], 'categories': [['cat']], 'abstract': [""]
    })
    # Need to mock Timestamp.now() for this call too if _calculate_event_rate is not fully mocked out
    with patch('pandas.Timestamp.now', return_value=pd.Timestamp('2023-12-01')):
        features_empty_abstract = engineer_custom_nlp.create_research_features(empty_abstract_df)
    assert features_empty_abstract['avg_keyword_count'].iloc[0] == 0
    assert features_empty_abstract['sum_keyword_count'].iloc[0] == 0

    no_kw_config_dict = default_feature_engineer.feature_config.copy()
    no_kw_config_dict['emerging_tech_keywords'] = []
    engineer_no_kw = engineer.FeatureEngineer(config=no_kw_config_dict)
    with patch('pandas.Timestamp.now', return_value=pd.Timestamp('2023-12-01')):
        features_no_kw = engineer_no_kw.create_research_features(sample_papers_df.copy())
    assert features_no_kw['avg_keyword_count'].iloc[0] == 0
    assert features_no_kw['sum_keyword_count'].iloc[0] == 0

```
