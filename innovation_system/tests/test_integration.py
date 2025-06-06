import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone # Ensure timezone is imported
from unittest.mock import patch, MagicMock
import os # For collectors that might save files

# Import necessary classes from the innovation_system
from innovation_system.data_collection import collectors
from innovation_system.feature_engineering import engineer
from innovation_system.config import settings # For default configs if needed

# --- Mock API Responses for Collectors (Simplified from unit tests) ---
SAMPLE_USPTO_API_RESPONSE = { # Represents USPTO Patent Data System (PDS) API V1 bulk search results
    "results": [{
        "patentApplicationNumber": "12345678", # Changed to match expected key from unit tests
        "inventionTitleText": "Test Patent 1",   # Changed to match expected key
        "filingDate": "2023-01-15",
        "patentGrantIdentification": {"grantDate": "2024-01-01"}, # Added for completeness if used
        "inventorNameArrayText": [{"inventorNameText": "Inventor A"}], # Changed to match expected key
        "assigneeEntityNameText": "Test Assignee Inc.", # Changed to match expected key
        "uspcClassification": [{"classificationSymbolText": "G06F"}], # Changed to match expected key
        "citation": [{"text": "Cited Patent 1"}] # Example citation data
    }]
}
SAMPLE_CRUNCHBASE_API_RESPONSE = { # Represents Crunchbase API funding rounds search
    "entities": [{
        "uuid": "round-uuid-1", "properties": {
            "money_raised": {"value_usd": 100000, "currency": "USD"},
            "announced_on": "2023-02-01",
            "funded_organization_identifier": {"uuid": "org-uuid-1", "value": "org-uuid-1", "name": "Innovatech"},
            "investment_type": "seed"
        }
    }], "count": 1
}

# Mock arxiv.Result structure (as defined in arxiv library)
class MockArxivResultIntegration:
    def __init__(self, entry_id, title, authors, summary, categories, published, pdf_url, comment=None, doi=None, journal_ref=None, primary_category=None, updated=None):
        self.entry_id = entry_id
        self.title = title
        self.authors = [MockArxivAuthor(name) for name in authors] # arxiv lib uses Author objects
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
# For PubMed, this structure is what _fetch_pubmed_details is expected to return (list of dicts)
SAMPLE_PUBMED_API_RESPONSE_INTEGRATION = [{
    'paper_id': 'pmid1', 'title': 'PubMed Paper 1', 'authors': ['AuthY'],
    'abstract': 'Abstract 2', 'published_date': (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
    'source': 'PubMed', 'url': 'https://pubmed.ncbi.nlm.nih.gov/pmid1/'
    # Removed citation_count as it's not directly fetched by _fetch_pubmed_details
}]


@pytest.fixture
def feature_engineer_instance():
    return engineer.FeatureEngineer(config=settings.feature_config)

# Patch file saving and directory creation globally for these integration tests
@patch('os.makedirs', MagicMock(return_value=None))
@patch('pandas.DataFrame.to_parquet', MagicMock(return_value=None))
def test_data_collection_to_feature_engineering_flow(feature_engineer_instance):
    # 1. Setup Collectors and Mock their API calls
    # Ensure API keys are mocked if collectors try to read them from settings/env
    # For this test, assume settings are configured with placeholder/mock keys if needed.
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
         patch('time.sleep', MagicMock(return_value=None)): # Mock sleep for rate limiting

        # --- Patent Collection Mock ---
        mock_uspto_response = MagicMock()
        mock_uspto_response.status_code = 200
        mock_uspto_response.json.return_value = SAMPLE_USPTO_API_RESPONSE

        # --- Funding Collection Mock ---
        mock_crunchbase_response = MagicMock()
        mock_crunchbase_response.status_code = 200
        mock_crunchbase_response.json.return_value = SAMPLE_CRUNCHBASE_API_RESPONSE
        mock_requests_post.return_value = mock_crunchbase_response

        # --- Research Collection Mocks ---
        mock_arxiv_instance = MagicMock()
        # The results method of arxiv.Client returns a generator of arxiv.Result objects
        mock_arxiv_instance.results.return_value = iter(SAMPLE_ARXIV_API_RESULTS)
        mock_arxiv_client_constructor.return_value = mock_arxiv_instance

        mock_pubmed_esearch_response = MagicMock()
        mock_pubmed_esearch_response.status_code = 200
        # PMIDs that would be used by _fetch_pubmed_details
        mock_pubmed_esearch_response.json.return_value = {"esearchresult": {"idlist": ["pmid1", "pmid2"]}}

        # requests.get needs to handle both USPTO and PubMed esearch calls
        def requests_get_side_effect(url, **kwargs):
            if "pds.uspto.gov/api/search" in url: # USPTO PDS API
                return mock_uspto_response
            if "eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi" in url: # PubMed eSearch
                return mock_pubmed_esearch_response
            return MagicMock(status_code=404)
        mock_requests_get.side_effect = requests_get_side_effect

        # Mock _fetch_pubmed_details directly to simplify PubMed data path
        with patch.object(research_collector, '_fetch_pubmed_details', return_value=SAMPLE_PUBMED_API_RESPONSE_INTEGRATION):
            start_date = datetime(2023,1,1)
            end_date = datetime(2023,1,31)
            days_back_research = 30

            collected_patents_df = patent_collector.collect_uspto_patents(start_date, end_date, "G01")
            collected_funding_df = funding_collector.collect_funding_rounds(start_date.strftime('%Y-%m-%d'), ["some_funding_cat_uuid"])

            arxiv_papers_list = research_collector.collect_arxiv_papers(categories=["cs.AI"], days_back=days_back_research)
            pubmed_papers_list = research_collector.collect_pubmed_papers(search_terms=["cancer"], days_back=days_back_research)

            # Ensure they are DataFrames before concat
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

    # 4. Run Feature Engineering
    # Ensure date columns are in expected format (datetime) before passing to feature engineer
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

    if not collected_research_df.empty: # Only assert if we had research data
        assert not research_features.empty, "Research features are empty"
        assert isinstance(research_features, pd.DataFrame)
        assert 'publication_rate_3m' in research_features.columns
    else: # If no research data collected (e.g. mocks returned empty)
        assert research_features.empty, "Research features should be empty if no research data"

```
