import pytest
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from unittest.mock import patch, MagicMock, mock_open

# Make sure innovation_system is in path or package is installed
from innovation_system.data_collection import collectors
from innovation_system.config import settings # To allow reloading/mocking if necessary

# Sample data for mocking API responses
SAMPLE_USPTO_RESPONSE_VALID = {
    "results": [
        {
            "applicationNumberText": "12345678",
            "inventionTitle": "Test Patent 1",
            "filingDate": "2023-01-15",
            "inventors": [{"fullName": "Inventor A"}],
            "assigneeEntityName": "Test Assignee Inc.",
            "mainClassificationSymbol": "G06F",
            "referencedBy": [{}, {}] # Simulating 2 citations
        },
        {
            "patentNumber": "US9876543B2", # Different ID key
            "inventionTitle": "Test Patent 2",
            "filingDate": "2023-02-20",
            "inventors": [{"fullName": "Inventor B"}, {"fullName": "Inventor C"}],
            "applicantName": "Another Assignee LLC", # Different assignee key
            "classificationNationalCurrent": {"classificationSymbolText": {"text": "H04L"}}, # Different classification key
            "referencedBy": []
        }
    ]
}

SAMPLE_USPTO_RESPONSE_EMPTY = {"results": []}

SAMPLE_USPTO_MALFORMED_RESPONSE = {"unexpected_key": "some_value"}


@pytest.fixture
def patent_collector():
    # If API keys are used directly by collector methods, ensure they are set for tests
    # Here, USPTO_API_KEY is imported but not directly used in the current collect_uspto_patents
    return collectors.PatentDataCollector()

@patch('requests.get')
def test_collect_uspto_patents_success(mock_get, patent_collector):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = SAMPLE_USPTO_RESPONSE_VALID
    mock_get.return_value = mock_response

    # Mock os.makedirs and pd.DataFrame.to_parquet
    with patch('os.makedirs') as mock_makedirs, \
         patch('pandas.DataFrame.to_parquet') as mock_to_parquet:

        patents = patent_collector.collect_uspto_patents(
            datetime(2023, 1, 1), datetime(2023, 1, 31), "G06F"
        )

    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert patent_collector.uspto_base_url in args[0]
    assert kwargs['params']['searchText'] == 'ccl=(G06F)'

    assert len(patents) == 2
    assert patents[0]['patent_id'] == "12345678"
    assert patents[1]['title'] == "Test Patent 2"
    assert patents[0]['inventors'] == 1 # Length of inventors list
    assert patents[1]['citations'] == 0 # Length of referencedBy list

    mock_makedirs.assert_called_with("data/raw", exist_ok=True)
    mock_to_parquet.assert_called_once()
    # Further checks on the DataFrame passed to to_parquet can be added if needed


@patch('requests.get')
def test_collect_uspto_patents_api_error(mock_get, patent_collector, capsys):
    mock_get.side_effect = requests.RequestException("API down")

    patents = patent_collector.collect_uspto_patents(
        datetime(2023, 1, 1), datetime(2023, 1, 31), "G06F"
    )
    assert patents == []
    captured = capsys.readouterr()
    assert "USPTO API error: API down" in captured.out

@patch('requests.get')
def test_collect_uspto_patents_json_error(mock_get, patent_collector, capsys):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = json.JSONDecodeError("err", "doc", 0)
    mock_response.text = "Invalid JSON"
    mock_get.return_value = mock_response

    patents = patent_collector.collect_uspto_patents(
        datetime(2023, 1, 1), datetime(2023, 1, 31), "G06F"
    )
    assert patents == []
    captured = capsys.readouterr()
    assert "USPTO API JSON parsing error" in captured.out
    assert "Response: Invalid JSON" in captured.out


def test_parse_uspto_response_valid(patent_collector):
    parsed = patent_collector._parse_uspto_response(SAMPLE_USPTO_RESPONSE_VALID)
    assert len(parsed) == 2
    assert parsed[0]['patent_id'] == "12345678"
    assert parsed[0]['title'] == "Test Patent 1"
    assert parsed[0]['filing_date'] == "2023-01-15"
    assert parsed[0]['inventors'] == 1
    assert parsed[0]['assignee'] == "Test Assignee Inc."
    assert parsed[0]['tech_class'] == "G06F"
    assert parsed[0]['citations'] == 2
    assert parsed[0]['source'] == "USPTO"

    assert parsed[1]['patent_id'] == "US9876543B2"
    assert parsed[1]['assignee'] == "Another Assignee LLC"
    assert parsed[1]['tech_class'] == "H04L"


def test_parse_uspto_response_empty(patent_collector):
    parsed = patent_collector._parse_uspto_response(SAMPLE_USPTO_RESPONSE_EMPTY)
    assert parsed == []

def test_parse_uspto_response_malformed(patent_collector):
    # Depending on implementation, this might raise an error or return empty list
    # Current _parse_uspto_response is robust to missing keys and returns empty list for 'results'
    parsed = patent_collector._parse_uspto_response(SAMPLE_USPTO_MALFORMED_RESPONSE)
    assert parsed == [] # Because 'results' key is missing

def test_validate_patent_data_valid_and_invalid(patent_collector, capsys):
    patents_to_validate = [
        {'patent_id': '1', 'filing_date': '2023-01-01', 'title': 'Valid Patent'},
        {'patent_id': '2', 'filing_date': 'bad-date', 'title': 'Invalid Date Patent'},
        {'patent_id': None, 'filing_date': '2023-01-03', 'title': 'Missing ID Patent'},
        {'patent_id': '4', 'filing_date': '2023-01-04', 'title': None}, # Missing title
    ]

    # Mock os.makedirs and pd.DataFrame.to_parquet as they are called in _validate_patent_data
    with patch('os.makedirs') as mock_makedirs, \
         patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
        validated = patent_collector._validate_patent_data(patents_to_validate)

    assert len(validated) == 1
    assert validated[0]['patent_id'] == '1'

    captured = capsys.readouterr()
    assert "Invalid filing date format for patent ID 2: bad-date" in captured.out
    assert "Missing critical data for patent: None" in captured.out # For patent with None ID
    assert "Missing critical data for patent: 4" in captured.out # For patent with missing title

    mock_makedirs.assert_called_with("data/raw", exist_ok=True)
    mock_to_parquet.assert_called_once()


def test_validate_patent_data_empty_input(patent_collector, capsys):
     with patch('os.makedirs') as mock_makedirs, \
          patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
        validated = patent_collector._validate_patent_data([])

    assert validated == []
    captured = capsys.readouterr()
    assert "Validated 0/0 patents" in captured.out
    # Parquet saving should ideally not be called if validated list is empty
    mock_to_parquet.assert_not_called()


def test_validate_patent_data_saves_to_parquet(patent_collector):
    valid_patents = [
        {'patent_id': '123', 'filing_date': '2023-01-01', 'title': 'Test'},
    ]
    with patch('os.makedirs') as mock_makedirs, \
         patch('pandas.DataFrame') as mock_pd_dataframe_constructor, \
         patch('pandas.DataFrame.to_parquet') as mock_to_parquet_method:

        # Mock the DataFrame instance that would be created
        mock_df_instance = MagicMock()
        mock_pd_dataframe_constructor.return_value = mock_df_instance
        mock_df_instance.empty = False # Simulate non-empty DataFrame

        patent_collector._validate_patent_data(valid_patents)

        mock_makedirs.assert_called_once_with("data/raw", exist_ok=True)
        mock_pd_dataframe_constructor.assert_called_once_with(valid_patents)
        mock_to_parquet_method.assert_called_once_with(os.path.join("data/raw", "patents.parquet"), index=False)

def test_validate_patent_data_no_save_for_empty_df(patent_collector):
    # Test that if the DataFrame is empty (e.g., after validation nothing remains),
    # to_parquet is not called, or handled as per implementation.
    # The current _validate_patent_data saves if the list is not empty.
    # If the list becomes empty after validation, df.to_parquet might be called on an empty df.
    # Let's assume the implementation means if 'valid_patents' list is empty, don't save.

    with patch('os.makedirs') as mock_makedirs, \
         patch('pandas.DataFrame.to_parquet') as mock_to_parquet:

        patent_collector._validate_patent_data([]) # Empty list of patents

        mock_makedirs.assert_called_once_with("data/raw", exist_ok=True) # makedirs might still be called
        # Based on current implementation:
        # if not df.empty: df.to_parquet(...)
        # If valid_patents is empty, df will be empty, so to_parquet should not be called.
        mock_to_parquet.assert_not_called()


# Tests for FundingDataCollector
# Keep existing imports, add more if needed (e.g. time for patching sleep)
import time

SAMPLE_CRUNCHBASE_RESPONSE_VALID = {
    "entities": [
        {
            "uuid": "round-uuid-1",
            "properties": {
                "money_raised": {"value_usd": 5000000, "currency": "USD"},
                "announced_on": "2023-05-10",
                "funded_organization_identifier": {"uuid": "org-uuid-1", "value": "org-uuid-1", "name": "Innovatech"},
                "investment_type": "series_a"
            }
        },
        {
            "uuid": "round-uuid-2",
            "properties": {
                "money_raised": {"value_usd": 10000000, "currency": "USD"},
                "announced_on": "2023-06-15",
                "funded_organization_identifier": {"uuid": "org-uuid-2", "value": "org-uuid-2", "name": "Future Solutions"},
                "investment_type": "series_b"
            }
        }
    ],
    "count": 2
}

SAMPLE_CRUNCHBASE_RESPONSE_EMPTY = {"entities": [], "count": 0}

@pytest.fixture
def funding_collector():
    # Temporarily mock the API key for the purpose of instantiating FundingDataCollector
    # if settings.CRUNCHBASE_API_KEY is directly accessed at __init__ or module level.
    # The collector currently loads it from settings.py at init.
    # We can also patch settings directly if needed.
    with patch('innovation_system.data_collection.collectors.CRUNCHBASE_API_KEY', 'test_cb_key'):
        collector = collectors.FundingDataCollector()
    return collector

@patch('requests.post')
@patch('time.sleep', return_value=None) # Mock time.sleep
def test_collect_funding_rounds_success(mock_sleep, mock_post, funding_collector):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = SAMPLE_CRUNCHBASE_RESPONSE_VALID
    mock_post.return_value = mock_response

    with patch('os.makedirs') as mock_makedirs, \
         patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
        rounds = funding_collector.collect_funding_rounds(
            start_date_str="2023-01-01", categories=["ai_category_uuid"]
        )

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert funding_collector.crunchbase_url in args[0]
    assert kwargs['headers']['X-cb-user-key'] == 'test_cb_key' # Check if the mocked key is used
    assert kwargs['json']['query'][0]['values'][0] == "2023-01-01" # Check start date in query

    assert len(rounds) == 2
    assert rounds[0]['company_uuid'] == "org-uuid-1"
    assert rounds[1]['amount_usd'] == 10000000
    assert mock_sleep.called # Ensure rate limiting sleep was called

    mock_makedirs.assert_called_with("data/raw", exist_ok=True)
    mock_to_parquet.assert_called_once()


@patch('requests.post')
@patch('time.sleep', return_value=None)
def test_collect_funding_rounds_api_error(mock_sleep, mock_post, funding_collector, capsys):
    mock_post.side_effect = requests.RequestException("Crunchbase API unavailable")

    with patch('os.makedirs'), patch('pandas.DataFrame.to_parquet'): # Mock savers
        rounds = funding_collector.collect_funding_rounds(
            start_date_str="2023-01-01", categories=["ai_category"]
        )

    assert rounds == []
    captured = capsys.readouterr()
    assert "Error collecting 'ai_category' funding: Crunchbase API unavailable" in captured.out
    mock_sleep.assert_not_called() # Should not sleep if the first call fails before sleep

@patch('requests.post')
@patch('time.sleep', return_value=None)
def test_fetch_category_funding_success(mock_sleep, mock_post, funding_collector):
    # This tests the private method _fetch_category_funding
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = SAMPLE_CRUNCHBASE_RESPONSE_VALID
    mock_post.return_value = mock_response

    # We need to ensure CRUNCHBASE_API_KEY is correctly patched for this method if it's re-read
    # The fixture `funding_collector` already patches it for the instance.

    fetched_rounds = funding_collector._fetch_category_funding(
        category_identifier="some_category_uuid",
        announced_on_after="2023-01-01"
    )

    assert len(fetched_rounds) == 2
    assert fetched_rounds[0]['uuid'] == "round-uuid-1"
    mock_post.assert_called_once()


def test_validate_funding_data_valid_and_invalid(funding_collector):
    rounds_to_validate = [
        # Valid round
        {"uuid": "r1", "properties": {"money_raised": {"value_usd": 100}, "announced_on": "2023-01-01", "funded_organization_identifier": {"value": "c1", "name": "Comp1"}}},
        # Missing money_raised
        {"uuid": "r2", "properties": {"announced_on": "2023-01-02", "funded_organization_identifier": {"value": "c2", "name": "Comp2"}}},
        # Missing announced_on
        {"uuid": "r3", "properties": {"money_raised": {"value_usd": 200}, "funded_organization_identifier": {"value": "c3", "name": "Comp3"}}},
        # Missing funded_organization_identifier
        {"uuid": "r4", "properties": {"money_raised": {"value_usd": 300}, "announced_on": "2023-01-04"}},
    ]
    validated = funding_collector._validate_funding_data(rounds_to_validate)
    assert len(validated) == 1
    assert validated[0]['company_uuid'] == 'c1'
    assert validated[0]['amount_usd'] == 100

def test_validate_funding_data_empty(funding_collector):
    validated = funding_collector._validate_funding_data([])
    assert validated == []

def test_collect_funding_rounds_saves_to_parquet(funding_collector):
    # Similar to patent test, ensure parquet saving is called
    with patch('requests.post') as mock_post, \
         patch('time.sleep', return_value=None), \
         patch('os.makedirs') as mock_makedirs, \
         patch('pandas.DataFrame') as mock_pd_dataframe_constructor, \
         patch('pandas.DataFrame.to_parquet') as mock_to_parquet_method:

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_CRUNCHBASE_RESPONSE_VALID
        mock_post.return_value = mock_response

        mock_df_instance = MagicMock()
        mock_pd_dataframe_constructor.return_value = mock_df_instance
        mock_df_instance.empty = False

        funding_collector.collect_funding_rounds("2023-01-01", ["cat1"])

        mock_makedirs.assert_called_once_with("data/raw", exist_ok=True)
        mock_pd_dataframe_constructor.assert_called_once() # With the validated data
        mock_to_parquet_method.assert_called_once_with(os.path.join("data/raw", "funding.parquet"), index=False)

def test_collect_funding_rounds_no_save_for_empty_df(funding_collector):
    with patch('requests.post') as mock_post, \
         patch('time.sleep', return_value=None), \
         patch('os.makedirs') as mock_makedirs, \
         patch('pandas.DataFrame.to_parquet') as mock_to_parquet:

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_CRUNCHBASE_RESPONSE_EMPTY # No entities
        mock_post.return_value = mock_response

        funding_collector.collect_funding_rounds("2023-01-01", ["cat1"])

        mock_makedirs.assert_called_once_with("data/raw", exist_ok=True)
        # If SAMPLE_CRUNCHBASE_RESPONSE_EMPTY leads to empty validated_rounds,
        # then DataFrame constructor might be called with empty list,
        # and df.empty would be true, so to_parquet should not be called.
        mock_to_parquet.assert_not_called()


# Tests for ResearchDataCollector
# Add/confirm necessary imports: arxiv, ElementTree for PubMed XML
import arxiv # Already imported via collectors, but good practice for clarity
import xml.etree.ElementTree as ET

# Mock arxiv.Result structure
class MockArxivResult:
    def __init__(self, entry_id, title, authors, summary, categories, published, pdf_url):
        self.entry_id = entry_id
        self.title = title
        # In actual arxiv.Result, authors is a list of arxiv.Author objects.
        # For simplicity in mock, we can use strings or mock Author objects.
        self.authors = [arxiv.Author(name) for name in authors] if isinstance(authors, list) else [arxiv.Author(authors)]
        self.summary = summary
        self.categories = categories
        self.published = published # datetime object
        self.pdf_url = pdf_url

SAMPLE_ARXIV_RESULTS = [
    MockArxivResult(
        entry_id="http://arxiv.org/abs/2301.00001v1",
        title="Paper Title 1",
        authors=["Author One", "Author Two"],
        summary="Abstract for paper 1.",
        categories=["cs.AI", "cs.LG"],
        published=datetime.now(datetime.timezone.utc) - timedelta(days=5),
        pdf_url="http://arxiv.org/pdf/2301.00001v1.pdf"
    ),
    MockArxivResult(
        entry_id="http://arxiv.org/abs/2301.00002v1",
        title="Paper Title 2",
        authors=["Author Three"],
        summary="Abstract for paper 2, very long and detailed.",
        categories=["stat.ML"],
        published=datetime.now(datetime.timezone.utc) - timedelta(days=10),
        pdf_url="http://arxiv.org/pdf/2301.00002v1.pdf"
    )
]

# Sample PubMed XML for efetch
SAMPLE_PUBMED_EFETCH_XML_VALID = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">12345678</PMID>
      <Article PubModel="Print">
        <Journal>
          <Title>Journal of Medical Research</Title>
        </Journal>
        <ArticleTitle>A Study on Test Data</ArticleTitle>
        <Abstract>
          <AbstractText>This is the abstract of the test study.</AbstractText>
        </Abstract>
        <AuthorList CompleteYN="Y">
          <Author ValidYN="Y">
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
            <Initials>J</Initials>
          </Author>
        </AuthorList>
        <PubDate>
          <Year>2023</Year>
          <Month>03</Month>
          <Day>15</Day>
        </PubDate>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">98765432</PMID>
      <Article PubModel="Print">
        <ArticleTitle>Another Test Paper</ArticleTitle>
        <Abstract>
          <AbstractText>Abstract for another paper.</AbstractText>
        </Abstract>
        <AuthorList CompleteYN="Y">
          <Author ValidYN="Y">
            <LastName>Doe</LastName>
            <ForeName>Jane</ForeName>
            <Initials>J</Initials>
          </Author>
           <Author ValidYN="Y">
            <LastName>Roe</LastName>
            <ForeName>Richard</ForeName>
            <Initials>R</Initials>
          </Author>
        </AuthorList>
        <PubDate>
          <Year>2022</Year>
          <Month>Feb</Month> <!-- Test month name -->
          <Day>10</Day>
        </PubDate>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

SAMPLE_PUBMED_ESEARCH_RESPONSE_VALID = {
    "esearchresult": {
        "count": "2", "retmax": "2", "retstart": "0",
        "idlist": ["12345678", "98765432"],
        "querytranslation": "test term AND (\"2023/01/01\"[Date - Publication] : \"2023/01/31\"[Date - Publication])"
    }
}


@pytest.fixture
def research_collector():
    # Patch API key for PubMed if it's accessed during __init__ or at module level
    with patch('innovation_system.data_collection.collectors.PUBMED_API_KEY', 'test_pubmed_key'):
        collector = collectors.ResearchDataCollector()
    return collector

@patch('arxiv.Client')
def test_collect_arxiv_papers_success(mock_arxiv_client_constructor, research_collector):
    mock_arxiv_instance = MagicMock()
    mock_arxiv_client_constructor.return_value = mock_arxiv_instance

    # Make results method a generator or list
    mock_arxiv_instance.results.return_value = iter(SAMPLE_ARXIV_RESULTS)

    papers = research_collector.collect_arxiv_papers(categories=["cs.AI"], days_back=30)

    mock_arxiv_client_constructor.assert_called_once()
    # Check if arxiv.Search was called with correct parameters
    # The actual Search object is created inside, so we check results call
    mock_arxiv_instance.results.assert_called_once()
    search_args = mock_arxiv_instance.results.call_args[0][0] # Get the Search object
    assert search_args.query == "cat:cs.AI"
    # assert search_args.max_results == 1000 # Default in code, consider making this configurable or smaller for tests
    assert search_args.sort_by == arxiv.SortCriterion.SubmittedDate

    assert len(papers) == 2
    assert papers[0]['title'] == "Paper Title 1"
    assert len(papers[0]['authors']) == 2
    assert papers[1]['source'] == 'arXiv'

@patch('arxiv.Client')
def test_collect_arxiv_papers_api_error(mock_arxiv_client_constructor, research_collector, capsys):
    mock_arxiv_instance = MagicMock()
    mock_arxiv_client_constructor.return_value = mock_arxiv_instance
    mock_arxiv_instance.results.side_effect = Exception("arXiv API error")

    papers = research_collector.collect_arxiv_papers(categories=["cs.LG"], days_back=30)

    assert papers == []
    captured = capsys.readouterr()
    assert "Error collecting arXiv 'cs.LG' papers: arXiv API error" in captured.out

@patch('requests.get')
def test_collect_pubmed_papers_success(mock_requests_get, research_collector):
    # Mock esearch response
    mock_esearch_response = MagicMock()
    mock_esearch_response.status_code = 200
    mock_esearch_response.json.return_value = SAMPLE_PUBMED_ESEARCH_RESPONSE_VALID

    # Mock efetch response
    mock_efetch_response = MagicMock()
    mock_efetch_response.status_code = 200
    mock_efetch_response.content = SAMPLE_PUBMED_EFETCH_XML_VALID.encode('utf-8')

    # requests.get should return esearch then efetch responses
    mock_requests_get.side_effect = [mock_esearch_response, mock_efetch_response]

    with patch('time.sleep', return_value=None) as mock_sleep: # Mock sleep if any
        papers = research_collector.collect_pubmed_papers(search_terms=["test term"], days_back=30)

    assert len(papers) == 2
    assert papers[0]['paper_id'] == "12345678"
    assert papers[0]['title'] == "A Study on Test Data"
    assert len(papers[0]['authors']) == 1
    assert papers[1]['source'] == 'PubMed'
    assert papers[1]['published_date'] is not None # Check that date parsing worked

    assert mock_requests_get.call_count == 2 # One for esearch, one for efetch
    esearch_call_args = mock_requests_get.call_args_list[0]
    efetch_call_args = mock_requests_get.call_args_list[1]

    assert research_collector.pubmed_base_url + "esearch.fcgi" == esearch_call_args[0][0] # Corrected base_url
    assert "test term" in esearch_call_args[1]['params']['term']
    assert esearch_call_args[1]['params']['api_key'] == 'test_pubmed_key'

    assert research_collector.pubmed_base_url + "efetch.fcgi" == efetch_call_args[0][0] # Corrected base_url
    assert "12345678,98765432" in efetch_call_args[1]['params']['id']
    assert efetch_call_args[1]['params']['api_key'] == 'test_pubmed_key'

    assert mock_sleep.called


@patch('requests.get')
def test_collect_pubmed_papers_esearch_error(mock_requests_get, research_collector, capsys):
    mock_requests_get.side_effect = requests.RequestException("PubMed esearch failed")

    with patch('time.sleep', return_value=None):
        papers = research_collector.collect_pubmed_papers(search_terms=["cancer"], days_back=30)

    assert papers == []
    captured = capsys.readouterr()
    assert "Error searching PubMed for 'cancer': PubMed esearch failed" in captured.out


@patch('requests.get')
def test_fetch_pubmed_details_success(mock_requests_get, research_collector):
    mock_efetch_response = MagicMock()
    mock_efetch_response.status_code = 200
    mock_efetch_response.content = SAMPLE_PUBMED_EFETCH_XML_VALID.encode('utf-8')
    mock_requests_get.return_value = mock_efetch_response

    with patch('time.sleep', return_value=None) as mock_sleep:
        details = research_collector._fetch_pubmed_details(pmids=["12345678", "98765432"])

    assert len(details) == 2
    assert details[0]['paper_id'] == "12345678"
    assert details[1]['authors'] == ["Doe Jane", "Roe Richard"] # Check combined name
    assert mock_sleep.called


def test_validate_research_data_valid_and_invalid(research_collector, capsys):
    papers_to_validate = [
        {'paper_id': 'p1', 'title': 'Valid Paper', 'abstract': 'Valid abstract.', 'published_date': '2023-01-01', 'authors': ['Auth A'], 'source': 'arXiv', 'url': 'url1'},
        {'paper_id': 'p2', 'title': 'N/A', 'abstract': 'Abstract here.', 'published_date': '2023-01-02', 'authors': ['Auth B'], 'source': 'arXiv', 'url': 'url2'}, # Invalid title
        {'paper_id': 'p3', 'title': 'Good Title', 'abstract': None, 'published_date': '2023-01-03', 'authors': ['Auth C'], 'source': 'PubMed', 'url': 'url3'}, # Missing abstract
        {'paper_id': 'p4', 'title': 'Another Title', 'abstract': 'Good abstract.', 'published_date': None, 'authors': ['Auth D'], 'source': 'arXiv', 'url': 'url4'}, # Missing date
        {'paper_id': 'p5', 'title': 'Yet Another', 'abstract': 'Fine abstract.', 'published_date': '2023-01-05', 'authors': [], 'source': 'PubMed', 'url': 'url5'}, # Missing authors
    ]
    validated = research_collector._validate_research_data(papers_to_validate)
    assert len(validated) == 1
    assert validated[0]['paper_id'] == 'p1'

    captured = capsys.readouterr()
    assert "Invalid or incomplete paper data for ID p2 (N/A): Missing or invalid title." in captured.out
    assert "Invalid or incomplete paper data for ID p3 (Good Title): Missing abstract." in captured.out
    assert "Invalid or incomplete paper data for ID p4 (Another Title): Missing published date." in captured.out
    assert "Invalid or incomplete paper data for ID p5 (Yet Another): Missing authors." in captured.out
    assert "Validated 1/5 research papers." in captured.out # Corrected print statement

def test_validate_research_data_empty(research_collector, capsys):
    validated = research_collector._validate_research_data([])
    assert validated == []
    captured = capsys.readouterr()
    assert "Validated 0/0 research papers." in captured.out # Corrected print statement

# Example of date parsing test in PubMed (if more complex scenarios arise)
def test_pubmed_date_parsing_in_fetch_details(research_collector, capsys):
    # Simplified XML focusing on PubDate variations
    xml_content_various_dates = """
    <PubmedArticleSet>
      <PubmedArticle><MedlineCitation><PMID>1</PMID><Article><ArticleTitle>T1</ArticleTitle><Abstract><AbstractText>A1</AbstractText></Abstract><AuthorList CompleteYN="Y"><Author><LastName>LN</LastName><ForeName>FN</ForeName></Author></AuthorList>
        <PubDate><Year>2023</Year><Month>01</Month><Day>15</Day></PubDate>
      </Article></MedlineCitation></PubmedArticle>
      <PubmedArticle><MedlineCitation><PMID>2</PMID><Article><ArticleTitle>T2</ArticleTitle><Abstract><AbstractText>A2</AbstractText></Abstract><AuthorList CompleteYN="Y"><Author><LastName>LN</LastName><ForeName>FN</ForeName></Author></AuthorList>
        <PubDate><Year>2022</Year><Month>Dec</Month></PubDate> <!-- Missing Day -->
      </Article></MedlineCitation></PubmedArticle>
      <PubmedArticle><MedlineCitation><PMID>3</PMID><Article><ArticleTitle>T3</ArticleTitle><Abstract><AbstractText>A3</AbstractText></Abstract><AuthorList CompleteYN="Y"><Author><LastName>LN</LastName><ForeName>FN</ForeName></Author></AuthorList>
        <PubDate><MedlineDate>2021 Winter</MedlineDate></PubDate> <!-- MedlineDate, not parsed by current code -->
      </Article></MedlineCitation></PubmedArticle>
    </PubmedArticleSet>
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = xml_content_various_dates.encode('utf-8')

    # Mock the collectors.PUBMED_API_KEY directly as it's used in _fetch_pubmed_details
    with patch('innovation_system.data_collection.collectors.PUBMED_API_KEY', 'test_pubmed_key_direct'), \
         patch('requests.get', return_value=mock_response), \
         patch('time.sleep', return_value=None):
        details = research_collector._fetch_pubmed_details(pmids=["1", "2", "3"])

    assert details[0]['published_date'] == datetime(2023, 1, 15).isoformat()
    assert details[1]['published_date'] == datetime(2022, 12, 1).isoformat() # Defaults to day 1
    assert details[2]['published_date'] is None

    # Add dummy 'source' and 'url' for validation to pass other checks
    for item in details:
        item['source'] = 'PubMed' # Or derive as needed
        item['url'] = f"http://example.com/{item['paper_id']}"

    validated = research_collector._validate_research_data(details)
    assert len(validated) == 2
    assert validated[0]['paper_id'] == '1'
    assert validated[1]['paper_id'] == '2'
