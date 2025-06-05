import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pandas as pd
import json
import time
import arxiv
# feedparser was imported but not used directly in PatentDataCollector or ResearchDataCollector,
# but keeping it here if it's intended for WIPO/EPO extensions.
import feedparser

class PatentDataCollector:
    def __init__(self):
        self.uspto_base_url = "https://developer.uspto.gov/ibd-api/v1"
        self.wipo_rss_feed = "https://patentscope.wipo.int/search/rss" # Example, actual usage would need parsing logic
        self.epo_ops_url = "https://ops.epo.org/3.2" # Example, actual usage would need OAuth and specific client

    def collect_uspto_patents(self, start_date, end_date, tech_category):
        """
        Collect USPTO patent data via API
        """
        params = {
            'searchText': f'ccl=({tech_category})', # Note: searchText syntax might vary based on API version/specifics
            'start': start_date.strftime('%Y%m%d'),
            'end': end_date.strftime('%Y%m%d'),
            'rows': 1000 # Consider pagination for more than 1000 results
        }

        try:
            response = requests.get(f"{self.uspto_base_url}/applications", # This endpoint might not be for bulk search
                                  params=params, timeout=30)
            response.raise_for_status()

            patents = self._parse_uspto_response(response.json())
            return self._validate_patent_data(patents)

        except requests.RequestException as e:
            print(f"USPTO API error: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"USPTO API JSON parsing error: {e} - Response: {response.text}")
            return []

    def _parse_uspto_response(self, data):
        """Parse USPTO API response into standardized format"""
        patents = []
        for item in data.get('results', []): # Structure depends heavily on actual API endpoint
            patents.append({
                'patent_id': item.get('applicationNumberText', item.get('patentNumber')), # Adjust based on endpoint
                'title': item.get('inventionTitle'),
                'filing_date': item.get('filingDate'), # Ensure correct date parsing
                'inventors': len(item.get('inventors', []) if isinstance(item.get('inventors'), list) else [item.get('inventors')]),
                'assignee': item.get('assigneeEntityName', item.get('applicantName')), # Adjust based on endpoint
                'tech_class': item.get('mainClassificationSymbol', item.get('classificationNationalCurrent', {}).get('classificationSymbolText', {}).get('text')),
                'citations': len(item.get('referencedBy', [])), # This is highly dependent on the API providing citation info
                'source': 'USPTO'
            })
        return patents

    def _validate_patent_data(self, patents):
        """Validate patent data quality and completeness"""
        valid_patents = []
        for patent in patents:
            if (patent.get('patent_id') and
                patent.get('filing_date') and
                patent.get('title')):
                try:
                    if isinstance(patent['filing_date'], str):
                        datetime.strptime(patent['filing_date'], '%Y-%m-%d') # Example format, adjust as needed
                    valid_patents.append(patent)
                except (ValueError, TypeError):
                    print(f"Invalid filing date format for patent ID {patent.get('patent_id')}: {patent.get('filing_date')}")
            else:
                 print(f"Missing critical data for patent: {patent.get('patent_id')}")

        print(f"Validated {len(valid_patents)}/{len(patents)} patents")
        return valid_patents

class FundingDataCollector:
    def __init__(self, crunchbase_key): # Add keys/auth for AngelList, SEC Edgar etc.
        self.crunchbase_key = crunchbase_key
        self.crunchbase_url = "https://api.crunchbase.com/api/v4"
        self.rate_limit_delay = 1.2  # seconds between requests

    def collect_funding_rounds(self, start_date_str, categories): # Expects start_date as YYYY-MM-DD string
        """
        Collect funding data from Crunchbase API
        """
        all_rounds = []
        start_date_dt = datetime.strptime(start_date_str, '%Y-%m-%d')

        for category_uuid_or_name in categories: # Crunchbase often uses UUIDs for categories/industries
            try:
                rounds = self._fetch_category_funding(category_uuid_or_name, start_date_dt.strftime('%Y-%m-%d'))
                validated_rounds = self._validate_funding_data(rounds)
                all_rounds.extend(validated_rounds)
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                print(f"Error collecting '{category_uuid_or_name}' funding: {e}")
                continue
        return all_rounds

    def _fetch_category_funding(self, category_identifier, announced_on_after):
        """Fetch funding rounds for specific category from Crunchbase"""
        headers = {'X-cb-user-key': self.crunchbase_key, 'accept': 'application/json'}
        query_body = {
            "field_ids": [
                "money_raised", "announced_on", "funded_organization_identifier",
                "investment_type", "investor_identifiers"
            ],
            "query": [
                {"type": "predicate", "field_id": "announced_on", "operator_id": "gte", "values": [announced_on_after]},
            ],
            "order": [{"field_id": "announced_on", "sort": "desc"}],
            "limit": 1000
        }
        response = requests.post(
            f"{self.crunchbase_url}/searches/funding_rounds",
            headers=headers, json=query_body, timeout=30
        )
        response.raise_for_status()
        return response.json().get('entities', [])

    def _validate_funding_data(self, rounds):
        """Validate funding round data quality"""
        valid_rounds = []
        for round_data in rounds:
            properties = round_data.get('properties', {})
            money_raised_obj = properties.get('money_raised')
            funded_org_identifier = properties.get('funded_organization_identifier', {}).get('value')

            if (money_raised_obj and money_raised_obj.get('value_usd') and
                properties.get('announced_on') and funded_org_identifier):
                valid_rounds.append({
                    'company_uuid': funded_org_identifier,
                    'company_name': properties.get('funded_organization_identifier', {}).get('name', 'Unknown'),
                    'amount_usd': money_raised_obj['value_usd'],
                    'currency': money_raised_obj.get('currency', 'USD'),
                    'date': properties['announced_on'],
                    'stage': properties.get('investment_type', 'Unknown'),
                    'source': 'Crunchbase'
                })
        return valid_rounds

class ResearchDataCollector:
    def __init__(self):
        self.arxiv_client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.pubmed_api_key = None

    def collect_arxiv_papers(self, categories, days_back=30):
        """
        Collect recent papers from arXiv
        """
        papers = []
        cutoff_date = datetime.now(datetime.timezone.utc) - timedelta(days=days_back)

        for category in categories:
            try:
                search = arxiv.Search(
                    query=f"cat:{category}", max_results=1000,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                results = self.arxiv_client.results(search)
                for paper in results:
                    if paper.published.replace(tzinfo=datetime.timezone.utc) > cutoff_date:
                        papers.append({
                            'paper_id': paper.entry_id, 'title': paper.title,
                            'authors': [str(author) for author in paper.authors],
                            'abstract': paper.summary.replace('\n', ' '),
                            'categories': paper.categories, 'published_date': paper.published.isoformat(),
                            'pdf_url': paper.pdf_url, 'citation_count': 0, 'source': 'arXiv'
                        })
            except Exception as e:
                print(f"Error collecting arXiv '{category}' papers: {e}")
                continue
        return self._validate_research_data(papers)

    def _fetch_pubmed_details(self, pmids):
        """Fetch full paper details for a list of PMIDs."""
        if not pmids: return []
        details_list = []
        for i in range(0, len(pmids), 200):
            batch_pmids = pmids[i:i+200]
            fetch_url = f"{self.pubmed_base}/efetch.fcgi"
            fetch_params = {'db': 'pubmed', 'id': ','.join(batch_pmids), 'retmode': 'xml', 'rettype': 'abstract'}
            if self.pubmed_api_key: fetch_params['api_key'] = self.pubmed_api_key

            try:
                response = requests.get(fetch_url, params=fetch_params, timeout=60)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                for article_xml in root.findall('.//PubmedArticle'):
                    title_e = article_xml.find('.//ArticleTitle')
                    abs_e = article_xml.find('.//AbstractText')
                    pmid_e = article_xml.find('.//PMID')
                    year_e = article_xml.find('.//PubDate/Year')
                    month_e = article_xml.find('.//PubDate/Month')
                    day_e = article_xml.find('.//PubDate/Day')
                    pub_date = None
                    try:
                        year = int(year_e.text) if year_e is not None and year_e.text else 1900
                        month_str = month_e.text if month_e is not None and month_e.text else "Jan"
                        month = datetime.strptime(month_str, "%b").month if len(month_str) == 3 and not month_str.isdigit() else (datetime.strptime(month_str, "%B").month if not month_str.isdigit() else int(month_str))
                        day = int(day_e.text) if day_e is not None and day_e.text else 1
                        pub_date = datetime(year, month, day).isoformat()
                    except Exception: pass
                    authors = [f"{auth.findtext('LastName', '')} {auth.findtext('ForeName', '')}".strip() for auth in article_xml.findall('.//AuthorList/Author')]
                    details_list.append({
                        'paper_id': pmid_e.text if pmid_e is not None else None,
                        'title': title_e.text if title_e is not None else "N/A",
                        'authors': authors, 'abstract': abs_e.text if abs_e is not None else "N/A",
                        'published_date': pub_date, 'citation_count': 0, 'source': 'PubMed'
                    })
                time.sleep(0.4)
            except requests.RequestException as e: print(f"Error fetching PubMed details: {e}")
            except ET.ParseError as e: print(f"Error parsing PubMed XML: {e}")
        return details_list

    def collect_pubmed_papers(self, search_terms, days_back=30):
        """Collect biomedical papers from PubMed using esearch then efetch."""
        all_papers_details = []
        mindate = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')
        maxdate = datetime.now().strftime('%Y/%m/%d')

        for term in search_terms:
            try:
                search_url = f"{self.pubmed_base}/esearch.fcgi"
                search_params = {
                    'db': 'pubmed', 'term': f"{term} AND (\"{mindate}\"[Date - Publication] : \"{maxdate}\"[Date - Publication])",
                    'retmax': 10000, 'sort': 'pub_date', 'retmode': 'json'
                }
                if self.pubmed_api_key: search_params['api_key'] = self.pubmed_api_key

                response = requests.get(search_url, params=search_params, timeout=30)
                response.raise_for_status()
                search_results = response.json()
                pmids = search_results.get('esearchresult', {}).get('idlist', [])

                if pmids:
                    paper_details = self._fetch_pubmed_details(pmids)
                    all_papers_details.extend(paper_details)
                time.sleep(0.4)
            except requests.RequestException as e: print(f"Error searching PubMed for '{term}': {e}")
            except json.JSONDecodeError as e: print(f"Error parsing PubMed search JSON for '{term}': {e} - Response: {response.text}")
            except Exception as e: print(f"An unexpected error occurred while collecting PubMed papers for '{term}': {e}")
        return self._validate_research_data(all_papers_details)

    def _validate_research_data(self, papers):
        """Validate research paper data quality"""
        valid_papers = []
        for paper in papers:
            if (paper.get('paper_id') and paper.get('title') and paper.get('title') != "N/A" and
                paper.get('abstract') and paper.get('abstract') != "N/A" and
                paper.get('published_date') and len(paper.get('authors', [])) > 0):
                valid_papers.append(paper)
            else:
                print(f"Invalid or incomplete paper data: {paper.get('paper_id')}")
        print(f"Validated {len(valid_papers)}/{len(papers)} research papers")
        return valid_papers
