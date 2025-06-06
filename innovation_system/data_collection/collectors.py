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
import os
from innovation_system.config.settings import USPTO_API_KEY, CRUNCHBASE_API_KEY, PUBMED_API_KEY, \
    EPO_OPS_CONSUMER_KEY, EPO_OPS_CONSUMER_SECRET, EPO_OPS_BASE_URL, EPO_OPS_ACCESS_TOKEN_URL

class PatentDataCollector:
    def __init__(self):
        self.uspto_base_url = "https://developer.uspto.gov/ibd-api/v1"
        self.wipo_rss_feed = "https://patentscope.wipo.int/search/rss"
        self.epo_ops_url = "https://ops.epo.org/3.2"
        self.api_key = USPTO_API_KEY
        # Note: The actual collect_uspto_patents method would need to be updated
        # to use self.api_key if the USPTO API requires it in headers/params.
        # Currently, it does not seem to use an API key in its request params.

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

        # Save to Parquet
        if valid_patents:
            df = pd.DataFrame(valid_patents)
            output_dir = "data/raw"
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, "patents.parquet")
            if not df.empty:
                try:
                    df.to_parquet(filepath, index=False)
                    print(f"Saved patent data to {filepath}")
                except Exception as e:
                    print(f"Error saving patent data to Parquet: {e}")
            elif os.path.exists(filepath): # If df is empty but file exists, implies current collection is empty
                print(f"No new patent data to save to {filepath}. Existing file unchanged or will be overwritten if it was from a different empty run.")


        return valid_patents

class FundingDataCollector:
    def __init__(self): # Removed crunchbase_key parameter
        self.crunchbase_key = CRUNCHBASE_API_KEY # Use imported key
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

        # Save to Parquet
        if all_rounds:
            df = pd.DataFrame(all_rounds)
            output_dir = "data/raw"
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, "funding.parquet")
            if not df.empty:
                try:
                    df.to_parquet(filepath, index=False)
                    print(f"Saved funding data to {filepath}")
                except Exception as e:
                    print(f"Error saving funding data to Parquet: {e}")
            elif os.path.exists(filepath):
                 print(f"No new funding data to save to {filepath}. Existing file unchanged or will be overwritten if it was from a different empty run.")

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
        self.arxiv_client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3) # arxiv import is at the top
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.pubmed_api_key = PUBMED_API_KEY # Use imported key

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
                            'abstract': paper.summary.replace('\n', ' ') if paper.summary else "",
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


class LivePatentDataCollector:
    def __init__(self):
        self.consumer_key = EPO_OPS_CONSUMER_KEY
        self.consumer_secret = EPO_OPS_CONSUMER_SECRET
        self.base_url = EPO_OPS_BASE_URL
        self.access_token_url = EPO_OPS_ACCESS_TOKEN_URL
        self.access_token = None
        self.token_expiry_time = None

    def _get_access_token(self):
        """
        Retrieves a new EPO OPS access token if the current one is invalid or expired.
        """
        if self.access_token and self.token_expiry_time and datetime.now() < self.token_expiry_time:
            return self.access_token

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {'grant_type': 'client_credentials'}
        auth = (self.consumer_key, self.consumer_secret)

        try:
            response = requests.post(self.access_token_url, headers=headers, data=data, auth=auth, timeout=30)
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data['access_token']
            expires_in = int(token_data['expires_in'])
            self.token_expiry_time = datetime.now() + timedelta(seconds=expires_in - 60) # 60s buffer
            print("Successfully obtained new EPO OPS access token.")
            return self.access_token
        except requests.RequestException as e:
            print(f"Error obtaining EPO OPS access token: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            self.access_token = None
            self.token_expiry_time = None
            return None
        except KeyError as e:
            print(f"Error parsing token response: Missing key {e}. Response: {response.text}")
            self.access_token = None
            self.token_expiry_time = None
            return None

    def _construct_cql_query(self, search_query, start_date_str=None, end_date_str=None):
        """
        Constructs the CQL query string for EPO OPS.
        Dates should be in 'YYYYMMDD' or 'YYYY-MM-DD' format.
        """
        cql_parts = [f'txt="{search_query}"']
        if start_date_str:
            start_date_formatted = start_date_str.replace('-', '')
            cql_parts.append(f'pd>="{start_date_formatted}"')
        if end_date_str:
            end_date_formatted = end_date_str.replace('-', '')
            cql_parts.append(f'pd<="{end_date_formatted}"')
        return " AND ".join(cql_parts)

    def collect_epo_patents(self, search_query, start_date_str=None, end_date_str=None):
        """
        Collects patent data from EPO OPS using a search query and optional date range.
        """
        token = self._get_access_token()
        if not token:
            print("Cannot collect EPO patents without an access token.")
            return []

        cql_query = self._construct_cql_query(search_query, start_date_str, end_date_str)
        headers = {'Authorization': f'Bearer {token}'}
        # Using /search to get biblio data by default. Range header for pagination if needed later.
        # For now, default range (first 25 results) is fine for initial testing.
        search_url = f"{self.base_url}/published-data/search/biblio"
        params = {'q': cql_query}

        print(f"Collecting EPO patents with query: {cql_query}")
        try:
            response = requests.get(search_url, headers=headers, params=params, timeout=60)
            response.raise_for_status()

            # Check content type, OPS usually returns XML
            content_type = response.headers.get('Content-Type', '')
            if 'application/xml' not in content_type and 'text/xml' not in content_type:
                print(f"Unexpected content type: {content_type}. Expected XML.")
                print(f"Response text: {response.text[:500]}") # Print first 500 chars
                return []

            parsed_patents = self._parse_epo_response(response.text)
            validated_patents = self._validate_patent_data(parsed_patents)
            if validated_patents:
                self._save_patent_data_to_parquet(validated_patents)
            return validated_patents
        except requests.RequestException as e:
            print(f"EPO OPS API error during patent collection: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text[:500]}") # Print first 500 chars of error response
            return []
        except ET.ParseError as e:
            print(f"Error parsing EPO OPS XML response: {e}")
            print(f"Response text: {response.text[:1000]}") # Print first 1000 chars of problematic XML
            return []


    def _parse_epo_response(self, response_xml_text):
        """
        Parses the XML response from EPO OPS.
        Initial focus: Extract total number of results and basic fields.
        """
        patents = []
        try:
            root = ET.fromstring(response_xml_text)

            # OPS XML namespace dictionary
            # Common namespaces, may need adjustment based on actual response
            ns = {
                'ops': 'http://ops.epo.org',
                'epo': 'http://www.epo.org/exchange',
                'xlink': 'http://www.w3.org/1999/xlink'
            }

            # Find total results
            # Path might be ops:biblio-search/ops:search-result[@total-result-count]
            # or ops:world-patent-data/ops:biblio-search/ops:search-result[@total-result-count]
            search_result_element = root.find('.//ops:biblio-search/ops:search-result', ns)
            if search_result_element is not None:
                total_results = search_result_element.get('total-result-count')
                print(f"EPO OPS: Found {total_results} total results for the query.")
            else:
                print("EPO OPS: Could not find total-result-count in response.")
                # Check for common error messages or unexpected structure
                if root.tag == "{http://ops.epo.org}fault": # Check if root is a fault message
                    fault_code = root.findtext('.//ops:code', namespaces=ns)
                    fault_message = root.findtext('.//ops:message', namespaces=ns)
                    print(f"EPO OPS API Error Response: Code {fault_code}, Message: {fault_message}")
                    return []


            # Iterate through exchange-document elements
            # Path: ops:world-patent-data/ops:biblio-search/ops:search-result/exchange-documents/exchange-document
            for doc in root.findall('.//ops:exchange-document', ns): # Adjusted path based on typical OPS structure
                try:
                    doc_number_element = doc.find('.//epo:publication-reference/epo:document-id[@document-id-type="epodoc"]/epo:doc-number', ns)
                    publication_number = doc_number_element.text if doc_number_element is not None else None

                    invention_title_element = doc.find('.//epo:invention-title[@lang="en"]', ns)
                    if invention_title_element is None: # Fallback to any language if English not found
                        invention_title_element = doc.find('.//epo:invention-title', ns)
                    title = invention_title_element.text if invention_title_element is not None else "N/A"

                    publication_date_element = doc.find('.//epo:publication-reference/epo:document-id[@document-id-type="epodoc"]/epo:date', ns)
                    publication_date = publication_date_element.text if publication_date_element is not None else None # YYYYMMDD format

                    patents.append({
                        'patent_id': publication_number, # Using publication number as patent_id
                        'title': title,
                        'publication_date': publication_date, # Keep as string, validation will handle format
                        'source': 'EPO'
                        # Add more fields as parsing logic develops
                    })
                except Exception as e_doc:
                    print(f"Error parsing individual EPO patent document: {e_doc}")
                    # Continue parsing other documents
                    continue

            if not patents and search_result_element is not None and total_results == "0":
                print("EPO OPS: Query successful, but no patents found matching the criteria.")
            elif not patents and search_result_element is None:
                 print("EPO OPS: No patent documents found in response, and search-result element was missing.")


        except ET.ParseError as e:
            print(f"XML parsing error in _parse_epo_response: {e}")
            print(f"Response snippet (first 1000 chars): {response_xml_text[:1000]}")
            return [] # Return empty list if major parsing error

        print(f"EPO OPS: Parsed {len(patents)} patent documents from response.")
        return patents

    def _validate_patent_data(self, patents):
        """
        Validates patent data, adapted for EPO fields.
        """
        valid_patents = []
        required_fields = ['patent_id', 'title', 'publication_date']
        for patent in patents:
            missing_fields = [field for field in required_fields if not patent.get(field)]
            if not missing_fields:
                # Validate publication_date format (YYYYMMDD from EPO)
                try:
                    if patent['publication_date']: # Ensure it's not None
                        datetime.strptime(patent['publication_date'], '%Y%m%d')
                    valid_patents.append(patent)
                except ValueError:
                    print(f"Invalid publication date format for EPO patent ID {patent.get('patent_id')}: {patent.get('publication_date')}. Expected YYYYMMDD.")
            else:
                print(f"Missing critical data for EPO patent ID {patent.get('patent_id', 'Unknown')}: {', '.join(missing_fields)}")

        print(f"Validated {len(valid_patents)}/{len(patents)} EPO patents.")
        return valid_patents

    def _save_patent_data_to_parquet(self, patents, filepath="data/raw/patents_epo.parquet"):
        """
        Saves the list of patent dictionaries to a Parquet file.
        """
        if not patents:
            print("No EPO patent data to save.")
            return

        df = pd.DataFrame(patents)
        if df.empty:
            print("DataFrame is empty, no EPO patent data to save.")
            return

        try:
            output_dir = os.path.dirname(filepath)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            df.to_parquet(filepath, index=False)
            print(f"Successfully saved {len(patents)} EPO patents to {filepath}")
        except Exception as e:
            print(f"Error saving EPO patent data to Parquet file {filepath}: {e}")
