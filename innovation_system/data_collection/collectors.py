# Cleared for new implementation
# This file will house data collection classes: PatentDataCollector, FundingDataCollector, ResearchDataCollector.

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import pandas as pd
import json
import time
import arxiv
import feedparser # Note: feedparser was imported in academic research but not directly used in the snippet; keeping it for completeness if intended for parsing RSS like WIPO.

# --- Data Collection Implementation ---

class PatentDataCollector:
    def __init__(self):
        self.uspto_base_url = "https://developer.uspto.gov/ibd-api/v1" # Example, actual API might differ
        self.wipo_rss_feed = "https://patentscope.wipo.int/search/rss" # Example, actual usage would need parsing logic
        self.epo_ops_url = "https://ops.epo.org/3.2" # Example, actual usage would need OAuth and specific client

    def collect_uspto_patents(self, start_date, end_date, tech_category_query):
        """
        Collect USPTO patent data via an example API.
        Note: The actual USPTO API for bulk search might be different (e.g., Patent Examination Data System - PEDS, or bulk downloads).
        This is a conceptual representation based on a generic API structure.
        """
        # Using a query parameter that might be relevant, e.g., 'qf' for query fields, 'query' for query text.
        # The query format 'ccl=...' is specific to some older systems or specific fields.
        # A more general approach might use keyword search:
        params = {
            'query': f'(inventionTitle:({tech_category_query}) OR abstractText:({tech_category_query})) AND filingDate:[{start_date.strftime("%Y%m%d")} TO {end_date.strftime("%Y%m%d")}]',
            'start': 0, # for pagination
            'rows': 100 # max rows per page, implement pagination if needed
        }

        # A more realistic endpoint for searching patents might be something like '/patents/query'
        # For this example, we'll stick to the provided '/applications' but acknowledge it might be for status.
        # Using a generic search endpoint if available from USPTO for published grants/applications.
        # The USPTO Bulk Data Access (https://developer.uspto.gov/product/patent-grant-full-text-dataxml)
        # is for downloading bulk files, not typically for live API queries of this nature.
        # Let's assume a hypothetical search endpoint for this example.
        search_endpoint = f"{self.uspto_base_url}/patents/search" # Hypothetical

        all_patents = []
        current_page = 0
        max_pages = 10 # Safety break for pagination

        while current_page < max_pages:
            params['start'] = current_page * params['rows']
            try:
                # This is a conceptual API call.
                # response = requests.get(search_endpoint, params=params, timeout=30)
                # For the sake of the example, we'll use the original endpoint and adapt parsing.
                response = requests.get(f"{self.uspto_base_url}/applications", params={'searchText': tech_category_query, 'start':start_date.strftime('%Y%m%d'), 'end':end_date.strftime('%Y%m%d'), 'rows':params['rows'] }, timeout=30)
                response.raise_for_status()
                data = response.json()

                patents_page = self._parse_uspto_response(data)
                if not patents_page:
                    break # No more results
                all_patents.extend(patents_page)

                # Simplified pagination check - real APIs provide totalResults or similar
                if len(patents_page) < params['rows'] or 'results' not in data or len(data['results']) < params['rows']:
                    break
                current_page += 1
                time.sleep(0.5) # Rate limiting

            except requests.RequestException as e:
                print(f"USPTO API error on page {current_page}: {e}")
                break # Stop on error
            except json.JSONDecodeError as e:
                print(f"USPTO API JSON parsing error on page {current_page}: {e} - Response: {response.text}")
                break

        return self._validate_patent_data(all_patents)

    def _parse_uspto_response(self, data):
        """Parse USPTO API response into standardized format"""
        patents = []
        # The structure of 'data' depends heavily on the actual API endpoint used.
        # Assuming 'data' is a dictionary and 'results' is a list of patent items.
        api_results = data.get('results', data.get('response', {}).get('docs', [])) # Common structures

        for item in api_results:
            # Try to extract relevant fields, allowing for variations in naming
            patent_id = item.get('applicationNumberText', item.get('patentNumber', item.get('publicationNumber')))
            title = item.get('inventionTitle', item.get('title'))

            filing_date_str = item.get('filingDate')
            # Try to parse date if it's a string, handle various formats if necessary
            parsed_filing_date = None
            if isinstance(filing_date_str, str):
                try:
                    # Example format, common API date format: YYYY-MM-DDTHH:MM:SSZ
                    if 'T' in filing_date_str:
                        parsed_filing_date = datetime.fromisoformat(filing_date_str.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                    else: # Simpler YYYY-MM-DD or YYYYMMDD
                        parsed_filing_date = pd.to_datetime(filing_date_str).strftime('%Y-%m-%d')
                except ValueError:
                    # print(f"Warning: Could not parse filing date '{filing_date_str}' for patent {patent_id}")
                    pass # Keep as None or original string if unparseable

            inventors_data = item.get('inventors', [])
            num_inventors = len(inventors_data) if isinstance(inventors_data, list) else 1 if inventors_data else 0

            assignee = item.get('assigneeEntityName', item.get('applicantName', item.get('assignees', [{}])[0].get('name')))
            if isinstance(assignee, list) and assignee: assignee = assignee[0] # Take first if list

            # Tech class can be complex (CPC, USPC, IPC). Prioritize CPC if available.
            main_class = None
            classifications = item.get('classificationsIpc', item.get('cpcClassification', item.get('classificationNationalCurrent', [])))
            if isinstance(classifications, list) and classifications:
                # Look for main classification or first one
                main_class_obj = next((c for c in classifications if c.get('isMain', False) or c.get('sequence', 0) == 0), classifications[0])
                main_class = main_class_obj.get('symbol', main_class_obj.get('classificationSymbolText', {}).get('text'))
            elif isinstance(classifications, dict): # If it's a single classification object
                 main_class = classifications.get('symbol', classifications.get('classificationSymbolText', {}).get('text'))


            # Citations data is often not in basic search results and requires specific citation APIs or bulk data.
            # For 'referencedBy', it suggests backward citations (patents cited by this one).
            # Forward citations (patents citing this one) are usually harder to get.
            num_citations = len(item.get('referencedBy', item.get('citation', []))) # Placeholder

            patents.append({
                'patent_id': str(patent_id) if patent_id else None,
                'title': str(title) if title else None,
                'filing_date': parsed_filing_date, # Store parsed date string
                'inventors_count': num_inventors,
                'assignee': str(assignee) if assignee else None,
                'tech_class': str(main_class) if main_class else None,
                'citations_count': num_citations, # Note: This is likely backward citations or placeholder
                'source': 'USPTO'
            })
        return patents

    def _validate_patent_data(self, patents_list):
        """Validate patent data quality and completeness"""
        valid_patents = []
        if not isinstance(patents_list, list):
            print("Validation error: input patents_list is not a list.")
            return []

        for patent in patents_list:
            if not isinstance(patent, dict):
                # print(f"Skipping invalid patent entry (not a dict): {patent}")
                continue

            if (patent.get('patent_id') and
                patent.get('filing_date') and # Ensure date is usable
                patent.get('title')):
                try:
                    # Further check filing_date format if it's stored as string
                    if isinstance(patent['filing_date'], str):
                        datetime.strptime(patent['filing_date'], '%Y-%m-%d')
                    valid_patents.append(patent)
                except (ValueError, TypeError) as e:
                    # print(f"Patent {patent.get('patent_id')} skipped due to invalid filing_date '{patent.get('filing_date')}': {e}")
                    pass
            # else:
                # print(f"Patent {patent.get('patent_id')} skipped due to missing critical fields.")

        # print(f"Validated {len(valid_patents)}/{len(patents_list)} patents.")
        return valid_patents

class FundingDataCollector:
    def __init__(self, crunchbase_key):
        self.crunchbase_key = crunchbase_key
        self.crunchbase_api_base = "https://api.crunchbase.com/api/v4"
        self.rate_limit_delay = 1.1  # Adhere to Crunchbase rate limits (e.g. ~1 req/sec)

    def collect_funding_rounds(self, announced_after_date_str, category_uuids_or_keywords):
        """
        Collect funding data from Crunchbase API using POST for search.
        announced_after_date_str: 'YYYY-MM-DD'
        category_uuids_or_keywords: List of Crunchbase category UUIDs or keywords.
        """
        all_rounds = []
        search_endpoint = f"{self.crunchbase_api_base}/searches/funding_rounds"
        headers = {'X-cb-user-key': self.crunchbase_key, 'Content-Type': 'application/json', 'Accept': 'application/json'}

        for category_identifier in category_uuids_or_keywords:
            after_id = None # For pagination using 'after_id'
            page_count = 0
            max_pages = 20 # Safety limit for pages per category

            while page_count < max_pages:
                query_body = {
                    "field_ids": [
                        "money_raised", "funded_organization", "announced_on", "investment_type",
                        "investor_identifiers", "funded_organization_categories", "funded_organization_location_identifiers"
                    ],
                    "order": [{"field_id": "announced_on", "sort": "desc"}],
                    "query": [
                        {"type": "predicate", "field_id": "announced_on", "operator_id": "gte", "values": [announced_after_date_str]},
                        # Example predicate for category (if 'category_identifier' is a UUID)
                        # This assumes funded_organization_categories contains UUIDs.
                        # If it's a keyword, you might search funded_organization.short_description or similar.
                        {"type": "predicate", "field_id": "funded_organization_categories", "operator_id": "includes", "values": [category_identifier]}
                        # Or for keywords in organization name/description:
                        # {"type": "predicate", "field_id": "funded_organization", "operator_id": "contains", "values": [category_identifier]}
                    ],
                    "limit": 100 # Crunchbase max limit is often 100 or 1000, check docs
                }
                if after_id:
                    query_body["after_id"] = after_id

                try:
                    response = requests.post(search_endpoint, headers=headers, json=query_body, timeout=45)
                    response.raise_for_status()
                    response_data = response.json()

                    entities = response_data.get('entities', [])
                    if not entities:
                        break # No more results for this category

                    validated_page_rounds = self._validate_funding_data(entities)
                    all_rounds.extend(validated_page_rounds)

                    # Pagination: Check for 'after_id' for next page
                    # The actual pagination mechanism might differ (e.g. 'next_page_url', 'paging.next_page_token')
                    # Crunchbase v4 uses 'after_id' from the last entity of the current page for cursor-based pagination.
                    if len(entities) < query_body["limit"] or not entities[-1].get('uuid'): # Check if less than limit or no UUID for after_id
                        break
                    after_id = entities[-1]['uuid'] # UUID of the last entity

                    page_count += 1
                    time.sleep(self.rate_limit_delay)

                except requests.RequestException as e:
                    print(f"Crunchbase API error for category '{category_identifier}', page {page_count}: {e} - Response: {response.text if 'response' in locals() else 'N/A'}")
                    break # Stop for this category on error
                except json.JSONDecodeError as e:
                    print(f"Crunchbase JSON parsing error for '{category_identifier}', page {page_count}: {e} - Response: {response.text}")
                    break

        return all_rounds

    def _validate_funding_data(self, rounds_entities):
        """Validate funding round data quality from Crunchbase entities."""
        valid_rounds = []
        for entity in rounds_entities:
            properties = entity.get('properties', {})
            money_raised_obj = properties.get('money_raised')

            org_identifier_obj = properties.get('funded_organization', {}).get('identifier', {})
            org_uuid = org_identifier_obj.get('uuid')
            org_name = org_identifier_obj.get('value') # Usually the permalink/name

            if (money_raised_obj and money_raised_obj.get('value_usd') is not None and # Check for None explicitly
                properties.get('announced_on') and
                org_uuid and org_name):

                valid_rounds.append({
                    'funding_round_uuid': entity.get('uuid'),
                    'company_uuid': org_uuid,
                    'company_name': org_name,
                    'amount_usd': money_raised_obj['value_usd'],
                    'currency': money_raised_obj.get('currency', 'USD'), # Default to USD
                    'date': properties['announced_on'], # Store as YYYY-MM-DD string
                    'stage': properties.get('investment_type', 'unknown'),
                    'source': 'Crunchbase'
                })
            # else:
                # print(f"Skipping funding round due to missing critical data: {entity.get('uuid')}")
        return valid_rounds

class ResearchDataCollector:
    def __init__(self, pubmed_api_key=None):
        self.arxiv_client = arxiv.Client(page_size=200, delay_seconds=3.1, num_retries=3)
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.pubmed_api_key = pubmed_api_key # NCBI API key for higher rate limits

    def collect_arxiv_papers(self, arxiv_categories_or_queries, days_back=30):
        """Collect recent papers from arXiv."""
        all_papers = []
        # Ensure cutoff_date is offset-aware (UTC) for comparison with arXiv's offset-aware dates
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

        for query_term in arxiv_categories_or_queries:
            # Construct query: can be category like 'cs.AI' or general search 'ti:"quantum entanglement" AND abs:teleportation'
            search_query = query_term if ":" in query_term or "cat:" in query_term else f'all:"{query_term}"'

            try:
                search = arxiv.Search(
                    query=search_query,
                    max_results=500, # Max results per query, can increase if needed with pagination logic
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )

                results = list(self.arxiv_client.results(search)) # Exhaust the generator

                for paper in results:
                    # paper.published is already timezone-aware (UTC)
                    if paper.published >= cutoff_date:
                        all_papers.append({
                            'paper_id': paper.entry_id,
                            'title': paper.title,
                            'authors': [str(author) for author in paper.authors],
                            'abstract': paper.summary.replace('\n', ' ').strip(),
                            'categories': paper.categories, # List of categories
                            'published_date': paper.published.isoformat(), # Store as ISO string
                            'pdf_url': paper.pdf_url,
                            'doi': paper.doi,
                            'journal_ref': paper.journal_ref,
                            # Citation count not available from arXiv; use external source if needed
                            'citation_count_external': 0,
                            'source': 'arXiv'
                        })
            except Exception as e:
                print(f"Error collecting arXiv papers for query '{query_term}': {e}")
                continue # Continue to next query/category

        return self._validate_research_data(all_papers)

    def _fetch_pubmed_details_batch(self, pmids_batch):
        """Fetch full paper details for a batch of PMIDs from PubMed."""
        if not pmids_batch: return []

        details_list = []
        fetch_url = f"{self.pubmed_base_url}/efetch.fcgi"
        fetch_params = {'db': 'pubmed', 'id': ','.join(pmids_batch), 'retmode': 'xml', 'rettype': 'abstract'}
        if self.pubmed_api_key: fetch_params['api_key'] = self.pubmed_api_key

        try:
            response = requests.get(fetch_url, params=fetch_params, timeout=60)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            for article_xml in root.findall('.//PubmedArticle'):
                pmid_el = article_xml.find('.//PMID')
                title_el = article_xml.find('.//ArticleTitle')
                abstract_el = article_xml.find('.//AbstractText') # Could be multiple segments

                # Concatenate abstract segments if they exist
                abstract_texts = [seg.text for seg in article_xml.findall('.//AbstractText') if seg.text]
                full_abstract = " ".join(abstract_texts).strip() if abstract_texts else None

                # Robust date parsing for PubMed
                pub_date_iso = None
                pub_date_node = article_xml.find('.//PubmedData/History/PubMedPubDate[@PubStatus="pubmed"]') # Prioritize "pubmed" status
                if pub_date_node is None: # Fallback to general ArticleDate or PubDate
                    pub_date_node = article_xml.find('.//ArticleDate') or article_xml.find('.//PubDate')

                if pub_date_node is not None:
                    year = pub_date_node.findtext('Year')
                    month = pub_date_node.findtext('Month') # Can be number or abbreviation
                    day = pub_date_node.findtext('Day')
                    if year and month and day:
                        try:
                            month_num_str = month if month.isdigit() else str(datetime.strptime(month, "%b").month if len(month) == 3 else datetime.strptime(month, "%B").month)
                            pub_date_iso = datetime(int(year), int(month_num_str), int(day)).isoformat()
                        except ValueError: pass # Date parts invalid

                authors_list = [f"{auth.findtext('LastName', '')} {auth.findtext('Initials', '')}".strip()
                                for auth in article_xml.findall('.//AuthorList/Author') if auth.findtext('LastName')]

                doi_el = article_xml.find(".//ArticleId[@IdType='doi']")

                details_list.append({
                    'paper_id': pmid_el.text if pmid_el is not None else None,
                    'title': title_el.text if title_el is not None else "N/A",
                    'authors': authors_list,
                    'abstract': full_abstract,
                    'published_date': pub_date_iso,
                    'doi': doi_el.text if doi_el is not None else None,
                    'journal_title': article_xml.findtext('.//ISOAbbreviation') or article_xml.findtext('.//Journal/Title'),
                    'citation_count_external': 0, # Use PMC or Semantic Scholar for citations
                    'source': 'PubMed'
                })
            time.sleep(0.11 if self.pubmed_api_key else 0.35) # Adhere to NCBI rate limits
        except requests.RequestException as e:
            print(f"Error fetching PubMed details batch: {e}")
        except ET.ParseError as e:
            print(f"Error parsing PubMed XML for batch: {e} - Response: {response.text if 'response' in locals() else 'N/A'}")
        return details_list

    def collect_pubmed_papers(self, search_terms_list, days_back=30):
        """Collect biomedical papers from PubMed using esearch then efetch."""
        all_papers_details = []
        # Date range for PubMed query
        max_date_str = datetime.now().strftime('%Y/%m/%d')
        min_date_str = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')
        date_query_part = f"AND (\"{min_date_str}\"[Date - Publication] : \"{max_date_str}\"[Date - Publication])"

        for term in search_terms_list:
            pmids_for_term = []
            try:
                search_url = f"{self.pubmed_base_url}/esearch.fcgi"
                # Batch PMIDs retrieval
                retstart_idx = 0
                retmax_batch = 500 # How many PMIDs to retrieve per esearch call
                total_retrieved_for_term = 0
                expected_total_for_term = float('inf') # Initialize to a high number

                while total_retrieved_for_term < expected_total_for_term:
                    search_params = {
                        'db': 'pubmed', 'term': f"({term}) {date_query_part}",
                        'retmax': retmax_batch, 'retstart': retstart_idx,
                        'sort': 'pub_date', 'retmode': 'json'
                    }
                    if self.pubmed_api_key: search_params['api_key'] = self.pubmed_api_key

                    response = requests.get(search_url, params=search_params, timeout=45)
                    response.raise_for_status()
                    search_results = response.json()

                    esearch_result = search_results.get('esearchresult', {})
                    current_batch_pmids = esearch_result.get('idlist', [])
                    pmids_for_term.extend(current_batch_pmids)

                    if retstart_idx == 0: # On first call, get total count
                        expected_total_for_term = int(esearch_result.get('count', 0))

                    total_retrieved_for_term += len(current_batch_pmids)
                    retstart_idx += retmax_batch

                    if not current_batch_pmids or total_retrieved_for_term >= expected_total_for_term or len(current_batch_pmids) < retmax_batch:
                        break # Exit loop if no more PMIDs or all retrieved
                    time.sleep(0.11 if self.pubmed_api_key else 0.35)

                # Fetch details for collected PMIDs in batches
                if pmids_for_term:
                    # print(f"Found {len(pmids_for_term)} PMIDs for term '{term}'. Fetching details...")
                    for i in range(0, len(pmids_for_term), 100): # efetch batch size
                        batch_to_fetch = pmids_for_term[i:i+100]
                        paper_details_batch = self._fetch_pubmed_details_batch(batch_to_fetch)
                        all_papers_details.extend(paper_details_batch)
                # else:
                    # print(f"No PMIDs found for term '{term}' in the specified date range.")

            except requests.RequestException as e:
                print(f"Error searching PubMed for '{term}': {e} - Response: {response.text if 'response' in locals() else 'N/A'}")
            except json.JSONDecodeError as e:
                print(f"Error parsing PubMed search JSON for '{term}': {e} - Response: {response.text if 'response' in locals() else 'N/A'}")
            except Exception as e:
                print(f"Unexpected error collecting PubMed for '{term}': {e}")

        return self._validate_research_data(all_papers_details)

    def _validate_research_data(self, papers_list):
        """Validate research paper data quality."""
        valid_papers = []
        if not isinstance(papers_list, list): return []

        for paper in papers_list:
            if not isinstance(paper, dict): continue
            # Check for essential fields and non-empty abstract/title
            if (paper.get('paper_id') and
                paper.get('title') and paper['title'] not in ["N/A", ""] and
                paper.get('abstract') and paper['abstract'] not in ["N/A", ""] and
                paper.get('published_date') and # Ensure date is valid
                isinstance(paper.get('authors'), list) and len(paper['authors']) > 0):

                # Optional: check abstract length
                # This needs research_config to be accessible here or passed in.
                # For now, this check is disabled as research_config is not in scope.
                # if 'min_abstract_length' in research_config and                 #    len(paper['abstract']) < research_config['min_abstract_length']:
                #     # print(f"Paper {paper['paper_id']} skipped: abstract too short.")
                #     continue
                valid_papers.append(paper)
            # else:
                # print(f"Skipping invalid research paper: {paper.get('paper_id')}")

        # print(f"Validated {len(valid_papers)}/{len(papers_list)} research papers.")
        return valid_papers

```
