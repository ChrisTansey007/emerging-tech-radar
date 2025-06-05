# Cleared for new implementation
# This file will house data collection classes: PatentDataCollector, FundingDataCollector, ResearchDataCollector.

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import pandas as pd
import json
import time
import arxiv

# import feedparser # Note: feedparser was imported in academic research but not directly used in the snippet; keeping it for completeness if intended for parsing RSS like WIPO.

# --- Data Collection Implementation ---

from typing import List, Dict, Any, Optional

class PatentDataCollector:
    """Collects patent data from various sources, with a primary example for USPTO."""

    def __init__(self):
        self.uspto_base_url: str = (
            "https://developer.uspto.gov/ibd-api/v1"  # Example, actual API might differ
        )
        # Placeholder for WIPO/EPO integration
        self.wipo_rss_feed: str = "https://patentscope.wipo.int/search/rss"
        self.epo_ops_url: str = "https://ops.epo.org/3.2"

    def collect_uspto_patents(
        self, start_date: datetime, end_date: datetime, tech_category_query: str
    ) -> List[Dict[str, Any]]:
        """
        Collects USPTO patent data for a given technology query within a date range.

        Note: The specific USPTO API endpoint and query parameters used here are
        conceptual and may need adjustment for actual USPTO bulk search APIs
        (e.g., PEDS or bulk data downloads). This implementation simulates paginated
        API calls.

        Args:
            start_date: The start date for the patent search.
            end_date: The end date for the patent search.
            tech_category_query: The technology category or keyword query string.

        Returns:
            A list of dictionaries, where each dictionary represents a patent
            and contains standardized fields like 'patent_id', 'title', 'filing_date'.
            Returns an empty list if an API error occurs or no patents are found.
        """
        # Example: using a keyword search approach for titles or abstracts.
        # USPTO API query syntax can be complex; this is a simplified version.
        # The query format 'ccl=...' is specific to some older systems or specific fields.
        # A more general approach might use keyword search:
        params = {
            "query": f'(inventionTitle:({tech_category_query}) OR abstractText:({tech_category_query})) AND filingDate:[{start_date.strftime("%Y%m%d")} TO {end_date.strftime("%Y%m%d")}]',
            "start": 0,  # for pagination
            "rows": 100,  # max rows per page, implement pagination if needed
        }

        # A more realistic endpoint for searching patents might be something like '/patents/query'
        # For this example, we'll stick to the provided '/applications' but acknowledge it might be for status.
        # Using a generic search endpoint if available from USPTO for published grants/applications.
        # The USPTO Bulk Data Access (https://developer.uspto.gov/product/patent-grant-full-text-dataxml)
        # is for downloading bulk files, not typically for live API queries of this nature.
        # Let's assume a hypothetical search endpoint for this example.
        # search_endpoint = f"{self.uspto_base_url}/patents/search" # Hypothetical - F841 assigned but never used

        all_patents = []
        current_page = 0
        max_pages = 10  # Safety break for pagination

        while current_page < max_pages:
            params["start"] = current_page * params["rows"]
            try:
                # This is a conceptual API call.
                # response = requests.get(search_endpoint, params=params, timeout=30)
                # For the sake of the example, we'll use the original endpoint and adapt parsing.
                response = requests.get(
                    f"{self.uspto_base_url}/applications",
                    params={
                        "searchText": tech_category_query,
                        "start": start_date.strftime("%Y%m%d"),
                        "end": end_date.strftime("%Y%m%d"),
                        "rows": params["rows"],
                    },
                    timeout=30,
                )
                response.raise_for_status()
                api_response_json = response.json()

                patents_page = self._parse_uspto_response(api_response_json)
                if not patents_page:
                    break  # No more results
                all_patents.extend(patents_page)

                # Simplified pagination check - real APIs provide totalResults or similar
                if (
                    len(patents_page) < params["rows"]
                    or "results" not in api_response_json  # Check in the original JSON
                    or len(api_response_json["results"]) < params["rows"]
                ):
                    break
                current_page += 1
                time.sleep(0.5)  # Rate limiting

            except requests.RequestException as e:
                print(f"USPTO API error on page {current_page}: {e}")
                break  # Stop on error
            except json.JSONDecodeError as e:
                print(
                    f"USPTO API JSON parsing error on page {current_page} for query '{tech_category_query}': {e} - Response: {response.text if 'response' in locals() else 'N/A'}"
                )
                break

        return self._validate_patent_data(all_patents)

    def _parse_uspto_response(
        self, api_response_json: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parses the JSON response from a conceptual USPTO API call.

        Args:
            api_response_json: The JSON data as a Python dictionary from the API.

        Returns:
            A list of dictionaries, each representing a parsed patent.
        """
        patents: List[Dict[str, Any]] = []
        # The structure of 'api_response_json' depends heavily on the actual API endpoint used.
        # Common structures for results list are 'results' or nested like 'response.docs'.
        patent_items: List[Dict[str, Any]] = api_response_json.get(
            "results", api_response_json.get("response", {}).get("docs", [])
        )

        for patent_item in patent_items:
            patent_id: Optional[str] = patent_item.get(
                "applicationNumberText",
                patent_item.get("patentNumber", patent_item.get("publicationNumber")),
            )
            title: Optional[str] = patent_item.get("inventionTitle", patent_item.get("title"))

            filing_date_str: Optional[str] = patent_item.get("filingDate")
            parsed_filing_date: Optional[str] = None
            if isinstance(filing_date_str, str):
                try:
                    # Handle date formats like YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD
                    if "T" in filing_date_str:
                        parsed_filing_date = datetime.fromisoformat(
                            filing_date_str.replace("Z", "+00:00")
                        ).strftime("%Y-%m-%d")
                    else:
                        parsed_filing_date = pd.to_datetime(filing_date_str).strftime(
                            "%Y-%m-%d"
                        )
                except ValueError:
                    # Optional: Log warning about parsing error for specific patent_id
                    pass  # Keep as None if unparseable

            inventors_data: List[Any] = patent_item.get("inventors", [])
            num_inventors: int = (
                len(inventors_data) if isinstance(inventors_data, list) else 0
            )

            assignee_data: List[Dict[str, Any]] = patent_item.get("assignees", [{}])
            assignee_name_from_list: Optional[str] = None
            if isinstance(assignee_data, list) and assignee_data:
                assignee_name_from_list = assignee_data[0].get("name")

            assignee: Optional[str] = patent_item.get(
                "assigneeEntityName",
                patent_item.get("applicantName", assignee_name_from_list),
            )
            if isinstance(assignee, list) and assignee:
                assignee = assignee[0]

            main_class: Optional[str] = None
            classifications: Any = patent_item.get( # Can be list or dict based on API
                "classificationsIpc",
                patent_item.get(
                    "cpcClassification", patent_item.get("classificationNationalCurrent", [])
                ),
            )
            if isinstance(classifications, list) and classifications:
                main_class_obj: Dict[str, Any] = next(
                    (
                        c
                        for c in classifications
                        if c.get("isMain", False) or c.get("sequence", 0) == 0
                    ),
                    classifications[0],
                )
                main_class = main_class_obj.get(
                    "symbol",
                    main_class_obj.get("classificationSymbolText", {}).get("text"),
                )
            elif isinstance(classifications, dict):
                main_class = classifications.get(
                    "symbol",
                    classifications.get("classificationSymbolText", {}).get("text"),
                )

            num_citations: int = len(
                patent_item.get("referencedBy", patent_item.get("citation", []))
            )

            patents.append(
                {
                    "patent_id": str(patent_id) if patent_id else None,
                    "title": str(title) if title else None,
                    "filing_date": parsed_filing_date,
                    "inventors_count": num_inventors,
                    "assignee": str(assignee) if assignee else None,
                    "tech_class": str(main_class) if main_class else None,
                    "citations_count": num_citations,
                    "source": "USPTO",
                }
            )
        return patents

    def _validate_patent_data(
        self, patents_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validates a list of parsed patent data dictionaries.

        Args:
            patents_list: A list of patent data dictionaries.

        Returns:
            A list containing only the valid patent data dictionaries.
        """
        valid_patents: List[Dict[str, Any]] = []
        if not isinstance(patents_list, list):
            # Consider logging this error
            print("Validation error: input patents_list is not a list.")
            return []

        for patent in patents_list:
            if not isinstance(patent, dict):
                continue

            if (
                patent.get("patent_id")
                and patent.get("filing_date")
                and patent.get("title")
            ):
                try:
                    if isinstance(patent["filing_date"], str):
                        datetime.strptime(patent["filing_date"], "%Y-%m-%d")
                    valid_patents.append(patent)
                except (ValueError, TypeError): # Catches parsing errors for date
                    # Optional: Log invalid date format for a patent
                    pass
            # else:
                # Optional: Log missing critical fields for a patent
        return valid_patents


class FundingDataCollector:
    """Collects funding data, primarily from Crunchbase."""

    def __init__(self, crunchbase_key: str):
        self.crunchbase_key: str = crunchbase_key
        self.crunchbase_api_base: str = "https://api.crunchbase.com/api/v4"
        self.rate_limit_delay: float = (
            1.1  # Seconds between requests to adhere to rate limits
        )

    def collect_funding_rounds(
        self,
        announced_after_date_str: str,
        category_uuids_or_keywords: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Collects funding round data from Crunchbase API for specified categories.

        Args:
            announced_after_date_str: Date string 'YYYY-MM-DD' to filter rounds announced after this date.
            category_uuids_or_keywords: A list of Crunchbase category UUIDs or keywords to search for.

        Returns:
            A list of dictionaries, where each dictionary represents a funding round.
            Returns an empty list if API errors occur or no rounds are found.
        """
        all_rounds: List[Dict[str, Any]] = []
        search_endpoint: str = f"{self.crunchbase_api_base}/searches/funding_rounds"
        headers = {
            "X-cb-user-key": self.crunchbase_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        for category_identifier in category_uuids_or_keywords:
            after_id = None  # For pagination using 'after_id'
            page_count = 0
            max_pages = 20  # Safety limit for pages per category

            while page_count < max_pages:
                query_body = {
                    "field_ids": [
                        "money_raised",
                        "funded_organization",
                        "announced_on",
                        "investment_type",
                        "investor_identifiers",
                        "funded_organization_categories",
                        "funded_organization_location_identifiers",
                    ],
                    "order": [{"field_id": "announced_on", "sort": "desc"}],
                    "query": [
                        {
                            "type": "predicate",
                            "field_id": "announced_on",
                            "operator_id": "gte",
                            "values": [announced_after_date_str],
                        },
                        # Example predicate for category (if 'category_identifier' is a UUID)
                        # This assumes funded_organization_categories contains UUIDs.
                        # If it's a keyword, you might search funded_organization.short_description or similar.
                        {
                            "type": "predicate",
                            "field_id": "funded_organization_categories",
                            "operator_id": "includes",
                            "values": [category_identifier],
                        },
                        # Or for keywords in organization name/description:
                        # {"type": "predicate", "field_id": "funded_organization", "operator_id": "contains", "values": [category_identifier]}
                    ],
                    "limit": 100,  # Crunchbase max limit is often 100 or 1000, check docs
                }
                if after_id:
                    query_body["after_id"] = after_id

                try:
                    response = requests.post(
                        search_endpoint, headers=headers, json=query_body, timeout=45
                    )
                    response.raise_for_status()
                    api_response_json = response.json()

                    funding_round_entities: List[Dict[str, Any]] = api_response_json.get("entities", [])
                    if not funding_round_entities:
                        break

                    validated_page_rounds = self._validate_funding_data(
                        funding_round_entities
                    )
                    all_rounds.extend(validated_page_rounds)

                    # Pagination for Crunchbase v4: use 'after_id' from the last entity of the current page.
                    last_entity_uuid: Optional[str] = funding_round_entities[-1].get("uuid")
                    if len(funding_round_entities) < query_body["limit"] or not last_entity_uuid:
                        break
                    after_id = last_entity_uuid

                    page_count += 1
                    time.sleep(self.rate_limit_delay)

                except requests.RequestException as e:
                    print(
                        f"Crunchbase API error for category '{category_identifier}', page {page_count}: {e} - Response: {response.text if 'response' in locals() else 'N/A'}"
                    )
                    break  # Stop for this category on error
                except json.JSONDecodeError as e:
                    print(
                        f"Crunchbase JSON parsing error for '{category_identifier}', page {page_count}: {e} - Response: {response.text}"
                    )
                    break

        return all_rounds

    def _validate_funding_data(
        self, funding_round_entities_page: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validates a page of funding round entities from Crunchbase response.

        Args:
            funding_round_entities_page: A list of funding round 'entity' dictionaries.

        Returns:
            A list of valid funding round dictionaries with standardized keys.
        """
        valid_rounds: List[Dict[str, Any]] = []
        for funding_round_entity in funding_round_entities_page:
            properties: Dict[str, Any] = funding_round_entity.get("properties", {})
            money_raised_obj: Optional[Dict[str, Any]] = properties.get("money_raised")

            funded_org_data: Dict[str, Any] = properties.get("funded_organization", {})
            org_identifier_obj: Dict[str, Any] = funded_org_data.get("identifier", {})
            org_uuid: Optional[str] = org_identifier_obj.get("uuid")
            org_name: Optional[str] = funded_org_data.get("name", org_identifier_obj.get("value"))

            announced_on_date: Optional[str] = properties.get("announced_on")
            value_usd: Optional[float] = money_raised_obj.get("value_usd") if money_raised_obj else None


            if (value_usd is not None and announced_on_date and org_uuid and org_name):
                valid_rounds.append(
                    {
                        "funding_round_uuid": funding_round_entity.get("uuid"),
                        "company_uuid": org_uuid,
                        "company_name": org_name,
                        "amount_usd": money_raised_obj["value_usd"],
                        "currency": money_raised_obj.get(
                            "currency", "USD"
                        ),  # Default to USD
                        "date": properties[
                            "announced_on"
                        ],  # Store as YYYY-MM-DD string
                        "stage": properties.get("investment_type", "unknown"),
                        "source": "Crunchbase",
                    }
                )
            # else:
            # print(f"Skipping funding round due to missing critical data: {entity.get('uuid')}")
        return valid_rounds


class ResearchDataCollector:
    """Collects research paper data from sources like arXiv and PubMed."""

    def __init__(self, pubmed_api_key: Optional[str] = None):
        self.arxiv_client: arxiv.Client = arxiv.Client(
            page_size=200, delay_seconds=3.1, num_retries=3
        )
        self.pubmed_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.pubmed_api_key: Optional[str] = pubmed_api_key

    def collect_arxiv_papers(
        self, arxiv_categories_or_queries: List[str], days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Collects recent papers from arXiv based on categories or search queries.

        Args:
            arxiv_categories_or_queries: A list of arXiv categories (e.g., 'cs.AI')
                                         or search queries.
            days_back: How many days back to search for papers.

        Returns:
            A list of dictionaries, each representing an arXiv paper.
        """
        all_papers: List[Dict[str, Any]] = []
        cutoff_date: datetime = datetime.now(timezone.utc) - timedelta(days=days_back)

        for query_term in arxiv_categories_or_queries:
            search_query: str = (
                query_term
                if ":" in query_term or "cat:" in query_term
                else f'all:"{query_term}"'
            )

            try:
                search = arxiv.Search(
                    query=search_query,
                    max_results=500,  # Max results per query; consider pagination for more
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                )
                arxiv_results: List[arxiv.Result] = list(self.arxiv_client.results(search))

                for arxiv_paper in arxiv_results:
                    if arxiv_paper.published >= cutoff_date:
                        all_papers.append(
                            {
                                "paper_id": arxiv_paper.entry_id,
                                "title": arxiv_paper.title,
                                "authors": [str(author) for author in arxiv_paper.authors],
                                "abstract": arxiv_paper.summary.replace("\n", " ").strip(),
                                "categories": arxiv_paper.categories,
                                "published_date": arxiv_paper.published.isoformat(),
                                "pdf_url": arxiv_paper.pdf_url,
                                "doi": arxiv_paper.doi,
                                "journal_ref": arxiv_paper.journal_ref,
                                "citation_count_external": 0,  # Placeholder
                                "source": "arXiv",
                            }
                        )
            except Exception as e:
                print(f"Error collecting arXiv papers for query '{query_term}': {e}")
                continue

        return self._validate_research_data(all_papers)

    def _fetch_pubmed_details_batch(
        self, pmids_batch: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Fetches full paper details for a batch of PubMed IDs (PMIDs).

        Args:
            pmids_batch: A list of PubMed ID strings.

        Returns:
            A list of dictionaries, each representing a PubMed paper's details.
        """
        if not pmids_batch:
            return []

        details_list: List[Dict[str, Any]] = []
        fetch_url: str = f"{self.pubmed_base_url}/efetch.fcgi"
        fetch_params: Dict[str, Any] = {
            "db": "pubmed", "id": ",".join(pmids_batch),
            "retmode": "xml", "rettype": "abstract",
        }
        if self.pubmed_api_key:
            fetch_params["api_key"] = self.pubmed_api_key

        try:
            response = requests.get(fetch_url, params=fetch_params, timeout=60)
            response.raise_for_status()
            xml_root: ET.Element = ET.fromstring(response.content)

            for pubmed_article_xml in xml_root.findall(".//PubmedArticle"):
                pmid_el: Optional[ET.Element] = pubmed_article_xml.find(".//PMID")
                title_el: Optional[ET.Element] = pubmed_article_xml.find(".//ArticleTitle")

                abstract_texts: List[str] = [
                    seg.text for seg in pubmed_article_xml.findall(".//AbstractText") if seg.text
                ]
                full_abstract: Optional[str] = " ".join(abstract_texts).strip() if abstract_texts else None

                pub_date_iso: Optional[str] = None
                pub_date_node: Optional[ET.Element] = pubmed_article_xml.find(
                    './/PubmedData/History/PubMedPubDate[@PubStatus="pubmed"]'
                ) or pubmed_article_xml.find(
                    ".//ArticleDate"
                ) or pubmed_article_xml.find(".//PubDate")

                if pub_date_node is not None:
                    year_str: Optional[str] = pub_date_node.findtext("Year")
                    month_str: Optional[str] = pub_date_node.findtext("Month")
                    day_str: Optional[str] = pub_date_node.findtext("Day")
                    if year_str and month_str and day_str:
                        try:
                            if not month_str.isdigit(): # Convert month name/abbr to number
                                month_val = datetime.strptime(month_str, "%b").month if len(month_str) == 3 else datetime.strptime(month_str, "%B").month
                                month_str = str(month_val)
                            pub_date_iso = datetime(
                                int(year_str), int(month_str), int(day_str)
                            ).isoformat()
                        except ValueError:
                            pass

                authors_list: List[str] = [
                    f"{auth.findtext('LastName', default='').strip()} {auth.findtext('Initials', default='').strip()}".strip()
                    for auth in pubmed_article_xml.findall(".//AuthorList/Author")
                    if auth.findtext('LastName') or auth.findtext('Initials')
                ]

                doi_el: Optional[ET.Element] = pubmed_article_xml.find(".//ArticleId[@IdType='doi']")

                details_list.append({
                        "paper_id": pmid_el.text if pmid_el is not None else None,
                        "title": title_el.text if title_el is not None else "N/A",
                        "authors": authors_list,
                        "abstract": full_abstract,
                        "published_date": pub_date_iso,
                        "doi": doi_el.text if doi_el is not None else None,
                        "journal_title": pubmed_article_xml.findtext(".//ISOAbbreviation") or pubmed_article_xml.findtext(".//Journal/Title"),
                        "citation_count_external": 0,
                        "source": "PubMed",
                    }
                )
            time.sleep(0.11 if self.pubmed_api_key else 0.35)
        except requests.RequestException as e:
            print(f"Error fetching PubMed details batch for PMIDs '{','.join(pmids_batch)}': {e}")
        except ET.ParseError as e:
            print(f"Error parsing PubMed XML for PMIDs '{','.join(pmids_batch)}': {e} - Response: {response.text if 'response' in locals() else 'N/A'}")
        return details_list

    def collect_pubmed_papers(
        self, search_terms_list: List[str], days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Collects biomedical papers from PubMed using eSearch then eFetch.

        Args:
            search_terms_list: A list of search terms/queries for PubMed.
            days_back: How many days back to search for papers.

        Returns:
            A list of dictionaries, each representing a PubMed paper.
        """
        all_papers_details: List[Dict[str, Any]] = []
        max_date_str: str = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        min_date_str: str = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y/%m/%d")
        date_query_part: str = f'AND ("{min_date_str}"[Date - Publication] : "{max_date_str}"[Date - Publication])'

        for term in search_terms_list:
            pmids_for_term: List[str] = []
            try:
                search_url: str = f"{self.pubmed_base_url}/esearch.fcgi"
                retstart_idx: int = 0
                retmax_batch: int = 500
                total_retrieved_for_term: int = 0
                expected_total_for_term: int = 1

                while total_retrieved_for_term < expected_total_for_term:
                    search_params: Dict[str, Any] = {
                        "db": "pubmed", "term": f"({term}) {date_query_part}",
                        "retmax": retmax_batch, "retstart": retstart_idx,
                        "sort": "pub_date", "retmode": "json",
                    }
                    if self.pubmed_api_key:
                        search_params["api_key"] = self.pubmed_api_key

                    response = requests.get(search_url, params=search_params, timeout=45)
                    response.raise_for_status()
                    api_response_json: Dict[str, Any] = response.json()

                    pubmed_esearch_result: Dict[str, Any] = api_response_json.get("esearchresult", {})
                    current_batch_pmids: List[str] = pubmed_esearch_result.get("idlist", [])
                    pmids_for_term.extend(current_batch_pmids)

                    if retstart_idx == 0:
                        expected_total_for_term = int(pubmed_esearch_result.get("count", 0))
                        if expected_total_for_term == 0:
                            break

                    total_retrieved_for_term += len(current_batch_pmids)

                    if not current_batch_pmids or total_retrieved_for_term >= expected_total_for_term:
                        break

                    retstart_idx += retmax_batch
                    time.sleep(0.11 if self.pubmed_api_key else 0.35)

                if pmids_for_term:
                    for i in range(0, len(pmids_for_term), 100):
                        batch_to_fetch: List[str] = pmids_for_term[i : i + 100]
                        paper_details_batch: List[Dict[str, Any]] = self._fetch_pubmed_details_batch(batch_to_fetch)
                        all_papers_details.extend(paper_details_batch)

            except requests.RequestException as e:
                print(f"Request error during PubMed search for '{term}': {e} - URL: {response.url if 'response' in locals() else search_url}")
            except json.JSONDecodeError as e:
                print(f"JSON parsing error during PubMed search for '{term}': {e} - Response text: {response.text if 'response' in locals() else 'N/A'}")
            except Exception as e:
                print(f"An unexpected error occurred while collecting PubMed papers for '{term}': {e}")

        return self._validate_research_data(all_papers_details)

    def _validate_research_data(
        self, papers_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validates a list of parsed research paper data dictionaries.

        Args:
            papers_list: A list of research paper data dictionaries.

        Returns:
            A list containing only the valid paper data dictionaries.
        """
        valid_papers: List[Dict[str, Any]] = []
        if not isinstance(papers_list, list):
            print("Validation error: input for _validate_research_data is not a list.")
            return []

        for paper in papers_list:
            if not isinstance(paper, dict):
                continue

            paper_id: Optional[str] = paper.get("paper_id")
            title: Optional[str] = paper.get("title")
            abstract: Optional[str] = paper.get("abstract")
            published_date: Optional[str] = paper.get("published_date")
            authors: List[Any] = paper.get("authors", [])

            if (paper_id and title and title not in ["N/A", ""] and
                abstract and abstract not in ["N/A", ""] and
                published_date and isinstance(authors, list) and authors):
                # Optional: Add more validation, e.g., abstract length.
                # if 'min_abstract_length' in research_config and len(abstract) < research_config['min_abstract_length']:
                #     continue
                valid_papers.append(paper)
        return valid_papers
