import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# TextBlob and networkx were imported but not directly used in the provided FeatureEngineer snippet.
# Keeping them commented out for now. If specific methods using them are added, they can be uncommented.
# from textblob import TextBlob
# import networkx as nx
import nltk
import re # For basic cleaning if needed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from innovation_system.config.settings import feature_config as global_feature_config # Import for default

def _ensure_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("NLTK 'stopwords' resource not found. Downloading...")
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' resource not found. Downloading...")
        nltk.download('punkt', quiet=True)

_ensure_nltk_resources() # Call it once when the module is loaded


class FeatureEngineer:
    def __init__(self, config=None):
        self.scaler = StandardScaler()
        self.feature_config = config if config is not None else global_feature_config
        # self.tech_categories = self._load_tech_taxonomy() # Conceptual

    def _load_tech_taxonomy(self):
        return {
            "Artificial Intelligence": ["G06N", "AI", "machine learning", "deep learning"],
            "Biotechnology": ["C12N", "biotech", "genetic engineering"],
        }

    def create_patent_features(self, patents_df):
        if patents_df.empty: return pd.DataFrame()
        patents_df['filing_date'] = pd.to_datetime(patents_df['filing_date'])
        features = {}
        features['filing_rate_3m'] = self._calculate_event_rate(patents_df, 'filing_date', months=3)
        features['filing_rate_6m'] = self._calculate_event_rate(patents_df, 'filing_date', months=6)
        features['filing_rate_12m'] = self._calculate_event_rate(patents_df, 'filing_date', months=12)
        features['tech_diversity_shannon'] = self._calculate_tech_diversity(patents_df, 'tech_class')
        features['unique_inventor_count'] = patents_df['inventors'].explode().nunique() if 'inventors' in patents_df.columns and not patents_df['inventors'].empty else 0
        features['citation_velocity_avg'] = self._calculate_citation_velocity(patents_df) if 'citations' in patents_df.columns else 0
        features['forward_citation_rate_avg'] = patents_df.get('forward_citations', pd.Series(0)).mean()
        features['international_filing_ratio'] = self._calculate_international_ratio(patents_df, 'source')
        return pd.DataFrame([features])

    def create_funding_features(self, funding_df):
        if funding_df.empty: return pd.DataFrame()
        funding_df['date'] = pd.to_datetime(funding_df['date'])
        features = {}
        features['funding_deals_velocity_3m'] = self._calculate_event_rate(funding_df, 'date', months=3)
        features['funding_deals_velocity_6m'] = self._calculate_event_rate(funding_df, 'date', months=6)
        features['funding_amount_velocity_3m_usd'] = self._calculate_sum_rate(funding_df, 'date', 'amount_usd', months=3)
        features['funding_amount_velocity_6m_usd'] = self._calculate_sum_rate(funding_df, 'date', 'amount_usd', months=6)
        features['avg_round_size_usd'] = funding_df['amount_usd'].mean() if not funding_df['amount_usd'].empty else 0
        features['median_round_size_usd'] = funding_df['amount_usd'].median() if not funding_df['amount_usd'].empty else 0
        features['funding_amount_gini'] = self._calculate_gini_coefficient(funding_df['amount_usd']) if not funding_df['amount_usd'].empty else 0
        if 'stage' in funding_df.columns and not funding_df['stage'].empty:
            stage_dist = funding_df['stage'].value_counts(normalize=True)
            features['seed_ratio'] = stage_dist.get('seed', stage_dist.get('Seed', 0))
            features['series_a_ratio'] = stage_dist.get('series_a', stage_dist.get('Series A', 0))
            late_stage_keys = ['series_c', 'series_d', 'series_e', 'late_stage_venture', 'Series C', 'Series D', 'Series E']
            features['late_stage_ratio'] = sum(stage_dist.get(key, 0) for key in late_stage_keys)
        else:
            features['seed_ratio'] = 0; features['series_a_ratio'] = 0; features['late_stage_ratio'] = 0
        return pd.DataFrame([features])

    def create_research_features(self, papers_df):
        if papers_df.empty: return pd.DataFrame()
        papers_df['published_date'] = pd.to_datetime(papers_df['published_date'])
        features = {}
        features['publication_rate_3m'] = self._calculate_event_rate(papers_df, 'published_date', months=3)
        features['publication_rate_6m'] = self._calculate_event_rate(papers_df, 'published_date', months=6)
        features['avg_citation_count'] = papers_df['citation_count'].mean() if 'citation_count' in papers_df.columns and not papers_df['citation_count'].empty else 0
        features['avg_authors_per_paper'] = papers_df['authors'].apply(len).mean() if 'authors' in papers_df.columns and not papers_df['authors'].empty else 0
        features['category_diversity_shannon'] = self._calculate_tech_diversity(papers_df, 'categories', explode_list=True) if 'categories' in papers_df.columns else 0

        # NLP Features
        if 'abstract' in papers_df.columns:
            papers_df['abstract_length'] = papers_df['abstract'].fillna('').apply(len)
        else:
            papers_df['abstract_length'] = 0

        keywords_from_config = [kw.lower() for kw in self.feature_config.get('emerging_tech_keywords', [])]
        stop_words_set = set(stopwords.words('english'))

        def count_kws(abstract_text):
            if pd.isna(abstract_text) or not abstract_text:
                return 0
            text = abstract_text.lower()
            try:
                tokens = word_tokenize(text)
            except Exception:
                tokens = text.split()

            processed_tokens = [word for word in tokens if word.isalnum() and word not in stop_words_set]

            current_kw_count = 0
            for keyword_to_find in keywords_from_config:
                current_kw_count += processed_tokens.count(keyword_to_find)
            return current_kw_count

        if 'abstract' in papers_df.columns and keywords_from_config:
            papers_df['keyword_count'] = papers_df['abstract'].apply(count_kws)
        else:
            papers_df['keyword_count'] = 0

        features['avg_abstract_length'] = papers_df['abstract_length'].mean() if 'abstract_length' in papers_df.columns and not papers_df['abstract_length'].empty else 0
        features['avg_keyword_count'] = papers_df['keyword_count'].mean() if 'keyword_count' in papers_df.columns and not papers_df['keyword_count'].empty else 0
        features['sum_keyword_count'] = papers_df['keyword_count'].sum() if 'keyword_count' in papers_df.columns and not papers_df['keyword_count'].empty else 0

        return pd.DataFrame([features])

    def _calculate_event_rate(self, df, date_column, months):
        if df.empty or date_column not in df.columns: return 0
        # Ensure date_column is timezone-naive before comparison with pd.Timestamp.now() which is also naive by default here.
        # Or make both timezone-aware consistently. Assuming naive for now.
        df[date_column] = pd.to_datetime(df[date_column]).dt.tz_localize(None)
        cutoff_date = pd.Timestamp.now().tz_localize(None) - pd.DateOffset(months=months)
        recent_events = df[df[date_column] >= cutoff_date]
        return len(recent_events) / max(months, 1)

    def _calculate_sum_rate(self, df, date_column, value_column, months):
        if df.empty or date_column not in df.columns or value_column not in df.columns: return 0
        df[date_column] = pd.to_datetime(df[date_column]).dt.tz_localize(None)
        cutoff_date = pd.Timestamp.now().tz_localize(None) - pd.DateOffset(months=months)
        recent_df = df[df[date_column] >= cutoff_date]
        total_sum = recent_df[value_column].sum()
        return total_sum / max(months, 1)

    _calculate_filing_rate = _calculate_event_rate
    _calculate_publication_rate = _calculate_event_rate
    _calculate_funding_velocity = _calculate_event_rate

    def _calculate_tech_diversity(self, df, category_column, explode_list=False):
        if df.empty or category_column not in df.columns or df[category_column].empty: return 0
        if explode_list:
            all_categories = df[category_column].explode().dropna()
            if all_categories.empty: return 0
            counts = all_categories.value_counts()
        else:
            counts = df[category_column].value_counts()
            if counts.empty: return 0
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _calculate_citation_velocity(self, df):
        if df.empty or 'citations' not in df.columns or 'filing_date' not in df.columns: return 0
        df_copy = df.copy()
        df_copy['filing_date'] = pd.to_datetime(df_copy['filing_date']).dt.tz_localize(None)
        now_ts = pd.Timestamp.now().tz_localize(None)
        df_copy['months_since_filing'] = (now_ts - df_copy['filing_date']).dt.days / 30.44
        df_copy['citation_rate'] = df_copy['citations'] / (df_copy['months_since_filing'].replace(0, 1))
        valid_rates = df_copy['citation_rate'][np.isfinite(df_copy['citation_rate'])]
        return valid_rates.mean() if not valid_rates.empty else 0

    def _calculate_international_ratio(self, df, source_column_name='source'):
        if df.empty or source_column_name not in df.columns or df[source_column_name].nunique() <= 1: return 0
        international_count = df[~df[source_column_name].astype(str).str.upper().isin(['USPTO', 'US'])].shape[0]
        total_count = df.shape[0]
        return international_count / total_count if total_count > 0 else 0

    def _calculate_gini_coefficient(self, values_series):
        if values_series.empty or values_series.isnull().all() or len(values_series) < 2: return 0
        arr = np.array(values_series.dropna().astype(float))
        if np.any(arr < 0): arr = arr[arr >=0]
        if len(arr) < 2 : return 0
        arr = np.sort(arr)
        index = np.arange(1, arr.shape[0] + 1)
        n = arr.shape[0]
        if n == 0 or np.sum(arr) == 0: return 0
        return ((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))
