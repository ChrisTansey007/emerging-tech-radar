# Cleared for new implementation
# This file will house the FeatureEngineer class.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from textblob import TextBlob # Note: TextBlob was imported in feature engineering but not directly used; keeping for potential sentiment analysis tasks.
import networkx as nx # Note: networkx was imported in feature engineering but not directly used in the snippet; keeping for potential network analysis.
from datetime import datetime # Required for pd.Timestamp.now()

# --- Feature Engineering Pipeline ---

class FeatureEngineer:
    def __init__(self):
        self.standard_scaler = StandardScaler() # For z-score normalization
        self.min_max_scaler = MinMaxScaler() # For 0-1 normalization
        # self.tech_taxonomy = self._load_tech_taxonomy() # Conceptual

    def _load_tech_taxonomy(self): # Placeholder for loading a technology mapping
        return {"AI": ["G06N", "artificial intelligence", "machine learning"], "Biotech": ["C12N", "biotechnology"]}

    def _preprocess_dataframe_dates(self, df, date_column_name):
        if df.empty or date_column_name not in df.columns: return df
        # Convert to datetime, making them timezone-naive for consistent comparison with pd.Timestamp.now() (naive)
        # If original dates are timezone-aware, convert to UTC then remove tz for naive comparison,
        # or make pd.Timestamp.now() timezone-aware. For simplicity, using naive.
        try:
            df[date_column_name] = pd.to_datetime(df[date_column_name], errors='coerce').dt.tz_localize(None)
        except TypeError as e: # Already naive
            # Check if the first element is a Timestamp and is naive
            if not df.empty and isinstance(df[date_column_name].iloc[0], pd.Timestamp) and df[date_column_name].iloc[0].tzinfo is None:
                pass # Already naive
            else: # Could be object type from mixed sources, try harder
                 df[date_column_name] = pd.to_datetime(df[date_column_name], errors='coerce')
        except AttributeError: # Happens if series is already datetime but tz_localize is called on non-datetime accessor
            if pd.api.types.is_datetime64_any_dtype(df[date_column_name]):
                 if df[date_column_name].dt.tz is not None: # If it's timezone-aware
                     df[date_column_name]] = df[date_column_name].dt.tz_localize(None)
                 # else it's already naive, do nothing
            else: # Not datetime, try to convert
                df[date_column_name] = pd.to_datetime(df[date_column_name], errors='coerce')


        return df.dropna(subset=[date_column_name]) # Remove rows where date couldn't be parsed

    def create_patent_features(self, patents_df_orig):
        if patents_df_orig.empty: return pd.DataFrame()
        patents_df = self._preprocess_dataframe_dates(patents_df_orig.copy(), 'filing_date')
        if patents_df.empty: return pd.DataFrame() # All rows dropped due to bad dates

        # Time reference for rate calculations (naive UTC for consistency if dates are naive)
        now_timestamp = pd.Timestamp.now(tz=None)
        features = {}

        features['filing_rate_3m'] = self._calculate_event_rate(patents_df, 'filing_date', now_timestamp, months=3)
        features['filing_rate_12m'] = self._calculate_event_rate(patents_df, 'filing_date', now_timestamp, months=12)
        features['tech_diversity_shannon'] = self._calculate_shannon_diversity(patents_df, 'tech_class')
        features['avg_inventors_count'] = patents_df['inventors_count'].mean() if 'inventors_count' in patents_df else 0

        # Citation velocity (citations per month since filing)
        if 'citations_count' in patents_df and 'filing_date' in patents_df:
            # Ensure filing_date is datetime
            patents_df['months_since_filing'] = ((now_timestamp - patents_df['filing_date']).dt.days / 30.44).clip(lower=1) # Min 1 month
            patents_df['citation_rate_monthly'] = patents_df['citations_count'] / patents_df['months_since_filing']
            features['citation_velocity_avg'] = patents_df['citation_rate_monthly'][np.isfinite(patents_df['citation_rate_monthly'])].mean()
        else: features['citation_velocity_avg'] = 0

        features['international_filing_ratio'] = self._calculate_international_patent_ratio(patents_df, 'source')
        return pd.DataFrame([features])

    def create_funding_features(self, funding_df_orig):
        if funding_df_orig.empty: return pd.DataFrame()
        funding_df = self._preprocess_dataframe_dates(funding_df_orig.copy(), 'date')
        if funding_df.empty: return pd.DataFrame()

        now_timestamp = pd.Timestamp.now(tz=None)
        features = {}

        features['funding_deals_velocity_3m'] = self._calculate_event_rate(funding_df, 'date', now_timestamp, months=3)
        features['funding_amount_velocity_3m_usd'] = self._calculate_sum_in_period(funding_df, 'date', 'amount_usd', now_timestamp, months=3) / 3.0
        features['avg_round_size_usd'] = funding_df['amount_usd'].mean() if 'amount_usd' in funding_df else 0
        features['funding_amount_gini'] = self._calculate_gini_coefficient(funding_df['amount_usd']) if 'amount_usd' in funding_df else 0

        if 'stage' in funding_df and not funding_df['stage'].empty:
            stage_dist = funding_df['stage'].str.lower().value_counts(normalize=True)
            features['seed_stage_ratio'] = stage_dist.get('seed', 0) + stage_dist.get('pre-seed', 0)
            features['series_a_ratio'] = stage_dist.get('series a', 0)
            late_keys = ['series b', 'series c', 'series d', 'series e', 'late stage venture', 'private equity']
            features['late_stage_ratio'] = sum(stage_dist.get(k, 0) for k in late_keys)
        else: features.update({'seed_stage_ratio':0, 'series_a_ratio':0, 'late_stage_ratio':0})

        return pd.DataFrame([features])

    def create_research_features(self, papers_df_orig):
        if papers_df_orig.empty: return pd.DataFrame()
        papers_df = self._preprocess_dataframe_dates(papers_df_orig.copy(), 'published_date')
        if papers_df.empty: return pd.DataFrame()

        now_timestamp = pd.Timestamp.now(tz=None)
        features = {}

        features['publication_rate_3m'] = self._calculate_event_rate(papers_df, 'published_date', now_timestamp, months=3)
        features['avg_citation_external'] = papers_df['citation_count_external'].mean() if 'citation_count_external' in papers_df else 0
        features['avg_authors_per_paper'] = papers_df['authors'].apply(len).mean() if 'authors' in papers_df and not papers_df['authors'].empty else 0

        # Category diversity (Shannon index on primary categories if available)
        # ArXiv 'categories' is a list, need to handle this. For simplicity, take first category or explode.
        if 'categories' in papers_df and not papers_df['categories'].empty:
            # Simple: use first category. Robust: explode list and count.
            papers_df['primary_category'] = papers_df['categories'].apply(lambda x: x[0] if isinstance(x, list) and x else None)
            features['research_category_diversity_shannon'] = self._calculate_shannon_diversity(papers_df, 'primary_category')
        else: features['research_category_diversity_shannon'] = 0

        return pd.DataFrame([features])

    def _calculate_event_rate(self, df, date_col, ref_time, months):
        """Rate of events (count) per month over the last 'months'."""
        if df.empty: return 0
        cutoff = ref_time - pd.DateOffset(months=months)
        count = df[df[date_col] >= cutoff].shape[0]
        return count / float(months) if months > 0 else 0

    def _calculate_sum_in_period(self, df, date_col, value_col, ref_time, months):
        """Sum of 'value_col' over the last 'months'."""
        if df.empty or value_col not in df.columns: return 0
        cutoff = ref_time - pd.DateOffset(months=months)
        return df.loc[df[date_col] >= cutoff, value_col].sum()

    def _calculate_shannon_diversity(self, df, category_col):
        if df.empty or category_col not in df.columns or df[category_col].dropna().empty: return 0
        counts = df[category_col].dropna().value_counts()
        if counts.empty: return 0
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12)) # Epsilon for log(0)

    def _calculate_international_patent_ratio(self, df, source_col):
        if df.empty or source_col not in df.columns or df[source_col].nunique() <=1: return 0
        # Example: 'USPTO' vs other sources. Needs clear source tagging.
        is_uspto = df[source_col].str.upper().isin(['USPTO', 'US'])
        non_uspto_count = (~is_uspto).sum()
        return non_uspto_count / float(len(df)) if len(df) > 0 else 0

    def _calculate_gini_coefficient(self, values_series_orig):
        if values_series_orig.empty or values_series_orig.isnull().all(): return 0
        values_series = values_series_orig.dropna().astype(float)
        if len(values_series) < 2 or values_series.sum() == 0: return 0
        arr = np.sort(np.array(values_series[values_series >= 0])) # Non-negative values
        if len(arr) < 2 or np.sum(arr) == 0: return 0
        idx = np.arange(1, len(arr) + 1)
        n = len(arr)
        return (np.sum((2 * idx - n - 1) * arr)) / (n * np.sum(arr))

```
