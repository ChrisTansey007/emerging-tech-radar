# Cleared for new implementation
# This file will house the FeatureEngineer class.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# from textblob import (
#     TextBlob,
# )  # Note: TextBlob was imported in feature engineering but not directly used; keeping for potential sentiment analysis tasks.
# import networkx as nx  # Note: networkx was imported in feature engineering but not directly used in the snippet; keeping for potential network analysis.
# from datetime import datetime  # Not strictly required as pd.Timestamp.now() is used

# --- Feature Engineering Pipeline ---

from typing import Dict, Any, List, Optional # Added for type hints

class FeatureEngineer:
    """
    Transforms raw data from patents, funding, and research into engineered features
    for predictive modeling.
    """
    def __init__(self):
        self.standard_scaler = StandardScaler()  # For z-score normalization
        self.min_max_scaler = MinMaxScaler()  # For 0-1 normalization
        # self.tech_taxonomy = self._load_tech_taxonomy() # Conceptual: Load from config or file

    def _load_tech_taxonomy(self) -> Dict[str, List[str]]:
        """
        Placeholder for loading a technology taxonomy mapping.
        In a real system, this would load from a configuration file or database.

        Returns:
            A dictionary mapping broad technology areas to lists of specific keywords or classification codes.
        """
        return {
            "AI": ["G06N", "artificial intelligence", "machine learning"],
            "Biotech": ["C12N", "biotechnology"],
        }

    def _preprocess_dataframe_dates(self, df: pd.DataFrame, date_column_name: str) -> pd.DataFrame:
        """
        Preprocesses a specified date column in a DataFrame.
        Converts column to datetime objects, localizes to None (naive), and drops rows with NaT dates.

        Args:
            df: The input DataFrame.
            date_column_name: The name of the column containing date information.

        Returns:
            The DataFrame with the processed date column, or the original DataFrame if input is empty
            or date column is not found. Rows with unparseable dates are dropped.
        """
        if df.empty or date_column_name not in df.columns:
            return df
        # Convert to datetime, making them timezone-naive for consistent comparison
        # with pd.Timestamp.now(tz=None) (naive)
        # If original dates are timezone-aware, convert to UTC then remove tz for naive comparison,
        # or make pd.Timestamp.now() timezone-aware. For simplicity, using naive.
        try:
            df[date_column_name] = pd.to_datetime(
                df[date_column_name], errors="coerce"
            ).dt.tz_localize(None)
        except TypeError as _:  # Already naive, F841: local variable 'e' is assigned to but never used
            # Check if the first element is a Timestamp and is naive
            if (
                not df.empty
                and isinstance(df[date_column_name].iloc[0], pd.Timestamp)
                and df[date_column_name].iloc[0].tzinfo is None
            ):
                pass  # Already naive
            else:  # Could be object type from mixed sources, try harder
                df[date_column_name] = pd.to_datetime(
                    df[date_column_name], errors="coerce"
                )
        except (
            AttributeError
        ):  # Happens if series is already datetime but tz_localize is called on non-datetime accessor
            if pd.api.types.is_datetime64_any_dtype(df[date_column_name]):
                if df[date_column_name].dt.tz is not None:  # If it's timezone-aware
                    df[date_column_name] = df[date_column_name].dt.tz_localize(None)
                # else it's already naive, do nothing
            else:  # Not datetime, try to convert
                df[date_column_name] = pd.to_datetime(
                    df[date_column_name], errors="coerce"
                )

        return df.dropna(
            subset=[date_column_name]
        )

    def create_patent_features(self, patents_df_orig: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features from patent data.

        Args:
            patents_df_orig: DataFrame containing raw patent data with columns like
                             'filing_date', 'tech_class', 'inventors_count',
                             'citations_count', 'source'.

        Returns:
            A DataFrame with engineered patent features. Returns an empty DataFrame
            if input is empty or processing results in no usable data.
        """
        if patents_df_orig.empty:
            return pd.DataFrame()

        patents_df: pd.DataFrame = self._preprocess_dataframe_dates(
            patents_df_orig.copy(), "filing_date"
        )
        if patents_df.empty:
            return pd.DataFrame()

        now_timestamp: pd.Timestamp = pd.Timestamp.now(tz=None)
        features: Dict[str, Any] = {}

        features["filing_rate_3m"] = self._calculate_event_rate(
            patents_df, "filing_date", now_timestamp, months=3
        )
        features["filing_rate_12m"] = self._calculate_event_rate(
            patents_df, "filing_date", now_timestamp, months=12
        )
        features["tech_diversity_shannon"] = self._calculate_shannon_diversity(
            patents_df, "tech_class"
        )
        features["avg_inventors_count"] = (
            patents_df["inventors_count"].mean()
            if "inventors_count" in patents_df
            else 0
        )

        # Citation velocity (citations per month since filing)
        if "citations_count" in patents_df and "filing_date" in patents_df:
            # Ensure filing_date is datetime
            patents_df["months_since_filing"] = (
                (now_timestamp - patents_df["filing_date"]).dt.days / 30.44
            ).clip(
                lower=1
            )  # Min 1 month
            patents_df["citation_rate_monthly"] = (
                patents_df["citations_count"] / patents_df["months_since_filing"]
            )
            features["citation_velocity_avg"] = patents_df["citation_rate_monthly"][
                np.isfinite(patents_df["citation_rate_monthly"])
            ].mean()
        else:
            features["citation_velocity_avg"] = 0

        features["international_filing_ratio"] = (
            self._calculate_international_patent_ratio(patents_df, "source")
        )
        return pd.DataFrame([features])

    def create_funding_features(self, funding_df_orig: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features from funding data.

        Args:
            funding_df_orig: DataFrame containing raw funding data with columns like
                             'date', 'amount_usd', 'stage'.

        Returns:
            A DataFrame with engineered funding features. Returns an empty DataFrame
            if input is empty or processing results in no usable data.
        """
        if funding_df_orig.empty:
            return pd.DataFrame()

        funding_df: pd.DataFrame = self._preprocess_dataframe_dates(funding_df_orig.copy(), "date")
        if funding_df.empty:
            return pd.DataFrame()

        now_timestamp: pd.Timestamp = pd.Timestamp.now(tz=None)
        features: Dict[str, Any] = {}

        features["funding_deals_velocity_3m"] = self._calculate_event_rate(
            funding_df, "date", now_timestamp, months=3
        )
        features["funding_amount_velocity_3m_usd"] = (
            self._calculate_sum_in_period(
                funding_df, "date", "amount_usd", now_timestamp, months=3
            )
            / 3.0
        )
        features["avg_round_size_usd"] = (
            funding_df["amount_usd"].mean() if "amount_usd" in funding_df else 0
        )
        features["funding_amount_gini"] = (
            self._calculate_gini_coefficient(funding_df["amount_usd"])
            if "amount_usd" in funding_df
            else 0
        )

        if "stage" in funding_df and not funding_df["stage"].empty:
            stage_dist = funding_df["stage"].str.lower().value_counts(normalize=True)
            features["seed_stage_ratio"] = stage_dist.get("seed", 0) + stage_dist.get(
                "pre-seed", 0
            )
            features["series_a_ratio"] = stage_dist.get("series a", 0)
            late_keys = [
                "series b",
                "series c",
                "series d",
                "series e",
                "late stage venture",
                "private equity",
            ]
            features["late_stage_ratio"] = sum(stage_dist.get(k, 0) for k in late_keys)
        else:
            features.update(
                {"seed_stage_ratio": 0, "series_a_ratio": 0, "late_stage_ratio": 0}
            )

        return pd.DataFrame([features])

    def create_research_features(self, papers_df_orig: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features from research paper data.

        Args:
            papers_df_orig: DataFrame containing raw research paper data with columns
                            like 'published_date', 'citation_count_external',
                            'authors', 'categories'.

        Returns:
            A DataFrame with engineered research features. Returns an empty DataFrame
            if input is empty or processing results in no usable data.
        """
        if papers_df_orig.empty:
            return pd.DataFrame()

        papers_df: pd.DataFrame = self._preprocess_dataframe_dates(
            papers_df_orig.copy(), "published_date"
        )
        if papers_df.empty:
            return pd.DataFrame()

        now_timestamp: pd.Timestamp = pd.Timestamp.now(tz=None)
        features: Dict[str, Any] = {}

        features["publication_rate_3m"] = self._calculate_event_rate(
            papers_df, "published_date", now_timestamp, months=3
        )
        features["avg_citation_external"] = (
            papers_df["citation_count_external"].mean()
            if "citation_count_external" in papers_df
            else 0
        )
        features["avg_authors_per_paper"] = (
            papers_df["authors"].apply(len).mean()
            if "authors" in papers_df and not papers_df["authors"].empty
            else 0
        )

        # Category diversity (Shannon index on primary categories if available)
        # ArXiv 'categories' is a list, need to handle this. For simplicity, take first category or explode.
        if "categories" in papers_df and not papers_df["categories"].empty:
            # Simple: use first category. Robust: explode list and count.
            papers_df["primary_category"] = papers_df["categories"].apply(
                lambda x: x[0] if isinstance(x, list) and x else None
            )
            features["research_category_diversity_shannon"] = (
                self._calculate_shannon_diversity(papers_df, "primary_category")
            )
        else:
            features["research_category_diversity_shannon"] = 0

        return pd.DataFrame([features])

    def _calculate_event_rate(
        self, df: pd.DataFrame, date_col: str, ref_time: pd.Timestamp, months: int
    ) -> float:
        """
        Calculates the rate of events (count) per month over a specified period.

        Args:
            df: The input DataFrame containing event data.
            date_col: The name of the column in 'df' that contains event dates.
            ref_time: The reference timestamp from which to look back.
            months: The number of months to look back for calculating the rate.

        Returns:
            The calculated event rate (events per month). Returns 0.0 if input DataFrame
            is empty or months is zero or less.
        """
        if df.empty or months <= 0:
            return 0.0
        cutoff: pd.Timestamp = ref_time - pd.DateOffset(months=months)
        count: int = df[df[date_col] >= cutoff].shape[0]
        return float(count) / months

    def _calculate_sum_in_period(
        self, df: pd.DataFrame, date_col: str, value_col: str, ref_time: pd.Timestamp, months: int
    ) -> float:
        """
        Calculates the sum of a specified value column over a defined period.

        Args:
            df: Input DataFrame.
            date_col: Name of the date column.
            value_col: Name of the column whose values are to be summed.
            ref_time: Reference timestamp to look back from.
            months: Number of months to look back.

        Returns:
            The sum of 'value_col' in the specified period. Returns 0.0 if input is empty,
            value column is missing, or months is zero or less.
        """
        if df.empty or value_col not in df.columns or months <= 0:
            return 0.0
        cutoff: pd.Timestamp = ref_time - pd.DateOffset(months=months)
        return float(df.loc[df[date_col] >= cutoff, value_col].sum())

    def _calculate_shannon_diversity(self, df: pd.DataFrame, category_col: str) -> float:
        """
        Calculates the Shannon diversity index for a categorical column.

        Args:
            df: Input DataFrame.
            category_col: Name of the categorical column.

        Returns:
            The Shannon diversity index. Returns 0.0 if data is insufficient.
        """
        if (
            df.empty
            or category_col not in df.columns
            or df[category_col].dropna().empty
        ):
            return 0.0
        counts: pd.Series = df[category_col].dropna().value_counts()
        if counts.empty:
            return 0.0
        probs: pd.Series = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-12)))  # Epsilon for log(0)

    def _calculate_international_patent_ratio(self, df: pd.DataFrame, source_col: str) -> float:
        """
        Calculates the ratio of non-USPTO patents to total patents.
        Assumes 'source_col' identifies the patent office (e.g., 'USPTO').

        Args:
            df: Input DataFrame with patent source information.
            source_col: Name of the column indicating the patent source.

        Returns:
            The ratio of international (non-USPTO) patents. Returns 0.0 if
            data is insufficient or only one source is present.
        """
        if df.empty or source_col not in df.columns or df[source_col].nunique(dropna=False) <= 1:
            return 0.0

        is_uspto: pd.Series = df[source_col].astype(str).str.upper().isin(["USPTO", "US"])
        non_uspto_count: int = (~is_uspto).sum()
        total_count: int = len(df)
        return float(non_uspto_count) / total_count if total_count > 0 else 0.0

    def _calculate_gini_coefficient(self, values_series_orig: pd.Series) -> float:
        """
        Calculates the Gini coefficient for a series of values.
        Handles NaNs, negative values (by filtering them out), and empty series.

        Args:
            values_series_orig: A pandas Series of numerical values.

        Returns:
            The Gini coefficient (float). Returns 0.0 if data is insufficient
            or values do not allow for calculation (e.g., all zeros).
        """
        if values_series_orig.empty or values_series_orig.isnull().all():
            return 0.0

        # Ensure values are float, drop NaNs, and keep only non-negative values
        values_series: pd.Series = values_series_orig.dropna().astype(float)
        values_series = values_series[values_series >= 0]

        if len(values_series) < 2 or values_series.sum() == 0:
            return 0.0

        # Sort values to prepare for Gini calculation
        arr: np.ndarray = np.sort(values_series.to_numpy())
        n: int = len(arr)
        index: np.ndarray = np.arange(1, n + 1) # Rank

        # Gini coefficient formula
        # (Sum of (2 * rank - n - 1) * value) / (n * Sum of values)
        return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))
