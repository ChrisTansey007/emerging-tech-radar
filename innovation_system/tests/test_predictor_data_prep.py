```python
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from innovation_system.model_development.predictor import InnovationPredictor

class TestPredictorDataPreparation(unittest.TestCase):

    def setUp(self):
        self.predictor = InnovationPredictor(random_state=42)
        self.dates1 = pd.to_datetime([datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)])
        self.dates2 = pd.to_datetime([datetime(2023, 1, 3) + timedelta(days=i) for i in range(5)]) # Overlapping but offset
        self.dates3 = pd.to_datetime([datetime(2023, 2, 1) + timedelta(days=i) for i in range(3)]) # Non-overlapping

        self.patent_df_raw = pd.DataFrame({
            'date': self.dates1,
            'filings': np.random.rand(5),
            'citations': np.random.randint(1, 10, 5)
        })
        self.funding_df_raw = pd.DataFrame({
            'date': self.dates2,
            'deals': np.random.rand(5),
            'amount': np.random.randint(100, 1000, 5)
        })
        self.research_df_raw = pd.DataFrame({ # No common 'date' column initially
            'published_date': self.dates1,
            'papers': np.random.rand(5)
        })
        self.targets_df_raw = pd.DataFrame({
            'date': self.dates1, # Aligned with patent_df for simplicity in some tests
            'target_growth_6m': np.random.randn(5)
        })

    def test_ensure_datetime_index_with_column(self):
        df = self.predictor._ensure_datetime_index(self.patent_df_raw.copy(), 'patent_df_raw')
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertTrue('date' not in df.columns)

    def test_ensure_datetime_index_with_existing_index(self):
        df_indexed = self.patent_df_raw.set_index(pd.to_datetime(self.patent_df_raw['date']))
        df_processed = self.predictor._ensure_datetime_index(df_indexed.copy(), 'df_indexed')
        self.assertIsInstance(df_processed.index, pd.DatetimeIndex)
        self.assertEqual(len(df_indexed.columns), len(df_processed.columns)) # No column should be dropped

    def test_ensure_datetime_index_no_column_no_datetimeindex(self):
        df_no_date = pd.DataFrame({'data': [1,2,3]})
        # This should now return an empty DataFrame with DatetimeIndex as per implementation
        df_processed = self.predictor._ensure_datetime_index(df_no_date.copy(), 'df_no_date')
        self.assertTrue(df_processed.empty)
        self.assertIsInstance(df_processed.index, pd.DatetimeIndex)


    def test_temporal_alignment_empty_inputs(self):
        # _temporal_alignment now returns a tuple (df, series)
        aligned_output = self.predictor._temporal_alignment(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        if isinstance(aligned_output, tuple): # If it returns X, y tuple
            aligned_df, aligned_series = aligned_output
            self.assertTrue(aligned_df.empty)
            self.assertTrue(aligned_series.empty)
        else: # If it returns just a DataFrame (older version or different path)
             self.assertTrue(aligned_output.empty)


    def test_temporal_alignment_features_only(self):
        patent_df = self.predictor._ensure_datetime_index(self.patent_df_raw.copy(), 'patent_df_raw')
        funding_df = self.predictor._ensure_datetime_index(self.funding_df_raw.copy(), 'funding_df_raw')

        aligned_output = self.predictor._temporal_alignment(patent_df, funding_df, pd.DataFrame(), pd.DataFrame())

        aligned_df = aligned_output # Assuming it returns only df when targets_df is empty
        if isinstance(aligned_output, tuple): # If it returns X, y tuple even for features only
            aligned_df, _ = aligned_output

        self.assertFalse(aligned_df.empty)
        self.assertIn('filings_patent', aligned_df.columns)
        self.assertIn('amount_funding', aligned_df.columns)
        self.assertFalse(aligned_df.isnull().any().any(), "NaNs should be handled by ffill/bfill in features only merge")


    def test_temporal_alignment_with_targets(self):
        patent_df = self.predictor._ensure_datetime_index(self.patent_df_raw.copy(), 'patent_df_raw')
        funding_df = self.predictor._ensure_datetime_index(self.funding_df_raw.copy(), 'funding_df_raw')
        targets_df = self.predictor._ensure_datetime_index(self.targets_df_raw.copy(), 'targets_df_raw')

        # _temporal_alignment is expected to return a single DataFrame after merging features and target
        aligned_df = self.predictor._temporal_alignment(patent_df, funding_df, pd.DataFrame(), targets_df)

        self.assertFalse(aligned_df.empty)
        self.assertIn('filings_patent', aligned_df.columns)
        # The target column name is either 'target_growth_6m' or the first column name if 'target_growth_6m' is not present.
        # In this setup, it should be 'target_growth_6m'.
        self.assertIn(targets_df.columns[0] if 'target_growth_6m' not in targets_df.columns else 'target_growth_6m', aligned_df.columns)

        # Inner merge with targets means only dates present in targets_df (and other features) should remain
        # Get the actual target column name from the targets_df that was passed
        actual_target_col_name = 'target_growth_6m' if 'target_growth_6m' in targets_df.columns else targets_df.columns[0]

        # Ensure all indices in aligned_df are present in the original targets_df.index
        self.assertTrue(all(idx in targets_df.index for idx in aligned_df.index))
        # After inner merge and ffill/bfill, there should be no NaNs in the relevant columns
        # if the original data for those indices was not NaN.
        # Check NaNs only in columns that were part of the merge.
        check_cols = [col for col in ['filings_patent', 'amount_funding', actual_target_col_name] if col in aligned_df.columns]
        if check_cols: # only check if columns exist
             self.assertFalse(aligned_df[check_cols].isnull().any().any(), "NaNs should be handled by ffill/bfill")


    def test_temporal_alignment_fill_behavior(self):
        df1 = pd.DataFrame({'value_df1': [1, np.nan, 3]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
        df2 = pd.DataFrame({'value_df2': [np.nan, 5, np.nan]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
        targets = pd.DataFrame({'target_growth_6m': [0,0,0]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))

        aligned = self.predictor._temporal_alignment(df1, df2, pd.DataFrame(), targets)
        self.assertEqual(aligned['value_df1_patent'].tolist(), [1.0, 1.0, 3.0])
        self.assertEqual(aligned['value_df2_funding'].tolist(), [5.0, 5.0, 5.0])

    def test_prepare_training_data_simple_run(self):
        patent_df = self.predictor._ensure_datetime_index(self.patent_df_raw.copy(), 'patent_df_raw')
        funding_df = self.predictor._ensure_datetime_index(self.funding_df_raw.copy(), 'funding_df_raw')
        # For research_df_raw, 'published_date' is the intended date column.
        research_df = self.predictor._ensure_datetime_index(self.research_df_raw.copy(), 'research_df_raw')
        targets_df = self.predictor._ensure_datetime_index(self.targets_df_raw.copy(), 'targets_df_raw')

        X, y = self.predictor.prepare_training_data(patent_df, funding_df, research_df, targets_df)

        self.assertFalse(X.empty)
        self.assertFalse(y.empty)
        self.assertEqual(len(X), len(y))

        # Check that original feature names (before suffixing and lagging) are not directly in X's columns
        # (they should be suffixed or transformed into lags)
        original_feature_cols = list(patent_df.columns) + list(funding_df.columns) + list(research_df.columns)
        if 'date' in original_feature_cols: original_feature_cols.remove('date') # if date was still a column
        if 'published_date' in original_feature_cols: original_feature_cols.remove('published_date')

        for orig_col in original_feature_cols:
            self.assertFalse(orig_col in X.columns, f"Original column {orig_col} should not be in X directly")

        self.assertTrue(any(col.endswith(('_patent', '_funding', '_research')) or '_lag' in col for col in X.columns))
        self.assertFalse(X.isnull().any().any(), "X should have no NaNs after processing")

    def test_prepare_training_data_empty_inputs(self):
        X, y = self.predictor.prepare_training_data(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        self.assertTrue(X.empty, "X should be empty for all empty inputs")
        self.assertTrue(y.empty, "y should be empty for all empty inputs")

    def test_prepare_training_data_empty_features_with_targets(self):
        targets_df = self.predictor._ensure_datetime_index(self.targets_df_raw.copy(), 'targets_df_raw')
        X, y = self.predictor.prepare_training_data(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), targets_df)
        self.assertTrue(X.empty)
        self.assertTrue(y.empty)

    def test_prepare_training_data_normalization_and_lagging(self):
        dates = pd.to_datetime([datetime(2023,1,1) + timedelta(days=i) for i in range(20)])
        patent_data = pd.DataFrame({'date': dates, 'filings': np.arange(20, dtype=float)}) # Ensure float for potential scaler issues
        patent_df = self.predictor._ensure_datetime_index(patent_data, 'patent_data')

        # Create a target series that also has a DatetimeIndex
        targets_data_values = np.arange(20,40, dtype=float)
        targets_df = pd.DataFrame({'target_growth_6m': targets_data_values}, index=dates)
        # No need to call _ensure_datetime_index if index is already set and correct type

        X, y = self.predictor.prepare_training_data(patent_df, pd.DataFrame(), pd.DataFrame(), targets_df)

        self.assertFalse(X.empty)
        self.assertTrue(any('_lag1' in col for col in X.columns))
        self.assertTrue(any('_lag3' in col for col in X.columns))
        self.assertTrue(any('_lag6' in col for col in X.columns))
        self.assertTrue(any('_lag12' in col for col in X.columns))

        self.assertFalse(X.isnull().any().any())
        self.assertEqual(len(X), len(y))

        # Check that y values align with X's index
        pd.testing.assert_index_equal(X.index, y.index)

        # Since prepare_training_data does an inner merge on target, and then lags features,
        # the number of rows in X (and y) will be less than the original 20 due to NaNs from lagging
        # (especially the 12-period lag) being dropped.
        # The exact number of rows can be calculated if needed, but for now, just check consistency.
        self.assertTrue(len(X) < 20)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

```
