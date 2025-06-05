import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# joblib was imported but not used directly in InnovationPredictor snippet.
# It's often used for saving/loading models, which can be added later if needed.
# import joblib

class InnovationPredictor:
    def __init__(self, random_state=42):
        self.models_blueprints = {
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=random_state)
        }
        self.ensemble_weights = {'random_forest': 0.6, 'gradient_boosting': 0.4}
        self.sector_models = {}
        self.feature_importances = {}
        self.scaler = StandardScaler() # For _normalize_features

    def _ensure_datetime_index(self, df, df_name):
        if df is None or df.empty:
            return pd.DataFrame() # Return empty DataFrame if input is None or empty

        if not isinstance(df.index, pd.DatetimeIndex):
            # Attempt to find a common date column
            date_col_candidates = ['date', 'filing_date', 'published_date', 'timestamp']
            date_col = None
            for col in date_col_candidates:
                if col in df.columns:
                    date_col = col
                    break

            if date_col:
                try:
                    df.index = pd.to_datetime(df[date_col])
                    df = df.drop(columns=[date_col])
                except Exception as e:
                    print(f"Warning: Could not convert column '{date_col}' to DatetimeIndex for {df_name}: {e}")
                    # Return an empty DataFrame with a DatetimeIndex if conversion fails, to prevent downstream errors
                    return pd.DataFrame(index=pd.to_datetime([]))
            else:
                print(f"Warning: No DatetimeIndex or common date column found for {df_name}. Returning empty DataFrame.")
                return pd.DataFrame(index=pd.to_datetime([]))
        return df

    def _temporal_alignment(self, patent_features_df, funding_features_df, research_features_df, targets_df):
        patent_features_df = self._ensure_datetime_index(patent_features_df, "patent_features_df")
        funding_features_df = self._ensure_datetime_index(funding_features_df, "funding_features_df")
        research_features_df = self._ensure_datetime_index(research_features_df, "research_features_df")
        targets_df = self._ensure_datetime_index(targets_df, "targets_df")

        all_dfs = []
        if not patent_features_df.empty:
            all_dfs.append(patent_features_df.add_suffix('_patent'))
        if not funding_features_df.empty:
            all_dfs.append(funding_features_df.add_suffix('_funding'))
        if not research_features_df.empty:
            all_dfs.append(research_features_df.add_suffix('_research'))

        if not all_dfs: # If all feature dfs are empty
            if not targets_df.empty:
                 # Ensure targets_df has a valid DatetimeIndex for this path too
                if not isinstance(targets_df.index, pd.DatetimeIndex) or targets_df.index.empty:
                     print("Warning: targets_df is not empty but has no valid DatetimeIndex. Returning empty X, y.")
                     return pd.DataFrame(index=pd.to_datetime([])), pd.Series(dtype='float64')

                # Assuming target is the first column if not specified.
                # A specific target column name should ideally be passed or configured.
                target_col_name = targets_df.columns[0] if not targets_df.columns.empty else 'target'
                return pd.DataFrame(index=targets_df.index), targets_df[target_col_name] if target_col_name in targets_df else pd.Series(dtype='float64')
            return pd.DataFrame(index=pd.to_datetime([])), pd.Series(dtype='float64') # Return empty X and y with DatetimeIndex for X

        # Merge feature dataframes
        merged_features = all_dfs[0]
        for df in all_dfs[1:]:
            # Ensure indices are DatetimeIndex before merging
            if not isinstance(merged_features.index, pd.DatetimeIndex):
                print(f"Warning: merged_features lost DatetimeIndex before merging with {df.columns}. This is unexpected.")
                # Attempt to recover or handle, though this indicates a deeper issue
                # For now, return empty to prevent further errors
                return pd.DataFrame(index=pd.to_datetime([])), pd.Series(dtype='float64')
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Warning: DataFrame for {df.columns} lost DatetimeIndex. This is unexpected.")
                # Skip merging this df or handle as error
                continue

            merged_features = pd.merge(merged_features, df, left_index=True, right_index=True, how='outer')

        # Merge with targets
        if not targets_df.empty:
            if not isinstance(targets_df.index, pd.DatetimeIndex):
                 print("Warning: targets_df lost DatetimeIndex before merging with features. This is unexpected.")
                 # If targets_df has no valid index, cannot meaningfully merge.
                 # Consider how to handle this; for now, may proceed with features only or error.
                 # Let's assume an inner merge would result in empty if indices mismatch.
                 final_df = merged_features # Effectively, no target data can be aligned.
            else:
                # Assume target column is 'target_growth_6m' or the first column if not named
                target_col_name = 'target_growth_6m' if 'target_growth_6m' in targets_df.columns else targets_df.columns[0] if not targets_df.columns.empty else None
                if target_col_name:
                    final_df = pd.merge(merged_features, targets_df[[target_col_name]], left_index=True, right_index=True, how='inner')
                else:
                    print("Warning: No target column identified in targets_df. Proceeding without target alignment.")
                    final_df = merged_features # No target column to merge.
        else: # No targets, just return features (e.g. for prediction on new data)
            final_df = merged_features

        final_df = final_df.fillna(method='ffill').fillna(method='bfill')
        return final_df

    def _create_lagged_features(self, df, feature_columns, lags=[1, 3, 6, 12]):
        # Ensure df has a DatetimeIndex before trying to shift, otherwise shift is meaningless
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Warning: Dataframe does not have a DatetimeIndex in _create_lagged_features. Lagged features may be incorrect.")
            # Potentially convert or raise error, for now, proceed cautiously.
        if df.empty: return pd.DataFrame()
        lagged_df = df.copy()
        for col in feature_columns:
            if col in lagged_df.columns:
                for lag in lags:
                    lagged_df[f'{col}_lag{lag}'] = lagged_df[col].shift(lag)
        return lagged_df

    def _calculate_innovation_indices(self, patent_features_norm, funding_features_norm, research_features_norm, config):
        # This method needs feature_config, which should be passed or accessed from a config module
        patent_score = 0
        if not patent_features_norm.empty:
            weighted_sum = sum(patent_features_norm.get(feat, 0) * weight for feat, weight in config['patent_weights'].items() if feat in patent_features_norm)
            total_weight = sum(weight for feat, weight in config['patent_weights'].items() if feat in patent_features_norm)
            patent_score = weighted_sum / total_weight if total_weight > 0 else 0

        funding_score = 0
        if not funding_features_norm.empty:
            weighted_sum = sum(funding_features_norm.get(feat, 0) * weight for feat, weight in config['funding_weights'].items() if feat in funding_features_norm)
            total_weight = sum(weight for feat, weight in config['funding_weights'].items() if feat in funding_features_norm)
            funding_score = weighted_sum / total_weight if total_weight > 0 else 0

        research_score = 0
        if not research_features_norm.empty:
            weighted_sum = sum(research_features_norm.get(feat, 0) * weight for feat, weight in config['research_weights'].items() if feat in research_features_norm)
            total_weight = sum(weight for feat, weight in config['research_weights'].items() if feat in research_features_norm)
            research_score = weighted_sum / total_weight if total_weight > 0 else 0

        innovation_index = (0.40 * patent_score + 0.35 * funding_score + 0.25 * research_score)
        commercial_readiness = (0.60 * funding_score + 0.40 * patent_score)
        research_momentum = (0.70 * research_score + 0.30 * patent_score)

        return pd.DataFrame({
            'innovation_index': [innovation_index],
            'commercial_readiness_index': [commercial_readiness],
            'research_momentum_index': [research_momentum]
        })

    def _normalize_features(self, features_df, method='z_score'):
        if features_df.empty: return features_df
        if method == 'z_score':
            # Ensure scaler is fitted only on training data.
            # For simplicity here, it might be fit_transformed.
            # Proper way: fit scaler on train, transform on train/test.
            return pd.DataFrame(self.scaler.fit_transform(features_df), columns=features_df.columns, index=features_df.index)
        elif method == 'min_max':
            min_max_scaler = MinMaxScaler()
            return pd.DataFrame(min_max_scaler.fit_transform(features_df), columns=features_df.columns, index=features_df.index)
        return features_df

    def prepare_training_data(self, patent_df, funding_df, research_df, targets_df, feature_engineering_config=None): # Simplified input
        # Assume target column in targets_df is named 'target_growth_6m'
        # This name should be configured or passed if it varies.
        target_column_name = 'target_growth_6m'

        aligned_df = self._temporal_alignment(patent_df, funding_df, research_df, targets_df)

        if aligned_df.empty:
            print("Warning: Alignment resulted in an empty DataFrame. Returning empty X, y.")
            return pd.DataFrame(), pd.Series(dtype='float64')

        if target_column_name not in aligned_df.columns:
            print(f"Warning: Target column '{target_column_name}' not found after alignment. Check target DataFrame and its main column name. Returning empty X, y.")
            # If target is not present, we can't form y.
            # Depending on use case, might return X only, but for training, y is needed.
            return pd.DataFrame(index=aligned_df.index), pd.Series(dtype='float64')

        y = aligned_df[target_column_name]
        # Drop target column and any other non-feature columns (e.g. if original target_df had multiple columns)
        X = aligned_df.drop(columns=[col for col in targets_df.columns if col in aligned_df.columns] if targets_df is not None and not targets_df.empty else [])


        if X.empty:
            print("Warning: Feature set X is empty after alignment and dropping target. Returning empty X, y.")
            return pd.DataFrame(), pd.Series(dtype='float64')

        # Normalize features
        # The scaler should be fitted on the training set only in a real pipeline.
        # For this refactoring, we use fit_transform, assuming this method prepares data for a single training run.
        X_normalized = self._normalize_features(X)
        if not isinstance(X_normalized, pd.DataFrame): # Ensure _normalize_features returns DataFrame
             X_normalized = pd.DataFrame(X_normalized, index=X.index, columns=X.columns)


        # Create lagged features
        feature_columns_for_lags = X_normalized.columns.tolist()
        X_lagged = self._create_lagged_features(X_normalized, feature_columns_for_lags)

        # Handle NaNs after lagging
        if X_lagged.empty:
            print("Warning: X_lagged is empty. Returning empty X, y.")
            return pd.DataFrame(), pd.Series(dtype='float64')

        X_processed = X_lagged.fillna(X_lagged.median())

        # Align y with X_processed before dropping more NaNs from X_processed
        # This ensures that y corresponds to the rows kept in X_processed so far
        y_aligned_pre_dropna = y.reindex(X_processed.index)

        # Drop rows that might still have NaNs (e.g., if all values in a column were NaN for median)
        initial_rows = len(X_processed)
        X_processed = X_processed.dropna()

        # Final alignment of y with X_processed after all NaN removals from X
        y_aligned = y_aligned_pre_dropna.reindex(X_processed.index)

        if initial_rows > 0 and len(X_processed) < initial_rows:
            print(f"Dropped {initial_rows - len(X_processed)} rows from X due to remaining NaNs after lagging and median fill.")

        if X_processed.empty:
            print("Warning: X_processed is empty after NaN handling. Returning empty X, y.")
            return pd.DataFrame(), pd.Series(dtype='float64')

        return X_processed, y_aligned

    def _get_param_grid(self, model_name):
        if model_name == 'random_forest':
            return {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]} # Reduced for speed
        elif model_name == 'gradient_boosting':
            return {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5], 'subsample': [0.8, 1.0]} # Reduced
        return {}

    def train_sector_models(self, X_train_full, y_train_full, sectors_column='sector_label'):
        if X_train_full.empty or y_train_full.empty:
            print("Training data is empty. Skipping model training.")
            return {}
        tscv = TimeSeriesSplit(n_splits=3) # Reduced splits for speed
        unique_sectors = X_train_full[sectors_column].unique()
        for sector in unique_sectors:
            print(f"Training models for sector: {sector}...")
            sector_mask = X_train_full[sectors_column] == sector
            X_sector = X_train_full[sector_mask].drop(columns=[sectors_column])
            y_sector = y_train_full[sector_mask]
            X_sector_numeric = X_sector.select_dtypes(include=np.number).fillna(X_sector.select_dtypes(include=np.number).median())
            if len(X_sector_numeric) < 20: # Min samples
                print(f"Insufficient data for {sector}: {len(X_sector_numeric)} samples. Skipping.")
                continue
            self.sector_models[sector] = {}
            self.feature_importances[sector] = {}
            for model_name, model_blueprint in self.models_blueprints.items():
                model = model_blueprint
                param_grid = self._get_param_grid(model_name)
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
                try:
                    grid_search.fit(X_sector_numeric, y_sector)
                    best_model = grid_search.best_estimator_
                    self.sector_models[sector][model_name] = best_model
                    print(f"  {sector} - {model_name}: Best MAE = {-grid_search.best_score_:.4f}")
                    if hasattr(best_model, 'feature_importances_'):
                        importances = pd.Series(best_model.feature_importances_, index=X_sector_numeric.columns)
                        self.feature_importances[sector][model_name] = importances.sort_values(ascending=False)
                except Exception as e:
                    print(f"  Error training {model_name} for sector {sector}: {e}")
        return self.sector_models

    def validate_models(self, X_test_full, y_test_full, sectors_column='sector_label'):
        if X_test_full.empty or y_test_full.empty:
            print("Test data is empty. Skipping model validation.")
            return {}
        validation_results = {}
        unique_sectors = X_test_full[sectors_column].unique()
        for sector in unique_sectors:
            if sector not in self.sector_models or not self.sector_models[sector]:
                print(f"No trained models for sector '{sector}'. Skipping validation.")
                continue
            sector_mask = X_test_full[sectors_column] == sector
            X_sector_test = X_test_full[sector_mask].drop(columns=[sectors_column])
            y_sector_test = y_test_full[sector_mask]
            X_sector_test_numeric = X_sector_test.select_dtypes(include=np.number).fillna(X_sector_test.select_dtypes(include=np.number).median())
            if len(X_sector_test_numeric) == 0: continue
            sector_results = {}
            for model_name, model in self.sector_models[sector].items():
                try:
                    y_pred = model.predict(X_sector_test_numeric)
                    mae = mean_absolute_error(y_sector_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_sector_test, y_pred))
                    direction_accuracy = np.mean(np.sign(y_sector_test.fillna(0)) == np.sign(pd.Series(y_pred).fillna(0))) if len(y_sector_test) > 0 else 0
                    sector_results[model_name] = {'mae': mae, 'rmse': rmse, 'direction_accuracy': direction_accuracy, 'predictions_sample': y_pred[:3].tolist(), 'actuals_sample': y_sector_test[:3].tolist()}
                    print(f"  Validation {sector} - {model_name}: MAE={mae:.4f}, DirAcc={direction_accuracy:.2%}")
                except Exception as e:
                    print(f"  Error validating {model_name} for sector {sector}: {e}")
                    sector_results[model_name] = {'error': str(e)}
            validation_results[sector] = sector_results
        return validation_results
