# Cleared for new implementation
# This file will house the InnovationPredictor class for model training and validation.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib # Note: joblib was imported in model training but not directly used in snippet; keeping for model saving/loading.
from sklearn.preprocessing import StandardScaler # Added for _normalize_and_impute_for_training

# --- Model Training Process ---

class InnovationPredictor:
    def __init__(self, random_state=42):
        self.models_blueprints = {
            'random_forest': RandomForestRegressor(random_state=random_state),
            'gradient_boosting': GradientBoostingRegressor(random_state=random_state)
        }
        self.ensemble_weights = {'random_forest': 0.6, 'gradient_boosting': 0.4} # For prediction
        self.trained_sector_models = {} # {sector: {model_name: trained_model_object}}
        self.trained_scalers = {} # {sector: {feature_name: {'scaler': scaler_object, 'median': median_val}}}
        self.trained_feature_names = {} # {sector: [feature_list_order]}

    def _get_hyperparam_grid(self, model_name):
        if model_name == 'random_forest':
            return {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, None], 'min_samples_leaf': [1, 3, 5]}
        elif model_name == 'gradient_boosting':
            return {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5]}
        return {}

    def _normalize_and_impute_for_training(self, X_df, sector_name, fit_scalers=True):
        """ Normalize numeric features and impute NaNs. Fit scalers if training. """
        X_numeric = X_df.select_dtypes(include=np.number).copy()

        if fit_scalers:
            self.trained_scalers[sector_name] = {}

        for col in X_numeric.columns:
            # Impute first (e.g., with median of current data if fitting, or stored median if transforming)
            median_val = X_numeric[col].median()

            X_numeric[col] = X_numeric[col].fillna(median_val)

            if fit_scalers:
                scaler = StandardScaler() # Or MinMaxScaler, etc.
                X_numeric[col] = scaler.fit_transform(X_numeric[[col]])
                self.trained_scalers[sector_name][col] = {'scaler': scaler, 'median': median_val}
            elif sector_name in self.trained_scalers and col in self.trained_scalers[sector_name]:
                scaler = self.trained_scalers[sector_name][col]['scaler']
                X_numeric[col] = scaler.transform(X_numeric[[col]])
            # else: feature not seen in training, or no scaler for this sector, leave as is or error
            # This case should ideally be handled by ensuring consistency in features upstream
            # or by raising an error if a feature is expected to be scaled but no scaler exists.
        return X_numeric

    def train_all_sector_models(self, X_historical_full_df, y_historical_full_series, sector_id_column):
        """
        Train models for each sector present in the historical data.
        X_historical_full_df: DataFrame with all features and a sector_id_column.
        y_historical_full_series: Series with target values, indexed same as X.
        """
        if X_historical_full_df.empty or y_historical_full_series.empty:
            print("Training data is empty. Aborting.")
            return

        unique_sectors = X_historical_full_df[sector_id_column].unique()
        # Adjust n_splits based on data size, ensure at least 2 splits if possible
        min_samples_per_series = 10 # Min samples needed for each training set in a split
        n_total_samples = len(X_historical_full_df)
        n_unique_sectors = len(unique_sectors) if unique_sectors.any() else 1

        # Estimate max possible splits: (total_samples / num_sectors) / samples_per_split_set
        # This is a rough guide. TimeSeriesSplit needs n_samples > n_splits.
        estimated_max_splits = (n_total_samples // n_unique_sectors // min_samples_per_series) if n_unique_sectors > 0 else 0
        n_splits_for_tscv = min(5, max(2, estimated_max_splits)) # Ensure at least 2 splits, max 5

        # Ensure there are enough samples for even the minimum number of splits
        if n_total_samples <= n_splits_for_tscv * n_unique_sectors :
             print(f"Warning: Not enough data for reliable time series cross-validation with {n_splits_for_tscv} splits across {n_unique_sectors} sectors. Adjusting to fewer splits if possible or skipping CV for very small datasets.")
             # Fallback for very small data: minimum 2 splits if data allows, or could skip CV.
             # For simplicity here, we'll proceed but this is a critical check.
             if n_total_samples <= n_unique_sectors: # Not enough data for even one split per sector
                print("  Critically low data. Model training will be unreliable or fail.")
                # Potentially return or raise an error

        tscv = TimeSeriesSplit(n_splits=n_splits_for_tscv)

        for sector_val in unique_sectors:
            print(f"Processing sector: {sector_val}")
            sector_mask = X_historical_full_df[sector_id_column] == sector_val
            X_sector_raw = X_historical_full_df[sector_mask].drop(columns=[sector_id_column])
            y_sector = y_historical_full_series[sector_mask]

            if len(X_sector_raw) < n_splits_for_tscv + 1 : # Min samples for TimeSeriesSplit
                print(f"  Skipping {sector_val}: Insufficient data for {n_splits_for_tscv}-fold CV ({len(X_sector_raw)} samples). Needs at least {n_splits_for_tscv+1}.")
                continue
            if len(X_sector_raw) < 20: # Arbitrary small number warning
                 print(f"  Warning: Low data count for {sector_val} ({len(X_sector_raw)} samples). Model may not be robust.")


            X_sector_processed = self._normalize_and_impute_for_training(X_sector_raw, sector_val, fit_scalers=True)
            self.trained_feature_names[sector_val] = X_sector_processed.columns.tolist() # Save feature order

            self.trained_sector_models[sector_val] = {}
            for model_name, model_blueprint in self.models_blueprints.items():
                model_instance = model_blueprint # Fresh instance
                param_grid = self._get_hyperparam_grid(model_name)

                grid_search = GridSearchCV(estimator=model_instance, param_grid=param_grid, cv=tscv,
                                           scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
                try:
                    grid_search.fit(X_sector_processed, y_sector)
                    best_model = grid_search.best_estimator_
                    self.trained_sector_models[sector_val][model_name] = best_model
                    print(f"  Trained {model_name} for {sector_val}. Best CV MSE: {-grid_search.best_score_:.4f}")
                except Exception as e:
                    print(f"  Error training {model_name} for {sector_val}: {e}")

        print("Sector model training complete.")


    def validate_sector_models(self, X_test_full_df, y_test_full_series, sector_id_column):
        """Validate trained models on a hold-out test set."""
        results = {}
        if X_test_full_df.empty or y_test_full_series.empty:
            print("Test data is empty. Skipping validation.")
            return results

        for sector_val in X_test_full_df[sector_id_column].unique():
            if sector_val not in self.trained_sector_models or not self.trained_sector_models.get(sector_val):
                # print(f"No trained model for {sector_val} to validate.")
                continue

            if sector_val not in self.trained_feature_names or sector_val not in self.trained_scalers:
                print(f"Missing feature names or scalers for sector {sector_val}. Cannot validate.")
                continue

            sector_mask = X_test_full_df[sector_id_column] == sector_val
            X_sector_test_raw = X_test_full_df[sector_mask].drop(columns=[sector_id_column])
            y_sector_test = y_test_full_series[sector_mask]

            if X_sector_test_raw.empty:
                # print(f"No test data for sector {sector_val} after filtering.")
                continue

            # Preprocess test data using stored scalers and feature order
            X_sector_test_processed = self._normalize_and_impute_for_training(X_sector_test_raw, sector_val, fit_scalers=False)

            # Reorder columns to match training feature order and handle missing columns
            expected_features = self.trained_feature_names[sector_val]
            X_reordered = pd.DataFrame(columns=expected_features, index=X_sector_test_processed.index)
            for col in expected_features:
                if col in X_sector_test_processed.columns:
                    X_reordered[col] = X_sector_test_processed[col]
                elif col in self.trained_scalers.get(sector_val, {}): # Was in training, use its median
                     X_reordered[col] = self.trained_scalers[sector_val][col]['median']
                else: # Should ideally not happen if preprocessing is consistent
                    X_reordered[col] = 0 # Fallback, or raise error

            results[sector_val] = {}
            for model_name, trained_model in self.trained_sector_models[sector_val].items():
                try:
                    y_pred = trained_model.predict(X_reordered)
                    mae = mean_absolute_error(y_sector_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_sector_test, y_pred))
                    # Direction accuracy (assuming positive target means growth)
                    dir_acc = np.mean(np.sign(y_sector_test.fillna(0).values) == np.sign(pd.Series(y_pred).fillna(0).values))
                    results[sector_val][model_name] = {'MAE': mae, 'RMSE': rmse, 'DirectionAccuracy': dir_acc}
                    # print(f"  Validation - {sector_val} ({model_name}): MAE={mae:.3f}, RMSE={rmse:.3f}, DirAcc={dir_acc:.2%}")
                except Exception as e:
                    print(f"  Error validating {model_name} for {sector_val}: {e}")
                    results[sector_val][model_name] = {'error': str(e)}
        return results

```
