# Cleared for new implementation
# This file will house the InnovationPredictor class for model training and validation.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# import joblib  # Note: joblib was imported in model training but not directly used in snippet; keeping for model saving/loading.
from sklearn.preprocessing import (
    StandardScaler,
)  # Added for _normalize_and_impute_for_training

# --- Model Training Process ---

from typing import List, Dict, Any, Optional, Union # Added Union for type hints

class InnovationPredictor:
    """
    Manages the training, validation, and persistence of innovation prediction models
    for different technology sectors. It handles data normalization, hyperparameter
    tuning using time-series cross-validation, and stores trained models, scalers,
    and feature names for each sector.
    """
    # Type hints for class attributes
    models_blueprints: Dict[str, Any]
    ensemble_weights: Dict[str, float]
    trained_sector_models: Dict[str, Dict[str, Any]]
    trained_scalers: Dict[str, Dict[str, Dict[str, Union[StandardScaler, float]]]] # {'scaler': scaler_obj, 'median': median_val}
    trained_feature_names: Dict[str, List[str]]


    def __init__(self, random_state: int = 42):
        """
        Initializes the InnovationPredictor.

        Args:
            random_state: Seed for random number generators to ensure reproducibility.
        """
        self.models_blueprints = {
            "random_forest": RandomForestRegressor(random_state=random_state),
            "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        }
        self.ensemble_weights = {
            "random_forest": 0.6,
            "gradient_boosting": 0.4,
        }
        self.trained_sector_models = {}
        self.trained_scalers = {}
        self.trained_feature_names = {}

    def _get_hyperparam_grid(self, model_name: str) -> Dict[str, List[Any]]:
        """
        Returns a hyperparameter grid for GridSearchCV based on the model name.

        Args:
            model_name: The name of the model (e.g., 'random_forest', 'gradient_boosting').

        Returns:
            A dictionary defining the hyperparameter grid for the specified model.
            Returns an empty dictionary if the model_name is not recognized.
        """
        if model_name == "random_forest":
            return {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, None],
                "min_samples_leaf": [1, 3, 5],
            }
        elif model_name == "gradient_boosting":
            return {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5],
            }
        return {}

    def _normalize_and_impute_for_training(
        self, X_df: pd.DataFrame, sector_name: str, fit_scalers: bool = True
    ) -> pd.DataFrame:
        """
        Normalizes numeric features using StandardScaler and imputes NaNs with the median.
        If `fit_scalers` is True, it fits new scalers and stores them along with medians.
        Otherwise, it uses previously stored scalers and medians for transformation.

        Args:
            X_df: Input DataFrame with features for a specific sector.
            sector_name: The name of the sector being processed.
            fit_scalers: If True, scalers are fitted and stored. Otherwise, existing
                         scalers for the sector are applied.

        Returns:
            A DataFrame with processed numeric features (scaled and imputed).
        """
        X_numeric: pd.DataFrame = X_df.select_dtypes(include=np.number).copy()

        if fit_scalers:
            self.trained_scalers[sector_name] = {}

        for col in X_numeric.columns:
            median_val: float = X_numeric[col].median()
            X_numeric[col] = X_numeric[col].fillna(median_val)

            if fit_scalers:
                scaler = StandardScaler()
                X_numeric[col] = scaler.fit_transform(X_numeric[[col]])
                self.trained_scalers[sector_name][col] = {
                    "scaler": scaler,
                    "median": median_val,
                }
            elif sector_name in self.trained_scalers and col in self.trained_scalers[sector_name]:
                # Apply stored scaler and use stored median for consistency if any NaNs still exist (shouldn't if pre-filled)
                scaler = self.trained_scalers[sector_name][col]["scaler"]
                X_numeric[col] = scaler.transform(X_numeric[[col]])
            else:
                # This case implies a new numeric column appeared during transform-only mode,
                # or a feature expected to be scaled was not found in stored scalers.
                # For robustness, one might log this and apply a default (e.g. pass-through or impute only).
                print(f"Warning: Scaler not found for column '{col}' in sector '{sector_name}' during transform. Imputing with median only.")
        return X_numeric

    def train_all_sector_models(
        self, X_historical_full_df: pd.DataFrame, y_historical_full_series: pd.Series, sector_id_column: pd.Series
    ) -> None:
        """
        Trains models for each sector present in the historical data.
        It iterates through unique sectors identified by `sector_id_column`,
        preprocesses data for each sector, performs hyperparameter tuning using
        TimeSeriesSplit cross-validation, and stores the best trained model,
        scalers, and feature order for each sector.

        Args:
            X_historical_full_df: DataFrame containing all features (without the sector ID column itself).
                                  The index should align with `y_historical_full_series` and `sector_id_column`.
            y_historical_full_series: Series containing the target variable, indexed same as `X_historical_full_df`.
            sector_id_column: A pandas Series containing the sector identifier for each row in
                              `X_historical_full_df` and `y_historical_full_series`.
        """
        if X_historical_full_df.empty or y_historical_full_series.empty:
            print("Training data is empty. Aborting.")
            return

        unique_sectors: np.ndarray = sector_id_column.unique()

        # --- TimeSeriesSplit Configuration ---
        # Heuristic to determine a reasonable number of splits for TimeSeriesSplit.
        # Aims for at least 'min_samples_per_series' in each training part of a split.
        # This helps ensure that cross-validation is meaningful and robust.
        min_samples_per_series: int = 10

        # Approximate number of samples per sector to guide split calculation.
        # Assumes X_historical_full_df contains data for all sectors together.
        n_samples_approx_per_sector: float = len(X_historical_full_df) / (
            len(unique_sectors) if len(unique_sectors) > 0 else 1
        )

        # Max splits should not leave too few samples for any training/validation set.
        # TimeSeriesSplit requires n_samples (per sector) > n_splits.
        estimated_max_splits: int = int(n_samples_approx_per_sector / min_samples_per_series)

        # Ensure n_splits is at least 2 (required by CV) and at most a practical limit (e.g., 5).
        # If estimated_max_splits is less than 2, default to 2, but training for that sector might be skipped
        # if its specific sample count is too low.
        n_splits_for_tscv: int = min(5, max(2, estimated_max_splits if estimated_max_splits > 1 else 2) )

        # Overall check for total dataset size relative to splits. This is a general sanity check;
        # per-sector checks are more critical.
        if len(X_historical_full_df) <= n_splits_for_tscv :
             print(
                f"Warning: Total dataset size ({len(X_historical_full_df)}) is very small for {n_splits_for_tscv} time-series splits. Model training might be unreliable or fail. Adjusting to 2 splits if feasible for any sector."
            )
             n_splits_for_tscv = 2 if len(X_historical_full_df) > 2 else 1
             if n_splits_for_tscv == 1:
                 print("  Critically low data for CV. Proceeding without effective cross-validation logic.")
                 # Consider alternative validation or simpler models if this occurs.

        tscv = TimeSeriesSplit(n_splits=n_splits_for_tscv)

        for sector_val in unique_sectors:
            print(f"Processing sector: {sector_val}")
            # Create mask from the passed sector_id_column Series
            sector_mask: pd.Series = sector_id_column == sector_val
            # Apply mask to the DataFrame (which doesn't have the sector_id column anymore)
            X_sector_raw: pd.DataFrame = X_historical_full_df[sector_mask]
            y_sector: pd.Series = y_historical_full_series[sector_mask]

            if (
        # Heuristic to determine a reasonable number of splits for TimeSeriesSplit
        # Aims for at least 'min_samples_per_series' in each training part of a split.
        min_samples_per_series = (
            10  # Min samples desired for each training set in a split
        )
        # n_total_samples here refers to the count for the current sector if called per sector,
        # or total if called once. The current implementation iterates unique_sectors outside.
        # The logic below assumes X_historical_full_df is for ALL data.
        n_samples_approx_per_sector = len(X_historical_full_df) / (
            len(unique_sectors) if len(unique_sectors) > 0 else 1
        )

        # Max splits should not leave too few samples for any training/validation set.
        # TimeSeriesSplit requires n_samples > n_splits.
        # If we want each split's training set to have at least min_samples_per_series,
        # and there are (n_splits + 1) total sets of data created by n_splits,
        # then n_samples_approx_per_sector should be roughly > n_splits * min_samples_per_series / 2 (approx).
        # For simplicity: ensure at least 2 samples per split on average, plus one for the initial train set.
        estimated_max_splits = int(n_samples_approx_per_sector / min_samples_per_series)

        # Ensure n_splits is at least 2 (required by CV) and at most a heuristic like 5,
        # but also limited by available data.
        n_splits_for_tscv = min(5, max(2, estimated_max_splits if estimated_max_splits > 1 else 2) )

        if len(X_historical_full_df) <= n_splits_for_tscv : # Check against total samples if tscv applied to all
             print(
                f"Warning: Total dataset size ({len(X_historical_full_df)}) is too small for {n_splits_for_tscv} time-series splits. Model training might be unreliable or fail. Adjusting to 2 splits if possible."
            )
             n_splits_for_tscv = 2 if len(X_historical_full_df) > 2 else 1 # Fallback, though 1 split is not CV.
             if n_splits_for_tscv == 1:
                 print("  Critically low data for CV. Proceeding without cross-validation logic effectively.")
                 # A different flow might be needed here if CV is essential.

        tscv = TimeSeriesSplit(n_splits=n_splits_for_tscv)

        for sector_val in unique_sectors:
            print(f"Processing sector: {sector_val}")
            # Create mask from the passed sector_id_column Series
            sector_mask = sector_id_column == sector_val
            # Apply mask to the DataFrame (which doesn't have the sector_id column anymore)
            X_sector_raw = X_historical_full_df[sector_mask]
            y_sector = y_historical_full_series[sector_mask]

            if (
                len(X_sector_raw) < n_splits_for_tscv + 1
            ):  # Min samples for TimeSeriesSplit
                print(
                    f"  Skipping {sector_val}: Insufficient data for {n_splits_for_tscv}-fold CV ({len(X_sector_raw)} samples). Needs at least {n_splits_for_tscv+1}."
                )
                continue
            if len(X_sector_raw) < 20:  # Arbitrary small number warning
                print(
                    f"  Warning: Low data count for {sector_val} ({len(X_sector_raw)} samples). Model may not be robust."
                )

            X_sector_processed = self._normalize_and_impute_for_training(
                X_sector_raw, sector_val, fit_scalers=True
            )
            self.trained_feature_names[sector_val] = (
                X_sector_processed.columns.tolist()
            )  # Save feature order

            self.trained_sector_models[sector_val] = {}
            for model_name, model_blueprint in self.models_blueprints.items():
                model_instance = model_blueprint  # Fresh instance
                param_grid = self._get_hyperparam_grid(model_name)

                grid_search = GridSearchCV(
                    estimator=model_instance,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                    verbose=0,
                )
                try:
                    grid_search.fit(X_sector_processed, y_sector)
                    best_model = grid_search.best_estimator_
                    self.trained_sector_models[sector_val][model_name] = best_model
                    print(
                        f"  Trained {model_name} for {sector_val}. Best CV MSE: {-grid_search.best_score_:.4f}"
                    )
                except Exception as e:
                    print(f"  Error training {model_name} for {sector_val}: {e}")

        print("Sector model training complete.")

    def validate_sector_models(
        self, X_test_full_df, y_test_full_series, sector_id_column
    ):
        """Validate trained models on a hold-out test set."""
        results = {}
        if X_test_full_df.empty or y_test_full_series.empty:
            print("Test data is empty. Skipping validation.")
            return results

        for sector_val in X_test_full_df[sector_id_column].unique():
            if (
                sector_val not in self.trained_sector_models
                or not self.trained_sector_models.get(sector_val)
            ):
                # print(f"No trained model for {sector_val} to validate.")
                continue

            if (
                sector_val not in self.trained_feature_names
                or sector_val not in self.trained_scalers
            ):
                print(
                    f"Missing feature names or scalers for sector {sector_val}. Cannot validate."
                )
                continue

            sector_mask = X_test_full_df[sector_id_column] == sector_val
            X_sector_test_raw = X_test_full_df[sector_mask].drop(
                columns=[sector_id_column]
            )
            y_sector_test = y_test_full_series[sector_mask]

            if X_sector_test_raw.empty:
                # print(f"No test data for sector {sector_val} after filtering.")
                continue

            # Preprocess test data using stored scalers and feature order
            X_sector_test_processed: pd.DataFrame = self._normalize_and_impute_for_training(
                X_sector_test_raw, sector_val, fit_scalers=False
            )

            # Reorder columns to match training feature order and handle missing columns.
            # This is crucial for ensuring the model sees features in the exact same order
            # and with the same preprocessing (scaling, imputation) as during training.
            # Missing columns in the test set (that were present in training) are filled
            # with their respective medians from the training phase, if available.
            expected_features: List[str] = self.trained_feature_names[sector_val]
            X_reordered: pd.DataFrame = pd.DataFrame(
                columns=expected_features, index=X_sector_test_processed.index
            )
            for col in expected_features:
                if col in X_sector_test_processed.columns:
                    X_reordered[col] = X_sector_test_processed[col]
                elif col in self.trained_scalers.get(sector_val, {}): # Check if median is available from scaler info
                    X_reordered[col] = self.trained_scalers[sector_val][col]["median"]
                else:
                    # This implies a feature was in 'expected_features' but has no scaling/median info,
                    # which could occur if it was non-numeric and passed through during training.
                    # Or, it's a new column not seen in training (should be dropped by X_processed[expected_features]).
                    # For robustness, if it's truly expected and missing, fill with a default.
                    X_reordered[col] = 0
                    print(f"Warning: Feature '{col}' for sector '{sector_val}' in test set was filled with 0 due to no training median or scaling info, and was not present in input.")

            results[sector_val] = {}
            for model_name, trained_model in self.trained_sector_models[
                sector_val
            ].items():
                try:
                    y_pred = trained_model.predict(X_reordered)
                    mae = mean_absolute_error(y_sector_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_sector_test, y_pred))
                    # Direction accuracy (assuming positive target means growth)
                    dir_acc = np.mean(
                        np.sign(y_sector_test.fillna(0).values)
                        == np.sign(pd.Series(y_pred).fillna(0).values)
                    )
                    results[sector_val][model_name] = {
                        "MAE": mae,
                        "RMSE": rmse,
                        "DirectionAccuracy": dir_acc,
                    }
                    # print(f"  Validation - {sector_val} ({model_name}): MAE={mae:.3f}, RMSE={rmse:.3f}, DirAcc={dir_acc:.2%}")
                except Exception as e:
                    print(f"  Error validating {model_name} for {sector_val}: {e}")
                    results[sector_val][model_name] = {"error": str(e)}
        return results
