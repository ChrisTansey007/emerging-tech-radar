# Cleared for new implementation
# This file will house the PredictionGenerator class.

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler # For type hinting scaler objects

# from scipy import (
#     stats,
# )  # Note: scipy.stats was imported in prediction generation but not directly used in snippet; keeping for statistical tests.

# --- Prediction Generation ---


class PredictionGenerator:
    """
    Uses trained models to generate forecasts, identify emerging technologies,
    and create investment opportunity assessments. It relies on model artifacts
    (trained models, scalers, feature names) provided by an InnovationPredictor instance
    and configurations for defining emergence and investment criteria.
    """

    # Type hints for class attributes
    trained_models: Dict[str, Dict[str, Any]] # {sector: {model_name: model_object}}
    trained_scalers: Dict[str, Dict[str, Dict[str, Union[StandardScaler, float]]]] # {sector: {feature: {'scaler': Scaler, 'median': float}}}
    trained_feature_names: Dict[str, List[str]] # {sector: [feature_list]}
    ensemble_weights: Dict[str, float]
    prediction_config: Dict[str, Any]
    rng: np.random.Generator

    def __init__(
        self,
        trained_models_info: Dict[str, Any],
        ensemble_weights_dict: Dict[str, float],
        global_pred_config: Dict[str, Any],
        random_seed: int = 42,
    ):
        """
        Initializes the PredictionGenerator.

        Args:
            trained_models_info: A dictionary containing 'models', 'scalers',
                                 and 'feature_names' from a trained InnovationPredictor.
                                 Example: {'models': {...}, 'scalers': {...}, 'feature_names': {...}}
            ensemble_weights_dict: Dictionary defining weights for ensemble model predictions
                                   (e.g., {'random_forest': 0.6, 'gradient_boosting': 0.4}).
            global_pred_config: Global configuration dictionary for prediction generation,
                                including emergence indicators and investment criteria.
            random_seed: Seed for the random number generator.
        """
        self.trained_models = trained_models_info.get("models", {})
        self.trained_scalers = trained_models_info.get("scalers", {})
        self.trained_feature_names = trained_models_info.get("feature_names", {})
        self.ensemble_weights = ensemble_weights_dict
        self.prediction_config = global_pred_config
        self.rng = np.random.default_rng(random_seed)

    def _preprocess_for_prediction(
        self, current_features_raw_df: pd.DataFrame, sector_name: str
    ) -> pd.DataFrame:
        """
        Preprocesses raw features for a given sector to match the state during model training.

        The process involves:
        1. Selecting only numeric features from the input DataFrame.
        2. Identifying the list of feature names that the model for this sector was trained on.
        3. For any expected feature missing from the current input:
            a. It's added to the DataFrame.
            b. It's filled with its median value observed during training (if available).
            c. If no training median was stored (e.g., for non-numeric or unscaled features),
               it defaults to 0 with a warning.
        4. For all features now present in the DataFrame:
            a. If the feature was scaled during training (i.e., a scaler is stored for it):
                i. Any remaining NaNs are filled with its training median.
                ii. The stored scaler is applied to transform the feature.
                iii. If scaling fails unexpectedly, the feature is imputed with its training median as a fallback.
            b. If the feature was expected by the model but not scaled during training (e.g., a binary flag):
                i. Any NaNs are filled with a default value (currently 0, with a warning).
        5. Finally, the columns are reordered to exactly match the feature order used during training,
           and any columns not part of the original training features are dropped.

        Args:
            current_features_raw_df: Raw input DataFrame for a single sector (typically one row).
            sector_name: The name of the sector for which to preprocess features.

        Returns:
            A DataFrame with processed features, ready for input to the prediction model.
            This DataFrame will have columns in the same order and with the same scaling
            as the data used for training the model for the specified sector.

        Raises:
            ValueError: If no scaler or feature name information is found for the given `sector_name`,
                        indicating that the model for this sector was likely not trained or its
                        artifacts were not correctly passed during initialization.
        """
        if sector_name not in self.trained_scalers or sector_name not in self.trained_feature_names:
            raise ValueError(
                f"No scaler/feature info for sector {sector_name}. Model likely not trained or info not passed."
            )

        X_processed: pd.DataFrame = current_features_raw_df.select_dtypes(include=np.number).copy()
        expected_features: List[str] = self.trained_feature_names[sector_name]
        sector_scalers_info: Dict[str, Dict[str, Any]] = self.trained_scalers.get(sector_name, {})

        # Add missing expected columns and fill them, preferably with training median.
        for col in expected_features:
            if col not in X_processed.columns:
                col_info: Dict[str, Any] = sector_scalers_info.get(col, {})
                median_val: Optional[float] = col_info.get("median")
                if median_val is not None:
                    X_processed[col] = median_val
                else:
                    X_processed[col] = 0  # Fallback for missing features without stored median
                    print(
                        f"Warning: Feature '{col}' for sector '{sector_name}' was missing from input and no training median found. Filled with 0."
                    )

        # Fill NaNs and scale features that were scaled during training.
        # Iterate over columns that are either expected or were scaled (some might be dropped later if not in expected_features)
        for col in X_processed.columns.tolist():
            if col in sector_scalers_info: # Check if this feature was scaled for this sector
                col_info_scaler: Dict[str, Any] = sector_scalers_info[col] # Renamed to avoid conflict
                X_processed[col] = X_processed[col].fillna(col_info_scaler["median"]) # Fill NaNs first
                try:
                    X_processed[[col]] = col_info_scaler["scaler"].transform(X_processed[[col]])
                except Exception as e:
                    print(
                        f"Error scaling column {col} for sector {sector_name}: {e}. Data: {X_processed[[col]]}. Imputing with median."
                    )
                    X_processed[col] = col_info_scaler["median"] # Fallback on scaling error
            elif col in expected_features:
                if X_processed[col].isnull().any():
                    print(f"Warning: Unscaled expected feature '{col}' in sector '{sector_name}' has NaNs. Filling with 0.")
                    X_processed[col] = X_processed[col].fillna(0)

        try:
            X_final_ordered: pd.DataFrame = X_processed[expected_features]
        except KeyError as e:
            missing_cols: List[str] = list(set(expected_features) - set(X_processed.columns))
            print(f"Critical Error: Could not reorder features for sector '{sector_name}'. Missing expected columns after processing: {missing_cols}. Error: {e}")
            X_final_ordered = pd.DataFrame(columns=expected_features, index=X_processed.index)
            for expected_col in expected_features:
                if expected_col in X_processed.columns:
                    X_final_ordered[expected_col] = X_processed[expected_col]
                else:
                    print(f"Critical Fallback: Filling completely missing expected column '{expected_col}' with 0 for sector '{sector_name}'.")
                    X_final_ordered[expected_col] = 0
        return X_final_ordered

    def _get_ensemble_prediction_for_sector(
        self, preprocessed_features_df: pd.DataFrame, sector_name: str
    ) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
        """
        Generates a prediction for a single sector using an ensemble of trained models.

        Args:
            preprocessed_features_df: DataFrame of features, preprocessed to match training conditions.
            sector_name: The name of the sector for which to predict.

        Returns:
            A tuple containing:
                - The final ensemble prediction value (float, or None if prediction fails).
                - A dictionary of individual model predictions ({model_name: prediction_value}).

        Raises:
            ValueError: If no trained models are found for the specified sector.
        """
        if not self.trained_models.get(sector_name): # More robust check
            raise ValueError(f"No trained models for sector {sector_name}.")

        predictions_from_models: Dict[str, Optional[float]] = {}
        for model_name, model_obj in self.trained_models[sector_name].items():
            try:
                pred: np.ndarray = model_obj.predict(preprocessed_features_df)
                predictions_from_models[model_name] = (
                    float(pred[0]) if len(pred) > 0 else np.nan
                )
            except Exception as e:
                print(f"Error predicting with {model_name} for {sector_name}: {e}")
                predictions_from_models[model_name] = np.nan

        valid_preds_sum: float = 0.0
        total_weight: float = 0.0
        for name, p_val in predictions_from_models.items():
            if pd.notna(p_val) and name in self.ensemble_weights:
                weight: float = self.ensemble_weights[name]
                # Ensure p_val is float for multiplication, though pd.notna should handle it
                valid_preds_sum += float(p_val) * weight
                total_weight += weight

        final_prediction: Optional[float] = (
            (valid_preds_sum / total_weight) if total_weight > 0 else np.nan
        )
        return final_prediction, predictions_from_models

    def _estimate_confidence_intervals(
        self, individual_model_preds_dict: Dict[str, Optional[float]], default_std_dev_factor: float = 0.1
    ) -> Dict[str, float]:
        """
        Estimate CI based on variance of ensemble predictions or a fallback heuristic.
        The default_std_dev_factor is relative to the prediction itself if only one
        model prediction is available, serving as a basic uncertainty measure.
        A more robust approach for single models would use their historical RMSE if available.
        """
        valid_preds: List[float] = [p for p in individual_model_preds_dict.values() if pd.notna(p) and p is not None]

        pred_std_dev: float
        if len(valid_preds) >= 2:
            pred_std_dev = np.std(valid_preds)
        elif len(valid_preds) == 1:
            pred_std_dev = abs(valid_preds[0] * default_std_dev_factor)
        else:
            pred_std_dev = default_std_dev_factor

        return {
            "68%_std_dev_factor": pred_std_dev * 1.0,
            "95%_std_dev_factor": pred_std_dev * 1.96
        }

    def _assess_prediction_quality_score(self, sector_name: str, current_features_df: pd.DataFrame) -> float:
        """
        Simplified quality score based on input feature completeness.
        A more advanced version could incorporate data drift measures or recent model performance.

        Args:
            sector_name: Name of the sector (for potential future use).
            current_features_df: The raw input DataFrame for which prediction is made.

        Returns:
            A float score between 0.1 and 1.0 representing data completeness.
        """
        if current_features_df.empty:
            return 0.0
        completeness: float = 1.0 - (
            current_features_df.isnull().sum().sum() / float(current_features_df.size)
        ) if current_features_df.size > 0 else 0.0 # Avoid division by zero
        return max(0.1, completeness)

    def generate_forecasts_for_sectors(
        self, current_features_per_sector_map: Dict[str, pd.DataFrame], horizons_list: List[int]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Generate forecasts for multiple sectors and specified future horizons.

        Args:
            current_features_per_sector_map: Dict {sector_name: raw_features_df_for_sector (1 row)}.
            horizons_list: List of integers, e.g., [6, 12], representing forecast horizons in months.
                           Note: The current underlying models are not inherently horizon-specific.
                           The 'h_adj_factor' is a placeholder for potential future horizon adjustments.
        Returns:
            A dictionary where keys are sector names and values are dictionaries
            of forecasts per horizon. Example:
            {
                'TechSectorA': {
                    '6m': {'prediction': 0.05, 'confidence_interval_68_abs': [0.02, 0.08], ...},
                    '12m': {'prediction': 0.05, ...}
                }, ...
            }
        """
        all_sector_forecasts: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if not current_features_per_sector_map:
            print("Warning: current_features_per_sector_map is empty. No forecasts to generate.")
            return all_sector_forecasts

        for sector_name, raw_features_df in current_features_per_sector_map.items():
            if raw_features_df.empty:
                print(f"Skipping forecast for {sector_name}: Raw features DataFrame is empty.")
                continue
            try:
                preprocessed_features: pd.DataFrame = self._preprocess_for_prediction(
                    raw_features_df, sector_name
                )
                final_pred_value: Optional[float]
                individual_preds: Dict[str, Optional[float]]
                final_pred_value, individual_preds = (
                    self._get_ensemble_prediction_for_sector(
                        preprocessed_features, sector_name
                    )
                )

                if pd.isna(final_pred_value): # Handles None or np.nan
                    print(f"Skipping forecast for {sector_name}: Final prediction is NaN after ensemble.")
                    continue

                ci_factors: Dict[str, float] = self._estimate_confidence_intervals(individual_preds)
                quality: float = self._assess_prediction_quality_score(
                    sector_name, raw_features_df
                )

                sector_horizon_forecasts: Dict[str, Dict[str, Any]] = {}
                for horizon_months in horizons_list:
                    h_adj_factor: float = 1.0
                    pred_at_horizon: float = final_pred_value * h_adj_factor # final_pred_value is now float

                    ci_68_lower: float = pred_at_horizon - ci_factors["68%_std_dev_factor"]
                    ci_68_upper: float = pred_at_horizon + ci_factors["68%_std_dev_factor"]
                    ci_95_lower: float = pred_at_horizon - ci_factors["95%_std_dev_factor"]
                    ci_95_upper: float = pred_at_horizon + ci_factors["95%_std_dev_factor"]

                    sector_horizon_forecasts[f"{horizon_months}m"] = {
                        "prediction": pred_at_horizon,
                        "confidence_interval_68_abs": [ci_68_lower, ci_68_upper],
                        "confidence_interval_95_abs": [ci_95_lower, ci_95_upper],
                        "quality_score": quality,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                all_sector_forecasts[sector_name] = sector_horizon_forecasts
            except ValueError as ve:
                print(f"Skipping forecast for {sector_name} due to ValueError: {ve}")
            except Exception as e:
                print(f"Error generating forecast for {sector_name}: {e}")
        return all_sector_forecasts

    def identify_emerging_technologies(
        self, current_state_features_df: pd.DataFrame, sector_col: str = "sector_id"
    ) -> List[Dict[str, Any]]:
        """
        Identifies emerging technologies based on weighted indicators from current features.

        Args:
            current_state_features_df: DataFrame containing features for all sectors.
                                       Must include features specified in
                                       `self.prediction_config.emergence_indicators_weights`.
                                       These features are typically derived (e.g., growth rates,
                                       accelerations) and should ideally be normalized before use.
            sector_col: Name of the column in `current_state_features_df` that
                        identifies the sector/technology area.

        Returns:
            A list of dictionaries, each representing an emerging technology area,
            including its score and contributing factors. Sorted by emergence score.
        """
        if self.prediction_config is None or not self.prediction_config.get("emergence_indicators_weights"):
            print("Warning: Emergence indicators configuration missing or empty.")
            return []

        indicators_config: Dict[str, float] = self.prediction_config.get("emergence_indicators_weights", {})
        required_cols: List[str] = list(indicators_config.keys()) + [sector_col]

        if not all(col in current_state_features_df.columns for col in required_cols):
            missing_cols: List[str] = [col for col in required_cols if col not in current_state_features_df.columns]
            print(f"Error: Missing one or more required columns for emergence identification: {missing_cols}. Need: {required_cols}")
            return []

        emerging_techs: List[Dict[str, Any]] = []
        for idx, row in current_state_features_df.iterrows():
            score: float = sum(
                row.get(feature_name, 0.0) * weight
                for feature_name, weight in indicators_config.items()
            )

            emergence_threshold: float = self.prediction_config.get("emergence_score_threshold", 0.5)
            if score > emergence_threshold:
                emerging_techs.append({
                    "technology_area": row.get(sector_col),
                    "emergence_score": score,
                    "contributing_factors": {
                        fname: row.get(fname, 0.0) * w
                        for fname, w in indicators_config.items()
                    },
                })

        emerging_techs.sort(key=lambda x: x.get("emergence_score", 0.0), reverse=True)
        return emerging_techs

    def create_investment_opportunities(
        self,
        sector_forecasts: Dict[str, Dict[str, Dict[str, Any]]],
        emerging_techs: List[Dict[str, Any]],
        market_data_map: Dict[str, Dict[str, float]],
        sector_col: str = "sector_id"
    ) -> List[Dict[str, Any]]:
        """
        Creates ranked investment opportunities by combining forecasts, emergence scores, and market data.

        Args:
            sector_forecasts: Output from `generate_forecasts_for_sectors`.
            emerging_techs: Output from `identify_emerging_technologies`.
            market_data_map: Dict {sector_name: {'market_size_potential_usd_norm': 0.8, 'risk_metric_norm': 0.3}}.
                             Values for market size and risk are assumed to be pre-calculated and normalized (0-1 range).
            sector_col: The key used for sectors/technology areas. Matches keys in `sector_forecasts` and `market_data_map`.

        Returns:
            A list of dictionaries, each representing a ranked investment opportunity.
        """
        if self.prediction_config is None or not self.prediction_config.get("investment_ranking_criteria_weights"):
            print("Warning: Investment ranking criteria configuration missing or empty.")
            return []

        criteria_weights: Dict[str, float] = self.prediction_config.get("investment_ranking_criteria_weights", {})
        opportunities: List[Dict[str, Any]] = []

        if not sector_forecasts and not emerging_techs:
            print("Warning: Both sector_forecasts and emerging_techs are empty. No opportunities to create.")
            return opportunities

        for sector, forecast_data_for_horizons in sector_forecasts.items():
            if not forecast_data_for_horizons:
                continue

            default_horizon_months: int = self.prediction_config.get('default_prediction_horizon_months', 12)
            default_horizon_key: str = f"{default_horizon_months}m"
            chosen_horizon_key: Optional[str] = default_horizon_key if default_horizon_key in forecast_data_for_horizons else next(iter(forecast_data_for_horizons), None)

            if not chosen_horizon_key or chosen_horizon_key not in forecast_data_for_horizons:
                print(f"Warning: No suitable forecast horizon found for sector '{sector}' for investment ranking.")
                continue

            forecast_details: Dict[str, Any] = forecast_data_for_horizons[chosen_horizon_key]
            forecast_value: float = forecast_details.get("prediction", 0.0)
            prediction_quality: float = forecast_details.get("quality_score", 0.0)

            emergence_entry: Optional[Dict[str, Any]] = next(
                (et for et in emerging_techs if et.get("technology_area") == sector), None
            )
            emergence_score_value: float = emergence_entry.get("emergence_score", 0.0) if emergence_entry else 0.0

            market_info: Dict[str, float] = market_data_map.get(sector, {})
            market_size_norm_value: float = market_info.get("market_size_potential_usd_norm", 0.0)
            risk_metric_value: float = market_info.get("risk_metric_norm", 0.5)
            risk_inverse_norm_value: float = 1.0 - risk_metric_value

            attractiveness_score: float = (
                forecast_value * criteria_weights.get("forecasted_growth_rate", 0.0) +
                prediction_quality * criteria_weights.get("prediction_quality_score", 0.0) +
                emergence_score_value * criteria_weights.get("emergence_score_norm", 0.0) +
                market_size_norm_value * criteria_weights.get("market_size_potential_usd_norm", 0.0) +
                risk_inverse_norm_value * criteria_weights.get("risk_metric_inverse_norm", 0.0)
            )

            opportunities.append(
                {
                    "sector": sector,
                    "attractiveness_score": attractiveness_score,
                    "forecasted_growth": forecast_value,
                    "prediction_quality": prediction_quality,
                    "emergence_metric": emergence_score_value,
                    "market_potential_metric": market_size_norm_value,
                    "risk_metric_inversed": risk_inverse_norm_value,
                    "recommended_action": "Review for Investment"
                        if attractiveness_score > self.prediction_config.get("investment_min_score_threshold", 0.65)
                        else "Monitor",
                }
            )

        # Note: This loop only considers sectors present in `sector_forecasts`.
        # Emerging tech areas not in `sector_forecasts` might need separate handling.
        opportunities.sort(key=lambda x: x.get("attractiveness_score", 0.0), reverse=True)
        return opportunities
>>>>>>> REPLACE
