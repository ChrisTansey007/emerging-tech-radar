# Cleared for new implementation
# This file will house the PredictionGenerator class.

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from scipy import stats # Note: scipy.stats was imported in prediction generation but not directly used in snippet; keeping for statistical tests.

# --- Prediction Generation ---

class PredictionGenerator:
    def __init__(self, trained_models_info, ensemble_weights_dict, global_pred_config, random_seed=42):
        # trained_models_info is expected to be a dict like:
        # {
        #     'models': innovation_model_trainer.trained_sector_models,
        #     'scalers': innovation_model_trainer.trained_scalers,
        #     'feature_names': innovation_model_trainer.trained_feature_names
        # }
        self.trained_models = trained_models_info.get('models', {})
        self.trained_scalers = trained_models_info.get('scalers', {})
        self.trained_feature_names = trained_models_info.get('feature_names', {})
        self.ensemble_weights = ensemble_weights_dict
        self.prediction_config = global_pred_config # For emergence, investment criteria
        self.rng = np.random.default_rng(random_seed)

    def _preprocess_for_prediction(self, current_features_raw_df, sector_name):
        """Preprocess raw features for a sector using stored scalers and feature order."""
        if sector_name not in self.trained_scalers or sector_name not in self.trained_feature_names:
            raise ValueError(f"No scaler/feature info for sector {sector_name}. Model likely not trained or info not passed.")

        X_processed = current_features_raw_df.select_dtypes(include=np.number).copy()

        # Ensure consistent feature set as used in training
        expected_features = self.trained_feature_names[sector_name]

        # Add missing expected columns and fill them before scaling
        for col in expected_features:
            if col not in X_processed.columns:
                # If a feature expected by the model is missing in current input,
                # fill with its median from training time before scaling.
                if col in self.trained_scalers.get(sector_name, {}) and 'median' in self.trained_scalers[sector_name][col]:
                    X_processed[col] = self.trained_scalers[sector_name][col]['median']
                else:
                    # Fallback if median wasn't stored (should not happen with current predictor logic) or unknown feature
                    X_processed[col] = 0
                    print(f"Warning: Feature '{col}' for sector '{sector_name}' was not in input and no training median found. Filled with 0.")


        # Scale features that were scaled during training
        for col in X_processed.columns:
            if col in self.trained_scalers.get(sector_name, {}): # Check if this feature was scaled for this sector
                col_info = self.trained_scalers[sector_name][col]
                # Fill NaNs with training median *before* scaling this specific column
                X_processed[col] = X_processed[col].fillna(col_info['median'])
                try:
                    X_processed[[col]] = col_info['scaler'].transform(X_processed[[col]])
                except Exception as e:
                    print(f"Error scaling column {col} for sector {sector_name}: {e}. Data: {X_processed[[col]]}")
                    # Decide on fallback: keep unscaled, fill with median, or error out
                    # For now, attempt to continue, but this indicates an issue.
                    pass # Or X_processed[col] = col_info['median'] to prevent downstream errors from non-numeric types
            elif col in expected_features: # Expected by model but not scaled (e.g. binary flags passed as is)
                pass # No scaling needed for this feature
            # If a column is in X_processed but not in expected_features, it will be dropped by reordering.


        # Reorder to match training feature order and select only expected features
        # This also drops any columns in X_processed not in expected_features
        X_final_ordered = pd.DataFrame(columns=expected_features, index=X_processed.index)
        for col in expected_features:
            if col in X_processed.columns:
                 X_final_ordered[col] = X_processed[col]
            # This case should have been handled by the addition of missing columns earlier
            # else:
            #    print(f"Critical Error: Expected feature '{col}' for sector '{sector_name}' still missing after pre-fill.")
            #    X_final_ordered[col] = 0 # Fallback to prevent crash

        return X_final_ordered


    def _get_ensemble_prediction_for_sector(self, preprocessed_features_df, sector_name):
        """Make prediction using the ensemble of models for a single sector."""
        if sector_name not in self.trained_models or not self.trained_models[sector_name]:
            raise ValueError(f"No trained models for sector {sector_name}.")

        predictions_from_models = {}
        for model_name, model_obj in self.trained_models[sector_name].items():
            try:
                pred = model_obj.predict(preprocessed_features_df) # Expects DataFrame
                predictions_from_models[model_name] = pred[0] if len(pred) > 0 else np.nan # Assuming single row prediction
            except Exception as e:
                print(f"Error predicting with {model_name} for {sector_name}: {e}")
                predictions_from_models[model_name] = np.nan

        # Weighted average for ensemble
        valid_preds_sum, total_weight = 0, 0
        for name, p_val in predictions_from_models.items():
            if pd.notna(p_val) and name in self.ensemble_weights: # Check for np.nan/pd.NA
                weight = self.ensemble_weights[name]
                valid_preds_sum += p_val * weight
                total_weight += weight

        final_prediction = (valid_preds_sum / total_weight) if total_weight > 0 else np.nan
        return final_prediction, predictions_from_models

    def _estimate_confidence_intervals(self, individual_model_preds_dict, default_std_dev_factor=0.1):
        """Estimate CI based on variance of ensemble predictions or fallback.
           The default_std_dev_factor is relative to the prediction itself if only one model.
        """
        valid_preds = [p for p in individual_model_preds_dict.values() if pd.notna(p)]

        pred_std_dev = 0
        if len(valid_preds) >= 2:
            pred_std_dev = np.std(valid_preds)
        elif len(valid_preds) == 1: # Only one successful model prediction
            # Fallback: CI is a percentage of the single prediction value
            # This is a heuristic and assumes the prediction itself gives some scale.
            # A more robust approach would use historical RMSE of that single model if available.
            pred_std_dev = abs(valid_preds[0] * default_std_dev_factor)
        else: # No valid predictions or all failed
            pred_std_dev = default_std_dev_factor # Treat as an absolute std dev if no prediction available (e.g. 0.1 if values are ~0-1)


        # Assuming normal distribution for CI (z-scores for 68% and 95%)
        return {
            '68%_std_dev_factor': pred_std_dev * 1.0,
            '95%_std_dev_factor': pred_std_dev * 1.96
        }

    def _assess_prediction_quality_score(self, sector_name, current_features_df):
        """Simplified quality score based on feature completeness or recent model performance (conceptual)."""
        # Example: Check for too many missing values in input features *before* preprocessing
        # This is a placeholder for a more robust quality assessment.
        if current_features_df.empty: return 0.0
        completeness = 1.0 - (current_features_df.isnull().sum().sum() / float(current_features_df.size))
        return max(0.1, completeness) # Base quality, e.g. 0.1 to 1.0

    def generate_forecasts_for_sectors(self, current_features_per_sector_map, horizons_list):
        """
        Generate forecasts for multiple sectors and horizons.
        current_features_per_sector_map: {sector_name: current_raw_features_df_for_that_sector (1 row)}
        horizons_list: e.g. [6, 12] - Note: current models are not horizon-specific. This is conceptual.
        """
        all_sector_forecasts = {}
        for sector_name, raw_features_df in current_features_per_sector_map.items():
            if raw_features_df.empty:
                print(f"Skipping forecast for {sector_name}: Raw features DataFrame is empty.")
                continue
            try:
                preprocessed_features = self._preprocess_for_prediction(raw_features_df, sector_name)
                final_pred_value, individual_preds = self._get_ensemble_prediction_for_sector(preprocessed_features, sector_name)

                if pd.isna(final_pred_value): # Check for np.nan or pd.NA
                    print(f"Skipping forecast for {sector_name}: Final prediction is NaN.")
                    continue

                ci_factors = self._estimate_confidence_intervals(individual_preds)
                quality = self._assess_prediction_quality_score(sector_name, raw_features_df) # Pass raw for completeness check

                sector_horizon_forecasts = {}
                for horizon_months in horizons_list:
                    # Adjust prediction/CI for horizon if model is horizon-agnostic (very simplified)
                    # A proper way is to train horizon-specific models.
                    # This is a placeholder: assume prediction scales somehow or is same for all short-term horizons.
                    h_adj_factor = 1.0 # No adjustment for now

                    pred_at_horizon = final_pred_value * h_adj_factor
                    ci_68_lower = pred_at_horizon - ci_factors['68%_std_dev_factor']
                    ci_68_upper = pred_at_horizon + ci_factors['68%_std_dev_factor']
                    ci_95_lower = pred_at_horizon - ci_factors['95%_std_dev_factor']
                    ci_95_upper = pred_at_horizon + ci_factors['95%_std_dev_factor']

                    sector_horizon_forecasts[f'{horizon_months}m'] = {
                        'prediction': pred_at_horizon,
                        'confidence_interval_68_abs': [ci_68_lower, ci_68_upper],
                        'confidence_interval_95_abs': [ci_95_lower, ci_95_upper],
                        'quality_score': quality,
                        'generated_at': datetime.now(timezone.utc).isoformat()
                    }
                all_sector_forecasts[sector_name] = sector_horizon_forecasts
            except ValueError as ve:
                print(f"Skipping forecast for {sector_name} due to ValueError: {ve}")
            except Exception as e:
                print(f"Error generating forecast for {sector_name}: {e}")
        return all_sector_forecasts

    # --- Emergence and Investment Opportunity methods would go here ---
    # These methods would use self.prediction_config
    # Example structure:
    def identify_emerging_technologies(self, current_state_features_df, sector_col='sector_id'):
        """
        Identifies emerging technologies based on weighted indicators from current features.
        current_state_features_df: DataFrame containing all necessary features for all sectors.
                                   Must include features specified in emergence_indicators_weights.
                                   And a sector_col to identify the sector.
        """
        if self.prediction_config is None or 'emergence_indicators_weights' not in self.prediction_config:
            print("Warning: Emergence indicators configuration missing.")
            return []

        indicators_config = self.prediction_config['emergence_indicators_weights']
        required_cols = list(indicators_config.keys()) + [sector_col]
        if not all(col in current_state_features_df.columns for col in required_cols):
            print(f"Error: Missing one or more required columns for emergence identification. Need: {required_cols}")
            return []

        # Assume features are already normalized if 'norm' is in their name in config
        # Or apply normalization here based on historical data (complex, omitted for brevity)

        emerging_techs = []
        for idx, row in current_state_features_df.iterrows():
            score = 0
            for feature_name, weight in indicators_config.items():
                score += row.get(feature_name, 0) * weight # Default to 0 if feature somehow missing

            # Thresholding can be percentile-based or absolute, defined in config
            # For simplicity, let's assume any positive score is "emerging"
            # A real system would use a more robust threshold (e.g., top 10 percentile of scores)
            emergence_threshold = self.prediction_config.get('emergence_score_threshold', 0.5) # Example threshold
            if score > emergence_threshold:
                emerging_techs.append({
                    'technology_area': row[sector_col], # Or a more specific tech field if available
                    'emergence_score': score,
                    'contributing_factors': {fname: row.get(fname,0) * w for fname, w in indicators_config.items()}
                })

        # Sort by score
        emerging_techs.sort(key=lambda x: x['emergence_score'], reverse=True)
        return emerging_techs

    def create_investment_opportunities(self, sector_forecasts, emerging_techs, market_data_map, sector_col='sector_id'):
        """
        Creates ranked investment opportunities.
        sector_forecasts: Output from generate_forecasts_for_sectors.
        emerging_techs: Output from identify_emerging_technologies.
        market_data_map: {sector_name: {'market_size_potential_usd': 123, 'risk_metric': 0.2}}
        """
        if self.prediction_config is None or 'investment_ranking_criteria_weights' not in self.prediction_config:
            print("Warning: Investment ranking criteria configuration missing.")
            return []

        criteria_weights = self.prediction_config['investment_ranking_criteria_weights']
        opportunities = []

        # Combine forecast data with emergence scores and market data
        for sector, forecast_data in sector_forecasts.items():
            # Use shortest horizon forecast for ranking, or average, or specific one
            # For demo, take the first available horizon's prediction (e.g., '6m')
            first_horizon_key = next(iter(forecast_data)) if forecast_data else None
            if not first_horizon_key: continue

            forecast = forecast_data[first_horizon_key]['prediction']
            quality = forecast_data[first_horizon_key]['quality_score']

            emergence_entry = next((et for et in emerging_techs if et['technology_area'] == sector), None)
            emergence_score_norm = emergence_entry['emergence_score'] if emergence_entry else 0 # Normalize this if not already

            market_info = market_data_map.get(sector, {})
            market_size_norm = market_info.get('market_size_potential_usd_norm', 0) # Assume normalized
            risk_inverse_norm = 1.0 - market_info.get('risk_metric_norm', 0.5) # Assume normalized risk, invert

            attractiveness_score = 0
            attractiveness_score += forecast * criteria_weights.get('forecasted_growth_rate', 0)
            attractiveness_score += quality * criteria_weights.get('prediction_quality_score', 0)
            attractiveness_score += emergence_score_norm * criteria_weights.get('emergence_score_norm', 0) # Add if in config
            attractiveness_score += market_size_norm * criteria_weights.get('market_size_potential_usd_norm', 0)
            attractiveness_score += risk_inverse_norm * criteria_weights.get('risk_metric_inverse_norm', 0)

            opportunities.append({
                'sector': sector,
                'attractiveness_score': attractiveness_score,
                'forecasted_growth': forecast,
                'prediction_quality': quality,
                'emergence_metric': emergence_score_norm,
                'market_potential_metric': market_size_norm,
                'risk_metric_inversed': risk_inverse_norm,
                'recommended_action': "Review for Investment" if attractiveness_score > self.prediction_config.get('investment_min_score_threshold', 0.65) else "Monitor"
            })

        opportunities.sort(key=lambda x: x['attractiveness_score'], reverse=True)
        return opportunities

```
