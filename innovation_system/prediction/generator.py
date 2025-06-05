import numpy as np
import pandas as pd
from datetime import datetime
# from scipy import stats # Was imported but not used in snippet.

# This class will need prediction_config. It should be imported.
# from ..config.settings import prediction_config # Example if settings.py contains it

class PredictionGenerator:
    def __init__(self, trained_sector_models, ensemble_weights, prediction_cfg, random_state=42): # Added prediction_cfg
        self.sector_models = trained_sector_models
        self.ensemble_weights = ensemble_weights
        self.confidence_levels = [0.68, 0.95]
        self.rng = np.random.default_rng(random_state)
        self.prediction_config = prediction_cfg # Store it

    def _ensemble_predict_single_sector(self, sector_name, current_features_for_sector):
        if sector_name not in self.sector_models or not self.sector_models[sector_name]:
            raise ValueError(f"No trained model for sector: {sector_name}")
        individual_predictions = {}
        # Convert Series to DataFrame for predict, ensure columns match training
        features_df_for_predict = pd.DataFrame([current_features_for_sector])

        for model_name, model_obj in self.sector_models[sector_name].items():
            try:
                # Ensure feature order/names match training. This is tricky.
                # Assuming model_obj.feature_names_in_ is available (sklearn >= 0.24)
                if hasattr(model_obj, 'feature_names_in_'):
                    ordered_features = features_df_for_predict[model_obj.feature_names_in_]
                else: # Fallback, assumes current_features_for_sector is already ordered
                    ordered_features = features_df_for_predict
                pred = model_obj.predict(ordered_features)[0] # Predict expects 2D array
                individual_predictions[model_name] = pred
            except Exception as e:
                print(f"Error during prediction with {model_name} for {sector_name}: {e}")
                individual_predictions[model_name] = np.nan

        if not individual_predictions or all(np.isnan(p) for p in individual_predictions.values()):
            return np.nan, {}
        weighted_sum_preds, total_weight = 0, 0
        for model_name, pred_value in individual_predictions.items():
            if not np.isnan(pred_value) and model_name in self.ensemble_weights:
                weight = self.ensemble_weights[model_name]
                weighted_sum_preds += pred_value * weight
                total_weight += weight
        if total_weight == 0:
            valid_numeric_preds = [p for p in individual_predictions.values() if not np.isnan(p)]
            return np.mean(valid_numeric_preds) if valid_numeric_preds else np.nan, individual_predictions
        return weighted_sum_preds / total_weight, individual_predictions

    def _calculate_confidence_intervals_bootstrap(self, sector_name, current_features_for_sector, n_bootstrap=100):
        _, individual_predictions = self._ensemble_predict_single_sector(sector_name, current_features_for_sector)
        preds_list = [p for p in individual_predictions.values() if not np.isnan(p)]
        if len(preds_list) < 2: return {'68%': [np.nan, np.nan], '95%': [np.nan, np.nan]}
        intervals = {}
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            intervals[f'{int(conf_level*100)}%'] = [np.percentile(preds_list, 100 * alpha / 2), np.percentile(preds_list, 100 * (1 - alpha / 2))]
        return intervals

    def _assess_prediction_quality(self, sector, current_features, horizon):
        return 0.75 # Placeholder

    def generate_sector_forecasts(self, current_features_all_sectors_df, sector_column_name, horizons=[6, 12, 18]):
        forecasts = {}
        if sector_column_name not in current_features_all_sectors_df.columns:
            print(f"Error: Sector column '{sector_column_name}' not found.")
            return forecasts
        for sector_name in current_features_all_sectors_df[sector_column_name].unique():
            if sector_name not in self.sector_models:
                print(f"Skipping forecast for {sector_name}: No trained model.")
                continue
            sector_features_series = current_features_all_sectors_df[current_features_all_sectors_df[sector_column_name] == sector_name].drop(columns=[sector_column_name]).iloc[0]
            sector_forecasts_all_horizons = {}
            for horizon in horizons:
                try:
                    prediction_value, _ = self._ensemble_predict_single_sector(sector_name, sector_features_series)
                    if np.isnan(prediction_value): continue
                    confidence_intervals = self._calculate_confidence_intervals_bootstrap(sector_name, sector_features_series)
                    quality_score = self._assess_prediction_quality(sector_name, sector_features_series, horizon)
                    sector_forecasts_all_horizons[f'{horizon}m_growth_rate'] = {
                        'prediction': prediction_value, 'confidence_intervals': confidence_intervals,
                        'quality_score': quality_score, 'generated_at': datetime.now().isoformat()
                    }
                except ValueError as ve: print(f"Skipping {sector_name} for {horizon}m: {ve}")
                except Exception as e: print(f"Error generating {sector_name} {horizon}m forecast: {e}")
            if sector_forecasts_all_horizons: forecasts[sector_name] = sector_forecasts_all_horizons
        return forecasts

    def _calculate_emergence_score(self, area_features_df, indicators_config):
        score = 0
        # Ensure area_features_df is a Series or can be accessed like one
        area_features_series = area_features_df.iloc[0] if isinstance(area_features_df, pd.DataFrame) else area_features_df

        score += area_features_series.get('filing_rate_3m_patent', 0) * indicators_config.get('patent_filing_acceleration', 0)
        score += area_features_series.get('funding_deals_velocity_3m_funding', 0) * indicators_config.get('funding_velocity_increase', 0)
        score += area_features_series.get('publication_rate_3m_research', 0) * indicators_config.get('research_publication_growth', 0)
        return score

    def identify_emerging_technologies(self, all_features_df, sector_col, indicators_config, threshold_percentile=90):
        emerging_techs_output = []
        if all_features_df.empty or sector_col not in all_features_df.columns: return emerging_techs_output
        emergence_scores = []
        tech_areas = all_features_df[sector_col].unique()
        for tech_area in tech_areas:
            area_features_series = all_features_df[all_features_df[sector_col] == tech_area].iloc[0]
            score = self._calculate_emergence_score(area_features_series, indicators_config)
            emergence_scores.append({'technology': tech_area, 'emergence_score': score, 'features': area_features_series})

        if not emergence_scores: return emerging_techs_output
        scores_values = [s['emergence_score'] for s in emergence_scores if s['emergence_score'] > 0] # Consider only positive scores for percentile
        if not scores_values or len(scores_values) < (100 / (100 - threshold_percentile)) if threshold_percentile < 100 else 1 :
            print(f"Warning: Not enough positive scores ({len(scores_values)}) for reliable {threshold_percentile}th percentile for emergence.")
            threshold_value = max(scores_values) if scores_values else 0 # Take max if too few, or 0 if no positive scores
        else:
            threshold_value = np.percentile(scores_values, threshold_percentile)

        for item in emergence_scores:
            if item['emergence_score'] >= threshold_value and item['emergence_score'] > 0:
                analysis = self._analyze_emergence_signals(item['features'], item['emergence_score'])
                emerging_techs_output.append({
                    'technology': item['technology'], 'emergence_score': item['emergence_score'], 'is_above_threshold': True,
                    'key_indicators': analysis['key_indicators'], 'timeline_estimate': analysis['timeline_estimate'],
                    'confidence_level': analysis['confidence_level'], 'risk_factors': analysis['risk_factors']
                })
        emerging_techs_output.sort(key=lambda x: x['emergence_score'], reverse=True)
        return emerging_techs_output

    def _analyze_emergence_signals(self, area_features_series, emergence_score):
        return {'key_indicators': ["Rapid patent filing", "Funding surge"], 'timeline_estimate': "1-2 years", 'confidence_level': "Medium", 'risk_factors': ["Market adoption"]}

    def _calculate_investment_attractiveness(self, sector, forecast_details, market_data, ranking_criteria):
        score = 0; drivers = []; risks = []
        growth_forecast = forecast_details.get('prediction', 0)
        score += growth_forecast * ranking_criteria.get('growth_potential', 0)
        drivers.append(f"Expected Growth: {growth_forecast:.2%}")
        confidence = forecast_details.get('quality_score', 0)
        score += confidence * ranking_criteria.get('confidence_score', 0)
        if confidence < 0.6: risks.append("Low prediction confidence")
        current_market_size = market_data.get('market_size_usd', 0)
        score += (current_market_size / 1e9) * ranking_criteria.get('market_size', 0)
        drivers.append(f"Market Size: ${current_market_size/1e6:.0f}M")
        score += ranking_criteria.get('risk_adjusted_return', 0)
        action = "Monitor"
        if score > 0.5 and growth_forecast > 0.05 and confidence > 0.7: action = "Consider Investment"
        elif score > 0.3 and growth_forecast > 0: action = "Monitor Closely"
        return {'score': score, 'drivers': drivers, 'risks': risks, 'action': action}

    def create_investment_opportunities(self, sector_forecasts, emerging_techs_list, market_data_map, ranking_criteria):
        opportunities = []
        for sector, forecast_data_all_horizons in sector_forecasts.items():
            if not forecast_data_all_horizons: continue
            key = '12m_growth_rate' if '12m_growth_rate' in forecast_data_all_horizons else next(iter(forecast_data_all_horizons))
            if not key: continue
            forecast_details = forecast_data_all_horizons[key]
            # Use prediction_config from self.prediction_config
            if forecast_details['quality_score'] < self.prediction_config['confidence_thresholds']['low']: continue
            attractiveness = self._calculate_investment_attractiveness(sector, forecast_details, market_data_map.get(sector, {}), ranking_criteria)
            opportunities.append({
                'type': 'Established Sector', 'sector': sector, 'horizon_analyzed': key.split('_')[0],
                'expected_growth': forecast_details['prediction'], 'prediction_confidence_score': forecast_details['quality_score'],
                'attractiveness_score': attractiveness['score'], 'key_drivers': attractiveness['drivers'],
                'risk_assessment': attractiveness['risks'], 'recommended_action': attractiveness['action']
            })
        for emerging_item in emerging_techs_list:
            opportunities.append({
                'type': 'Emerging Technology', 'sector': emerging_item['technology'], 'horizon_analyzed': emerging_item['timeline_estimate'],
                'expected_growth': 'High Potential', 'prediction_confidence_score': emerging_item['confidence_level'],
                'attractiveness_score': emerging_item['emergence_score'], 'key_drivers': emerging_item['key_indicators'],
                'risk_assessment': emerging_item['risk_factors'], 'recommended_action': 'Monitor for Early Investment'
            })
        opportunities.sort(key=lambda x: x['attractiveness_score'], reverse=True)
        return opportunities
