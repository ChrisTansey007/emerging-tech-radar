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
        # Access configurations from self.prediction_config
        indicators_config = self.prediction_config.get('emergence_indicators', {})
        analysis_thresholds = self.prediction_config.get('emergence_analysis_thresholds', {}) # Expected to be added to config
        risk_map = self.prediction_config.get('emergence_risk_factors_map', {}) # Expected to be added to config

        # 1. Determine Key Indicators
        weighted_contributions = {}
        # Ensure area_features_series is a Series
        if isinstance(area_features_series, pd.DataFrame):
            current_features = area_features_series.iloc[0] if not area_features_series.empty else pd.Series()
        else:
            current_features = area_features_series

        for feature_name, weight in indicators_config.items():
            value = current_features.get(feature_name, 0)
            # Only consider features present in the series and with positive values for this simplified contribution
            if value > 0 and feature_name in current_features:
                weighted_contributions[feature_name] = value * weight

        sorted_contributors = sorted(weighted_contributions.items(), key=lambda item: item[1], reverse=True)

        key_indicators_output = []
        for name, contrib in sorted_contributors[:3]: # Top 3
            if contrib > 0: # Ensure contribution is positive
                 # Try to get raw value for context, default to 0 if not found
                raw_value = current_features.get(name, 0)
                key_indicators_output.append(f"{name.replace('_', ' ').title()} (Value: {raw_value:.2f}, Contribution: {contrib:.2f})")

        if not key_indicators_output and emergence_score > 0: # If score is positive but no specific indicators from loop
            key_indicators_output = [f"Overall positive momentum (Score: {emergence_score:.2f})"]
        elif not key_indicators_output:
            key_indicators_output = ["No specific strong positive indicators found."]

        # 2. Estimate Timeline
        fast_thresh = analysis_thresholds.get('timeline_fast_threshold', 0.7)
        medium_thresh = analysis_thresholds.get('timeline_medium_threshold', 0.4)
        if emergence_score >= fast_thresh:
            timeline_str = "0-1 year"
        elif emergence_score >= medium_thresh:
            timeline_str = "1-2 years"
        else:
            timeline_str = "2-3+ years"

        # 3. Set Confidence Level
        high_conf_thresh = analysis_thresholds.get('score_high_confidence', 0.7)
        medium_conf_thresh = analysis_thresholds.get('score_medium_confidence', 0.4)
        if emergence_score >= high_conf_thresh:
            confidence_str = "High"
        elif emergence_score >= medium_conf_thresh:
            confidence_str = "Medium"
        else:
            confidence_str = "Low"

        # 4. Identify Risk Factors (Simplified)
        identified_risks_list = []
        # Arbitrary low thresholds for demonstration, assuming features are somewhat scaled or their typical ranges are known.
        # These thresholds should ideally come from config or be based on feature distributions.
        # For this subtask, using placeholder values.
        low_funding_threshold = current_features.filter(like='_funding').mean() * 0.25 if any(col for col in current_features.index if '_funding' in col) else 0.1
        low_research_threshold = current_features.filter(like='_research').mean() * 0.25 if any(col for col in current_features.index if '_research' in col) else 0.1
        low_patent_threshold = current_features.filter(like='_patent').mean() * 0.25 if any(col for col in current_features.index if '_patent' in col) else 0.1

        # Using specific feature names if they are consistently available from _calculate_emergence_score
        # These names are used in _calculate_emergence_score, so they should be in area_features_series
        if current_features.get('funding_deals_velocity_3m_funding', 0) < low_funding_threshold :
            identified_risks_list.append(risk_map.get('low_funding_signal', "Low funding velocity indicates commercialization lag."))
        if current_features.get('publication_rate_3m_research', 0) < low_research_threshold:
            identified_risks_list.append(risk_map.get('low_research_signal', "Low publication rate suggests weakening research base."))
        if current_features.get('filing_rate_3m_patent', 0) < low_patent_threshold:
            identified_risks_list.append(risk_map.get('low_patent_signal', "Low patent filing rate points to slowing IP generation."))

        if not identified_risks_list and confidence_str != "High": # If no specific risks and confidence is not High
            identified_risks_list.append(risk_map.get('nascent_market', "General market adoption and scalability uncertainties for nascent tech."))

        if len(identified_risks_list) > 2: # Limit to top 2 distinct risks
            identified_risks_list = list(dict.fromkeys(identified_risks_list))[:2]


        return {
            'key_indicators': key_indicators_output,
            'timeline_estimate': timeline_str,
            'confidence_level': confidence_str,
            'risk_factors': identified_risks_list
        }

    def _calculate_investment_attractiveness(self, sector, forecast_details, market_data, ranking_criteria):
        score = 0
        growth_forecast = forecast_details.get('prediction', 0)
        quality_score = forecast_details.get('quality_score', 0) # Prediction quality/confidence
        market_size = market_data.get('market_size_usd', 0)

        # Score calculation (existing logic maintained for overall score)
        score += growth_forecast * ranking_criteria.get('growth_potential', 0)
        score += quality_score * ranking_criteria.get('confidence_score', 0)
        score += (market_size / 1e9) * ranking_criteria.get('market_size', 0) # Normalize market size for scoring
        score += ranking_criteria.get('risk_adjusted_return', 0) # This is a placeholder, true risk adjustment is complex

        # Enhanced Drivers Logic
        drivers = []
        high_growth_thresh = ranking_criteria.get('high_growth_threshold', 0.10)
        large_market_thresh_usd = ranking_criteria.get('large_market_threshold_usd', 5e8)

        if growth_forecast >= high_growth_thresh:
            drivers.append(f"Strong Growth Potential ({growth_forecast:.2%})")
        elif growth_forecast > 0: # Consider any positive growth a driver, could be refined
            drivers.append(f"Moderate Growth Potential ({growth_forecast:.2%})")
        else:
            drivers.append(f"Low/Negative Growth ({growth_forecast:.2%})")


        if quality_score >= self.prediction_config.get('confidence_thresholds', {}).get('high', 0.8):
            drivers.append("High Prediction Confidence")

        if market_size >= large_market_thresh_usd:
            drivers.append(f"Large Market Size (${market_size/1e6:.0f}M)")
        elif market_size > 0:
             drivers.append(f"Market Size (${market_size/1e6:.0f}M)")

        if not drivers: drivers.append("General market dynamics.")


        # Enhanced Risks Logic
        risks = []
        min_market_size_thresh_usd = ranking_criteria.get('min_market_size_threshold_usd', 1e8)
        medium_confidence_thresh = self.prediction_config.get('confidence_thresholds', {}).get('medium', 0.6)
        low_confidence_thresh = self.prediction_config.get('confidence_thresholds', {}).get('low', 0.4)
        low_growth_risk_thresh = ranking_criteria.get('low_growth_threshold', 0.01)

        if quality_score < low_confidence_thresh:
            risks.append("Very Low Prediction Confidence")
        elif quality_score < medium_confidence_thresh:
            risks.append("Medium/Low Prediction Confidence")

        if market_size < min_market_size_thresh_usd and market_size > 0:
            risks.append(f"Niche or Potentially Small Market (${market_size/1e6:.0f}M)")
        elif market_size == 0 and sector != "Emerging Technology": # For established sectors, 0 market size is a risk
            risks.append("Market Size Unknown/Unverified")

        if growth_forecast < low_growth_risk_thresh:
            risks.append(f"Low or Negative Growth Forecast ({growth_forecast:.2%})")

        if not risks: # If no specific risks identified
            risks.append("Standard market and execution risks apply.")

        # Action determination (existing logic maintained)
        action = "Monitor"
        # Thresholds for action can also be moved to config
        action_consider_investment_score_thresh = ranking_criteria.get('action_invest_score_thresh', 0.5)
        action_invest_growth_thresh = ranking_criteria.get('action_invest_growth_thresh', 0.05)
        action_invest_confidence_thresh = self.prediction_config.get('confidence_thresholds', {}).get('high', 0.7) # Use high from general config

        action_monitor_closely_score_thresh = ranking_criteria.get('action_monitor_score_thresh', 0.3)

        if score > action_consider_investment_score_thresh and \
           growth_forecast > action_invest_growth_thresh and \
           quality_score >= action_invest_confidence_thresh:
            action = "Consider Investment"
        elif score > action_monitor_closely_score_thresh and growth_forecast > 0:
            action = "Monitor Closely"

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
