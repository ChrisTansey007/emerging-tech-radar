import logging
import pandas as pd # Added for pd.isna checks

# from ..config.settings import uncertainty_config # Example import

class UncertaintyManager:
    def __init__(self, uncertainty_cfg): # Added config
        self.config = uncertainty_cfg
        self.confidence_thresholds = self.config.get('confidence_thresholds', {'high':0.8, 'medium':0.6, 'low':0.4})

    def assess_data_completeness_impact(self, data_source_name, completeness_metric, sector_name):
        confidence_adjustment = 0.0; message = ""
        min_completeness = self.config.get('min_data_completeness_for_high_confidence', 0.8)
        if completeness_metric < min_completeness / 2:
            confidence_adjustment = -0.20
            message = f"Warning ({sector_name}): {data_source_name} data very incomplete ({completeness_metric*100:.0f}%). Confidence reduced."
        elif completeness_metric < min_completeness:
            confidence_adjustment = -0.10
            message = f"Caution ({sector_name}): {data_source_name} data incomplete ({completeness_metric*100:.0f}%). Confidence reduced."
        if message: logging.warning(message)
        return confidence_adjustment, message

    def handle_conflicting_signals(self, sector_name, signal_strengths_dict):
        message = ""; confidence_adjustment = 0.0; scenario_info = None
        positive_signals = [s for s in signal_strengths_dict.values() if s > 0.5]
        negative_signals = [s for s in signal_strengths_dict.values() if s < -0.5]
        if positive_signals and negative_signals:
            conflict_desc = (f"Conflicting signals for '{sector_name}'. Pos: "
                             f"{ {k for k,v in signal_strengths_dict.items() if v > 0.5} }, "
                             f"Neg: { {k for k,v in signal_strengths_dict.items() if v < -0.5} }.")
            message = f"{conflict_desc} Increased uncertainty. Consider scenario analysis."
            logging.warning(message)
            confidence_adjustment = -0.15
            scenario_info = {'status': 'conflicting', 'message': message, 'scenarios_suggested': True}
        else:
            message = f"Signals for '{sector_name}' generally consistent."
            scenario_info = {'status': 'consistent', 'message': message, 'scenarios_suggested': False}
        return confidence_adjustment, message, scenario_info

    def acknowledge_research_pipeline_gap(self, sector_name, paper_coverage_metric):
        message = ""; confidence_adjustment = 0.0
        min_coverage = self.config.get('min_paper_coverage_for_high_confidence', 0.6)
        if paper_coverage_metric < min_coverage / 2:
            message = f"Warning ({sector_name}): Research visibility gap - sparse papers ({paper_coverage_metric*100:.0f}%). Low research insight reliability."
            confidence_adjustment = -0.15
            logging.warning(message)
        elif paper_coverage_metric < min_coverage:
            message = f"Caution ({sector_name}): Moderate research visibility gap ({paper_coverage_metric*100:.0f}%). Caution with research insights."
            confidence_adjustment = -0.05
            logging.info(message)
        return confidence_adjustment, message

    def format_final_prediction_output(self, prediction_value, base_confidence_score, adjustments_list, confidence_interval_dict=None):
        final_confidence_score = max(0.0, min(1.0, base_confidence_score + sum(adjustments_list)))
        if final_confidence_score < self.confidence_thresholds['low']:
            label, disclaimer = "Low", "Low confidence. Interpret with extreme caution."
        elif final_confidence_score < self.confidence_thresholds['medium']:
            label, disclaimer = "Medium", "Medium confidence. Corroborate with other sources."
        else:
            label, disclaimer = "High", "High confidence. Subject to unforeseen events."

        interval_str = "N/A"
        if confidence_interval_dict:
            ci_key = '95%' if '95%' in confidence_interval_dict else ('68%' if '68%' in confidence_interval_dict else None)
            if ci_key:
                low, high = confidence_interval_dict[ci_key]
                if not (pd.isna(low) or pd.isna(high)): # check for NaN
                    interval_str = f"[{low:.3f}, {high:.3f}] ({ci_key} CI)"

        logging.info(f"Final Output - Pred: {prediction_value:.3f}, Conf: {label} ({final_confidence_score:.2f}), Interval: {interval_str}")
        return {'prediction_value': prediction_value, 'final_confidence_score': final_confidence_score,
                'confidence_label': label, 'confidence_interval_str': interval_str, 'disclaimer': disclaimer}

    def recommend_alternative_sources_or_methods(self, primary_source_name, sector_name, issue_description):
        message = (f"Issue with '{primary_source_name}' for '{sector_name}': {issue_description}. "
                   f"Consider: [Expert Interviews], [Market Reports], [Analogous Sector Analysis], [Scenario Planning].")
        logging.warning(message)
        return message
