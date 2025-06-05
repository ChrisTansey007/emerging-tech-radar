# Cleared for new implementation
# This file will house the UncertaintyManager class.

from datetime import datetime # Not strictly needed by class below but often useful in context
import logging

class UncertaintyManager:
    def __init__(self, global_prediction_config, logger_instance=None):
        self.pred_config = global_prediction_config # Contains 'confidence_thresholds_labels'
        self.conf_threshold_labels = self.pred_config.get('confidence_thresholds_labels',
                                                          {'high': 0.75, 'medium': 0.55, 'low': 0.0}) # Fallback
        if logger_instance:
            self.logger = logger_instance
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers:
                logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s - %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
                                    filename='uncertainty_manager.log', filemode='a')

    def get_overall_confidence_assessment(self,
                                          base_quality_score,
                                          data_completeness_map=None,
                                          research_coverage_map=None,
                                          conflicting_signals_info=None,
                                          custom_uncertainty_factors=None):
        """
        Aggregates various uncertainty factors into a final confidence score and label.
        base_quality_score: Initial quality/confidence from the prediction model (0 to 1).
        data_completeness_map: {source_name: completeness_0_to_1} e.g. {'patents': 0.9}
        research_coverage_map: {sector_name_or_general: coverage_0_to_1} e.g. {'AI_sector': 0.7}
        conflicting_signals_info: {'is_conflicting': True/False, 'severity_0_to_1': 0.5}
        custom_uncertainty_factors: list of dicts, e.g. [{'name': 'model_stability', 'impact_factor': -0.1, 'rationale': 'Recent volatility'}]
        """
        self.logger.debug(f"Starting confidence assessment. Base quality: {base_quality_score:.2f}")

        # Initialize maps/dicts if None
        data_completeness_map = data_completeness_map if data_completeness_map is not None else {}
        research_coverage_map = research_coverage_map if research_coverage_map is not None else {}
        conflicting_signals_info = conflicting_signals_info if conflicting_signals_info is not None else {}
        custom_uncertainty_factors = custom_uncertainty_factors if custom_uncertainty_factors is not None else []

        confidence_score = base_quality_score
        issues = []

        # 1. Adjust based on data completeness
        min_completeness_target = self.pred_config.get('min_data_completeness_target', 0.7)
        completeness_penalty_factor = self.pred_config.get('data_completeness_penalty_factor', 0.2)
        for source, completeness in data_completeness_map.items():
            if completeness < min_completeness_target:
                penalty = (min_completeness_target - completeness) * completeness_penalty_factor
                confidence_score -= penalty
                issues.append(f"Low {source} data completeness ({completeness:.0%}), penalty: {penalty:.2f}")
                self.logger.debug(f"Applied penalty for {source} completeness: {penalty:.2f}")


        # 2. Adjust based on research coverage (if applicable to the sector/prediction)
        # This assumes research_coverage_map might have one relevant entry for the current prediction context.
        min_research_coverage_target = self.pred_config.get('min_research_coverage_target', 0.5)
        research_coverage_penalty_factor = self.pred_config.get('research_coverage_penalty_factor', 0.15)
        for scope, coverage in research_coverage_map.items():
             if coverage < min_research_coverage_target:
                penalty = (min_research_coverage_target - coverage) * research_coverage_penalty_factor
                confidence_score -= penalty
                issues.append(f"Low research paper coverage for {scope} ({coverage:.0%}), penalty: {penalty:.2f}")
                self.logger.debug(f"Applied penalty for {scope} research coverage: {penalty:.2f}")


        # 3. Adjust for conflicting signals
        if conflicting_signals_info.get('is_conflicting', False):
            conflict_severity = conflicting_signals_info.get('severity_0_to_1', 0)
            conflict_penalty_factor = self.pred_config.get('conflict_penalty_factor', 0.25)
            penalty = conflict_severity * conflict_penalty_factor
            confidence_score -= penalty
            issues.append(f"Conflicting internal signals (severity: {conflict_severity:.0%}), penalty: {penalty:.2f}")
            self.logger.debug(f"Applied penalty for conflicting signals: {penalty:.2f}")

        # 4. Adjust for custom uncertainty factors
        for factor in custom_uncertainty_factors:
            impact = factor.get('impact_factor', 0) # Negative for reducing confidence, positive if it somehow increases
            confidence_score += impact # Direct adjustment
            rationale = factor.get('rationale', factor.get('name', 'Custom factor'))
            issues.append(f"{rationale} (impact: {impact:.2f})")
            self.logger.debug(f"Applied custom factor '{rationale}' impact: {impact:.2f}")


        final_confidence_score = max(0.0, min(1.0, confidence_score)) # Clamp score between 0 and 1
        self.logger.info(f"Final calculated confidence score: {final_confidence_score:.2f}")

        # Determine label based on thresholds
        label = "Medium" # Default
        # Order of checks matters if thresholds overlap. Standard: High > Medium > Low
        if final_confidence_score >= self.conf_threshold_labels.get('high', 0.75):
            label = "High"
        elif final_confidence_score >= self.conf_threshold_labels.get('medium', 0.55):
            label = "Medium"
        else: # Covers scores below 'medium' threshold
            label = "Low"
        self.logger.info(f"Assigned confidence label: {label}")

        # Construct disclaimer message
        disclaimer = "Standard forecast disclaimer applies. "
        if issues:
            disclaimer += "Specific uncertainties considered: " + "; ".join(issues) + ". "

        if label == "Low":
            disclaimer += self.pred_config.get('low_confidence_disclaimer', "Interpret with extreme caution.")
        elif label == "Medium":
            disclaimer += self.pred_config.get('medium_confidence_disclaimer', "Consider alternative scenarios and monitor closely.")
        elif label == "High":
            disclaimer += self.pred_config.get('high_confidence_disclaimer', "Confidence is relatively high, but outcomes are not guaranteed.")

        self.logger.debug(f"Final disclaimer: {disclaimer}")
        return final_confidence_score, label, disclaimer

# Example of how prediction_config might be structured regarding uncertainty,
# to be placed in config.settings later.
# prediction_config_example_for_uncertainty = {
#     'confidence_thresholds_labels': {'high': 0.80, 'medium': 0.60, 'low': 0.0},
#     'min_data_completeness_target': 0.7,
#     'data_completeness_penalty_factor': 0.2,
#     'min_research_coverage_target': 0.5,
#     'research_coverage_penalty_factor': 0.15,
#     'conflict_penalty_factor': 0.25,
#     'low_confidence_disclaimer': "Significant uncertainties exist; interpret with extreme caution.",
#     'medium_confidence_disclaimer': "Some uncertainties identified; consider these in your assessment.",
#     'high_confidence_disclaimer': "While confidence is high, predictions are inherently uncertain."
# }

```
