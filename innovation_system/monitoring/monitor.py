# Cleared for new implementation
# This file will house the SystemMonitor class.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
from sklearn.metrics import mean_absolute_error # For model performance evaluation

# Assuming model_config and PredictionGenerator would be imported or passed if needed by methods here.
# For now, model_config is accessed globally, will be fixed when configs are centralized.
# from ..config.settings import model_config # Example of future import
# from ..prediction.generator import PredictionGenerator # Example of future import for _preprocess_for_prediction

# --- Monitoring and Maintenance ---
# Setup basic logger for this module if it's run standalone,
# or rely on global logger if part of a larger app.
# logger = logging.getLogger(__name__) # Using __name__ is standard

class SystemMonitor:
    def __init__(self, trained_models_info_dict, global_model_config, db_conn_mock=None, logger_instance=None):
        # trained_models_info_dict expected to contain 'models', 'scalers', 'feature_names'
        self.trained_models_info = trained_models_info_dict
        self.model_config = global_model_config # For performance thresholds etc.
        self.db_connection = db_conn_mock # For fetching pipeline logs, historical performance
        self.data_drift_baselines = {} # {sector: {feature: {'mean': val, 'std': val}}}
        self.drift_threshold_pct = 0.20 # e.g. 20% change in mean from baseline
        self.perf_thresholds = self.model_config.get('performance_thresholds_monitoring',
                                                     {'mae_max': 0.20, 'direction_accuracy_min': 0.60}) # Fallback

        if logger_instance:
            self.logger = logger_instance
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers: # Avoid adding multiple handlers if already configured
                logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s - %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
                                    filename='system_monitor.log', filemode='a')


    def _get_db_pipeline_status(self, pipeline_name): # Mock
        # In a real system, this would query a database or monitoring service.
        self.logger.debug(f"Querying DB for status of pipeline: {pipeline_name}")
        if self.db_connection:
            # return self.db_connection.get_status(pipeline_name) # Example actual call
            pass # Replace with actual DB interaction logic

        # Mocked successful response:
        return {'last_success_run_utc': datetime.now(timezone.utc) - timedelta(hours=np.random.randint(1, 24)),
                'status': 'SUCCESS'}
        # Mocked failed response example:
        # return {'last_success_run_utc': datetime.now(timezone.utc) - timedelta(days=3), 'status': 'FAILED'}


    def check_all_data_pipelines_health(self, pipeline_configs_map):
        # pipeline_configs_map: {pipeline_name: {'max_staleness_days': X}}
        self.logger.info("Checking data pipeline health...")
        overall_healthy = True
        all_statuses = {}

        for name, config in pipeline_configs_map.items():
            status = self._get_db_pipeline_status(name)
            all_statuses[name] = status
            max_staleness = timedelta(days=config.get('max_staleness_days', 1)) # Default to 1 day if not specified

            if status.get('status') != 'SUCCESS' or                (datetime.now(timezone.utc) - status.get('last_success_run_utc', datetime.min.replace(tzinfo=timezone.utc))) > max_staleness:
                self.logger.error(f"Pipeline '{name}' issue: Status={status.get('status')}, LastSuccessRun={status.get('last_success_run_utc')}, MaxStaleness={max_staleness}.")
                overall_healthy = False
            else:
                self.logger.info(f"Pipeline '{name}' is healthy. Last run: {status.get('last_success_run_utc')}")

        if overall_healthy:
            self.logger.info("All data pipelines appear healthy.")
        else:
            self.logger.warning("One or more data pipelines have issues.")
        return overall_healthy, all_statuses


    def establish_drift_baselines(self, historical_X_full_df, sector_id_col):
        self.logger.info("Establishing data drift baselines...")
        self.data_drift_baselines = {} # Reset baselines

        if historical_X_full_df.empty or sector_id_col not in historical_X_full_df:
            self.logger.error("Cannot establish drift baselines: Historical data is empty or sector ID column missing.")
            return

        unique_sectors = historical_X_full_df[sector_id_col].unique()
        feature_names_map = self.trained_models_info.get('feature_names', {})

        for sector in unique_sectors:
            if sector not in feature_names_map:
                self.logger.debug(f"No trained feature names for sector {sector}, cannot establish drift baseline.")
                continue

            relevant_features_for_sector = feature_names_map[sector]
            X_sector = historical_X_full_df[historical_X_full_df[sector_id_col] == sector]

            # Select only the numeric features that were used in the model for this sector
            X_sector_relevant_numeric = X_sector[relevant_features_for_sector].select_dtypes(include=np.number)

            if X_sector_relevant_numeric.empty:
                self.logger.warning(f"No numeric data for baseline for sector {sector} with features: {relevant_features_for_sector}")
                continue

            # Calculate mean and std for each relevant numeric feature
            sector_baselines = X_sector_relevant_numeric.agg(['mean', 'std']).to_dict()
            # Filter out features where mean or std might be NaN (e.g. if all values were NaN for a feature)
            self.data_drift_baselines[sector] = {
                feat: stats for feat, stats in sector_baselines.items() if pd.notna(stats.get('mean')) and pd.notna(stats.get('std'))
            }
            self.logger.debug(f"Baselines for sector {sector}: {self.data_drift_baselines[sector]}")

        self.logger.info(f"Data drift baselines established for {len(self.data_drift_baselines)} sectors.")


    def monitor_for_data_drift(self, current_features_per_sector_map, temp_prediction_generator_instance):
        # temp_prediction_generator_instance: An instance of PredictionGenerator to use its _preprocess_for_prediction
        self.logger.info("Monitoring for data drift...")
        drift_alerts = {}

        if not self.data_drift_baselines:
            self.logger.warning("No data drift baselines established. Cannot monitor for drift.")
            return drift_alerts

        for sector, current_raw_features_df in current_features_per_sector_map.items():
            if sector not in self.data_drift_baselines:
                self.logger.debug(f"No baseline for sector {sector}. Skipping drift check.")
                continue
            if current_raw_features_df.empty:
                self.logger.debug(f"Empty current features for sector {sector}. Skipping drift check.")
                continue

            try:
                # Preprocess current data to align with model features (scaling, imputation, order)
                # This ensures we compare drift on the features as the model sees them.
                current_processed_features_df = temp_prediction_generator_instance._preprocess_for_prediction(current_raw_features_df, sector)
            except ValueError as ve:
                self.logger.error(f"Error preprocessing data for drift check in sector {sector}: {ve}. Skipping drift check for this sector.")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error during preprocessing for drift check, sector {sector}: {e}")
                continue


            for feature, baseline_stats in self.data_drift_baselines[sector].items():
                if feature not in current_processed_features_df.columns:
                    self.logger.warning(f"Feature '{feature}' from baseline not in current processed data for sector '{sector}'.")
                    continue

                current_mean = current_processed_features_df[feature].mean() # Mean of the single current (processed) row
                baseline_mean = baseline_stats.get('mean')
                baseline_std = baseline_stats.get('std') # For future use (e.g. Z-score drift)

                if pd.notna(current_mean) and pd.notna(baseline_mean):
                    if abs(baseline_mean) > 1e-9: # Avoid division by zero or near-zero
                        percentage_change = abs((current_mean - baseline_mean) / baseline_mean)
                        if percentage_change > self.drift_threshold_pct:
                            msg = (f"Data Drift Alert for '{sector}'-'{feature}': "
                                   f"Mean changed by {percentage_change:.2%}. "
                                   f"Baseline: {baseline_mean:.4f}, Current: {current_mean:.4f}")
                            self.logger.warning(msg)
                            drift_alerts.setdefault(sector, []).append(msg)
                    elif abs(current_mean - baseline_mean) > (self.drift_threshold_pct * (baseline_std if pd.notna(baseline_std) and baseline_std > 1e-9 else 1.0)):
                        # If baseline mean is near zero, check absolute change relative to std dev (heuristic)
                        msg = (f"Data Drift Alert for '{sector}'-'{feature}' (near zero mean): "
                               f"Absolute change significant. Baseline: {baseline_mean:.4f}, Current: {current_mean:.4f}")
                        self.logger.warning(msg)
                        drift_alerts.setdefault(sector, []).append(msg)
                # else: Cannot compare if current or baseline mean is NaN

        if not drift_alerts:
            self.logger.info("No significant data drift detected based on mean changes.")
        return drift_alerts


    def evaluate_recent_model_performance(self, recent_eval_data_map, temp_prediction_generator_instance):
        # recent_eval_data_map: {sector: (X_recent_raw_df, y_recent_series)}
        # temp_prediction_generator_instance: To use its _preprocess_for_prediction
        self.logger.info("Evaluating model performance on recent data...")
        performance_issues = {}

        trained_sector_models = self.trained_models_info.get('models', {})

        for sector, (X_raw_eval, y_eval) in recent_eval_data_map.items():
            if sector not in trained_sector_models or not trained_sector_models[sector]:
                self.logger.debug(f"No trained models for sector {sector} to evaluate performance.")
                continue
            if X_raw_eval.empty or y_eval.empty:
                self.logger.debug(f"No recent evaluation data for sector {sector}.")
                continue

            try:
                X_processed_eval = temp_prediction_generator_instance._preprocess_for_prediction(X_raw_eval, sector)

                # For evaluation, could use ensemble or pick primary model. Using ensemble via PredictionGenerator.
                # To do this properly, PredictionGenerator needs to be able to predict on multiple rows.
                # For simplicity here, let's assume we evaluate the first model available for the sector.
                model_to_eval_name = next(iter(trained_sector_models[sector].keys()), None)
                if not model_to_eval_name: continue
                model_to_eval = trained_sector_models[sector][model_to_eval_name]

                y_pred_eval = model_to_eval.predict(X_processed_eval)

                mae = mean_absolute_error(y_eval, y_pred_eval)
                dir_acc = np.mean(np.sign(y_eval.fillna(0).values) == np.sign(pd.Series(y_pred_eval).fillna(0).values))

                if mae > self.perf_thresholds['mae_max'] or dir_acc < self.perf_thresholds['direction_accuracy_min']:
                    msg = (f"Performance Issue Alert for '{sector}' (model: {model_to_eval_name}): "
                           f"MAE={mae:.3f} (Threshold: >{self.perf_thresholds['mae_max']}), "
                           f"DirectionAccuracy={dir_acc:.2%} (Threshold: <{self.perf_thresholds['direction_accuracy_min']})")
                    self.logger.warning(msg)
                    performance_issues.setdefault(sector, []).append(msg)
                else:
                    self.logger.info(f"Recent model performance for '{sector}' (model: {model_to_eval_name}) within thresholds. MAE={mae:.3f}, DirAcc={dir_acc:.2%}")

            except ValueError as ve: # Catch errors from preprocessing
                 self.logger.error(f"ValueError during performance evaluation for sector {sector}: {ve}")
            except Exception as e:
                self.logger.error(f"Unexpected error evaluating recent model performance for sector {sector}: {e}")

        if not performance_issues:
            self.logger.info("Recent model performance appears within thresholds for all checked sectors.")
        return performance_issues

    def suggest_retraining_trigger(self, sector_name, reason_message):
        self.logger.critical(f"RETRAINING SUGGESTED for sector '{sector_name}' due to: {reason_message}. Manual review and action may be required based on system design.")
        # In a fully automated system, this could trigger a retraining pipeline or alert an operator.
        # For this project, it's a log message.
        # print(f"ACTION_SUGGESTED: Consider retraining model for sector '{sector_name}'. Reason: {reason_message}")


# Note: The model_config would typically be imported from a central config file.
# For now, it's assumed to be passed or accessible.
# model_config = { ... } # Defined elsewhere or in config.settings

```
