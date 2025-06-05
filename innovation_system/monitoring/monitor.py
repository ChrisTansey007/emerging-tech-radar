import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
# from ..config.settings import model_config, monitoring_config # Example imports

class SystemMonitor:
    def __init__(self, trained_models_dict, model_cfg, monitor_cfg, db_connection_mock=None): # Added configs
        self.trained_models_dict = trained_models_dict
        self.db_connection = db_connection_mock
        self.data_drift_baselines = {}
        self.data_drift_thresholds = {
            'patent_filing_rate_3m_patent': 0.20, 'funding_amount_velocity_3m_usd_funding': 0.25,
            'publication_rate_3m_research': 0.15
        }
        self.model_config = model_cfg # Store it
        self.monitoring_config = monitor_cfg # Store it
        self.model_performance_thresholds = self.model_config['performance_threshold']
        self.logger = logging.getLogger(__name__)

    def _get_pipeline_status_from_db(self, pipeline_name):
        if self.db_connection:
            # Mock query
            if pipeline_name in ["patents_pipeline", "research_pipeline"]:
                return {'last_success_run': datetime.now() - timedelta(hours=12), 'status': 'SUCCESS'}
            elif pipeline_name == "funding_pipeline":
                 return {'last_success_run': datetime.now() - timedelta(days=3), 'status': 'SUCCESS'}
        return {'last_success_run': datetime.now() - timedelta(days=10), 'status': 'UNKNOWN'}

    def check_data_pipeline_health(self):
        self.logger.info("Starting data pipeline health check...")
        pipelines_to_check = {
            "patents_pipeline": {'max_delay_days': 2, 'error_if_not_success': True},
            "funding_pipeline": {'max_delay_days': 8, 'error_if_not_success': True},
            "research_pipeline": {'max_delay_days': 2, 'error_if_not_success': True},
        }
        all_healthy = True
        for pipeline_name, config in pipelines_to_check.items():
            status_info = self._get_pipeline_status_from_db(pipeline_name)
            last_run = status_info.get('last_success_run')
            current_status = status_info.get('status')
            if not last_run or (datetime.now() - last_run > timedelta(days=config['max_delay_days'])):
                self.logger.error(f"Pipeline '{pipeline_name}' delayed. Last run: {last_run}.")
                all_healthy = False
            if config['error_if_not_success'] and current_status != 'SUCCESS':
                self.logger.error(f"Pipeline '{pipeline_name}' status '{current_status}'. Expected SUCCESS.")
                all_healthy = False
        if all_healthy: self.logger.info("Data pipelines healthy.")
        else: self.logger.warning("One or more data pipelines have issues.")
        return all_healthy

    def set_data_drift_baseline(self, historical_features_df):
        self.logger.info("Setting data drift baselines...")
        for feature_name in self.data_drift_thresholds.keys(): # Use keys from instance var
            if feature_name in historical_features_df.columns:
                self.data_drift_baselines[feature_name] = {
                    'mean': historical_features_df[feature_name].mean(),
                    'std': historical_features_df[feature_name].std()
                }
            else:
                self.logger.warning(f"Feature '{feature_name}' for baseline not in historical data.")
        self.logger.info(f"Baselines set for {len(self.data_drift_baselines)} features.")

    def monitor_data_drift(self, current_features_df):
        self.logger.info("Starting data drift monitoring...")
        drift_detected_summary = {}; overall_drift = False
        for feature, baseline_stats in self.data_drift_baselines.items():
            if feature not in current_features_df.columns:
                self.logger.warning(f"Feature '{feature}' for drift not in current data. Skipping.")
                continue
            current_mean = current_features_df[feature].mean()
            baseline_mean = baseline_stats['mean']
            threshold = self.data_drift_thresholds.get(feature, 0.2)
            if pd.isna(baseline_mean) or pd.isna(current_mean):
                self.logger.warning(f"NaN mean for '{feature}'. Baseline: {baseline_mean}, Current: {current_mean}. Skipping.")
                continue
            percentage_change = abs(current_mean - baseline_mean) / abs(baseline_mean) if baseline_mean != 0 else (float('inf') if current_mean != 0 else 0)
            if percentage_change > threshold:
                self.logger.warning(f"Data drift for '{feature}'. Mean change: {percentage_change:.2%} (Base: {baseline_mean:.2f}, Curr: {current_mean:.2f})")
                drift_detected_summary[feature] = {'change': percentage_change, 'status': 'DRIFT'}
                overall_drift = True
            else:
                drift_detected_summary[feature] = {'change': percentage_change, 'status': 'STABLE'}
        if not overall_drift: self.logger.info("No significant data drift detected.")
        else: self.logger.warning(f"Data drift detected: { {k:v for k,v in drift_detected_summary.items() if v['status']=='DRIFT'} }")
        return overall_drift, drift_detected_summary

    def evaluate_model_performance_on_recent_data(self, sector_name, X_recent_sector, y_recent_sector):
        self.logger.info(f"Evaluating model performance for {sector_name} on recent data.")
        if sector_name not in self.trained_models_dict or not self.trained_models_dict[sector_name]:
            self.logger.error(f"No models for {sector_name} during performance evaluation.")
            return False
        sector_models = self.trained_models_dict[sector_name]
        performance_degraded = False
        for model_name, model_obj in sector_models.items():
            try:
                # Ensure X_recent_sector columns match model's expected features
                if hasattr(model_obj, 'feature_names_in_'):
                    X_recent_ordered = X_recent_sector[model_obj.feature_names_in_]
                else:
                    X_recent_ordered = X_recent_sector # Assume ordered

                y_pred = model_obj.predict(X_recent_ordered)
                mae = mean_absolute_error(y_recent_sector, y_pred)
                direction_accuracy = np.mean(np.sign(y_recent_sector.fillna(0)) == np.sign(pd.Series(y_pred).fillna(0)))
                self.logger.info(f"  Recent Perf - {model_name}, {sector_name} - MAE: {mae:.4f}, DirAcc: {direction_accuracy:.2%}")
                if mae > self.model_performance_thresholds['mae'] or direction_accuracy < self.model_performance_thresholds['direction_accuracy']:
                    self.logger.warning(f"  Perf degradation: {model_name} in {sector_name}. MAE: {mae:.4f}, DirAcc: {direction_accuracy:.2%}")
                    performance_degraded = True
            except Exception as e:
                self.logger.error(f"  Error evaluating {model_name} for {sector_name}: {e}")
                performance_degraded = True
        if performance_degraded: self.logger.warning(f"Overall performance for '{sector_name}' degraded. Retraining recommended.")
        else: self.logger.info(f"Performance for '{sector_name}' OK on recent data.")
        return not performance_degraded

    def trigger_retraining(self, sector_name):
        self.logger.info(f"Retraining triggered for sector: {sector_name}")
        print(f"ACTION: Retrain model for sector {sector_name}")

    def run_scheduled_maintenance(self, current_features_df_map, recent_evaluation_data_map):
        self.logger.info("Starting scheduled system maintenance...")
        if not self.check_data_pipeline_health():
            self.logger.error("Aborting maintenance due to pipeline health issues.")
            return
        self.logger.info("Checking data drift...")
        for sector, features_df in current_features_df_map.items():
            if not self.data_drift_baselines:
                self.logger.warning("Drift baselines not set. Set baselines first.")
                break
            drifted, _ = self.monitor_data_drift(features_df)
            if drifted: self.logger.warning(f"Drift detected for {sector}. Consider model impact.")
        self.logger.info("Evaluating model performance...")
        for sector, (X_eval, y_eval) in recent_evaluation_data_map.items():
            # Use monitoring_config from self.monitoring_config
            performance_ok = self.evaluate_model_performance_on_recent_data(sector, X_eval, y_eval)
            if not performance_ok and self.monitoring_config.get('auto_retraining_enabled', False):
                self.trigger_retraining(sector)
        self.logger.info("Scheduled maintenance completed.")
