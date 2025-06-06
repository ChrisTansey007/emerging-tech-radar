import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
import sqlite3
import os
# from ..config.settings import model_config, monitoring_config # Example imports

class SystemMonitor:
    def __init__(self, trained_models_dict, model_cfg, monitor_cfg, db_path="data/monitoring.sqlite"): # New db_path
        self.trained_models_dict = trained_models_dict
        # self.db_connection = db_connection_mock # Removed
        self.data_drift_baselines = {}
        self.data_drift_thresholds = {
            'patent_filing_rate_3m_patent': 0.20, 'funding_amount_velocity_3m_usd_funding': 0.25,
            'publication_rate_3m_research': 0.15
        }
        self.model_config = model_cfg # Store it
        self.monitoring_config = monitor_cfg # Store it
        self.model_performance_thresholds = self.model_config['performance_threshold']
        self.logger = logging.getLogger(__name__)

        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            self.logger.info(f"Created directory for monitoring database: {db_dir}")

        self.conn = sqlite3.connect(self.db_path)
        self._initialize_db()

    def _initialize_db(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pipeline_status (
                    pipeline_name TEXT PRIMARY KEY,
                    last_run_timestamp TEXT,
                    status TEXT,
                    details TEXT
                )
            ''')
            self.conn.commit()
            self.logger.info(f"Database initialized/ensured table 'pipeline_status' exists at {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing database table: {e}")
            # Depending on criticality, might raise or exit
            # For now, log and continue; methods using DB should handle conn absence or errors.

    def _get_pipeline_status_from_db(self, pipeline_name):
        # if self.db_connection: # Old mock logic
        # Replaced with actual DB query if self.conn is available
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT last_run_timestamp, status, details FROM pipeline_status WHERE pipeline_name=?", (pipeline_name,))
                row = cursor.fetchone()
                if row:
                    last_run_ts_str, status, details = row
                    # last_run_dt = datetime.fromisoformat(last_run_ts_str) if last_run_ts_str else None # Parsed in consumer
                    return {'last_run_timestamp': last_run_ts_str, 'status': status, 'details': details}
                else:
                    self.logger.info(f"No status found in DB for pipeline: {pipeline_name}")
                    return {'last_run_timestamp': None, 'status': 'NOT_FOUND', 'details': 'Pipeline status not found in DB.'}
            except sqlite3.Error as e:
                self.logger.error(f"Error querying pipeline status for {pipeline_name}: {e}")
                return {'last_run_timestamp': None, 'status': 'DB_ERROR', 'details': str(e)}

        self.logger.warning("No database connection available for _get_pipeline_status_from_db.")
        return {'last_run_timestamp': None, 'status': 'NO_DB_CONN', 'details': 'Database connection not available.'}

    def update_pipeline_status(self, pipeline_name, status, details=""):
        timestamp = datetime.now().isoformat()
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO pipeline_status
                    (pipeline_name, last_run_timestamp, status, details)
                    VALUES (?, ?, ?, ?)
                ''', (pipeline_name, timestamp, status, details))
                self.conn.commit()
                self.logger.info(f"Updated status for pipeline '{pipeline_name}' to '{status}'.")
            except sqlite3.Error as e:
                self.logger.error(f"Error updating pipeline status for '{pipeline_name}': {e}")
        else:
            self.logger.error(f"Cannot update pipeline status for '{pipeline_name}': No database connection.")


    def check_data_pipeline_health(self):
        self.logger.info("Starting data pipeline health check...")
        pipelines_to_check = {
            "patents_pipeline": {'max_delay_days': 2, 'error_if_not_success': True},
            "funding_pipeline": {'max_delay_days': 8, 'error_if_not_success': True},
            "research_pipeline": {'max_delay_days': 2, 'error_if_not_success': True},
        }
        all_healthy = True
        for pipeline_name, config_params in pipelines_to_check.items(): # Renamed config to config_params to avoid clash
            status_info = self._get_pipeline_status_from_db(pipeline_name)
            last_run_str = status_info.get('last_run_timestamp')
            current_status = status_info.get('status')

            last_run_dt = None
            if last_run_str:
                try:
                    last_run_dt = datetime.fromisoformat(last_run_str)
                except ValueError:
                    self.logger.warning(f"Could not parse timestamp '{last_run_str}' for pipeline '{pipeline_name}'.")

            if not last_run_dt or (datetime.now() - last_run_dt > timedelta(days=config_params['max_delay_days'])):
                self.logger.error(f"Pipeline '{pipeline_name}' delayed. Last run: {last_run_dt}.")
                all_healthy = False
            if config_params['error_if_not_success'] and current_status != 'SUCCESS': # Assuming 'SUCCESS' is the target status
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
