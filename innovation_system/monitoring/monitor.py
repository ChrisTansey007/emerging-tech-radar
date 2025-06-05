# Cleared for new implementation
# This file will house the SystemMonitor class.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
from sklearn.metrics import mean_absolute_error  # For model performance evaluation

# Assuming model_config and PredictionGenerator would be imported or passed if needed by methods here.
# For now, model_config is accessed globally, will be fixed when configs are centralized.
# from ..config.settings import model_config # Example of future import
# from ..prediction.generator import PredictionGenerator # Example of future import for _preprocess_for_prediction

# --- Monitoring and Maintenance ---

from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING

# Conditional import for type hinting to avoid circular dependency if PredictionGenerator also imports SystemMonitor
if TYPE_CHECKING:
    from innovation_system.prediction.generator import PredictionGenerator


class SystemMonitor:
    """
    Monitors the health and performance of the innovation prediction system.
    This includes checking data pipeline statuses, detecting data drift,
    evaluating model performance on recent data, and suggesting retraining
    when necessary.
    """

    # Type hints for class attributes
    trained_models_info: Dict[str, Any]
    model_config: Dict[str, Any]
    db_connection: Optional[Any] # Placeholder for a DB connection object type
    data_drift_baselines: Dict[str, Dict[str, Dict[str, float]]] # {sector: {feature: {'mean': val, 'std': val}}}
    drift_threshold_pct: float
    perf_thresholds: Dict[str, float]
    logger: logging.Logger

    def __init__(
        self,
        trained_models_info_dict: Dict[str, Any],
        global_model_config: Dict[str, Any],
        db_conn_mock: Optional[Any] = None, # Mock DB connection
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initializes the SystemMonitor.

        Args:
            trained_models_info_dict: Dictionary containing artifacts from model training,
                                      expected to have keys like 'models', 'scalers', 'feature_names'.
            global_model_config: Global configuration dictionary for models, including
                                 performance thresholds for monitoring.
            db_conn_mock: Optional mock database connection object for fetching pipeline statuses.
                          In a real system, this would be a live DB connection.
            logger_instance: Optional pre-configured logger instance. If None, a new
                             logger for this class will be set up.
        """
        self.trained_models_info = trained_models_info_dict
        self.model_config = global_model_config
        self.db_connection = db_conn_mock
        self.data_drift_baselines = {}
        self.drift_threshold_pct = 0.20
        self.perf_thresholds = self.model_config.get(
            "performance_thresholds_monitoring",
            {"mae_max": 0.20, "direction_accuracy_min": 0.60},
        )

        if logger_instance:
            self.logger = logger_instance
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers:
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
                    filename="system_monitor.log", # Consider making this configurable
                    filemode="a",
                )

    def _get_db_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Mock method to simulate fetching the status of a data pipeline.
        In a real system, this would query a monitoring database or log aggregator.

        Args:
            pipeline_name: The name of the pipeline to check.

        Returns:
            A dictionary with status information, including 'last_success_run_utc' (datetime)
            and 'status' (str, e.g., 'SUCCESS', 'FAILED').
        """
        self.logger.debug(f"Querying DB for status of pipeline: {pipeline_name}")
        if self.db_connection:
            # Example: return self.db_connection.query(f"SELECT status, last_success_run FROM pipeline_logs WHERE name = '{pipeline_name}' ORDER BY timestamp DESC LIMIT 1")
            # For now, returning mock data.
            pass

        # Mocked successful response:
        return {
            "last_success_run_utc": datetime.now(timezone.utc)
            - timedelta(hours=np.random.randint(1, 24)), # Simulate recent success
            "status": "SUCCESS",
        }

    def check_all_data_pipelines_health(
        self, pipeline_configs_map: Dict[str, Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks the health of all configured data pipelines based on their last successful run
        and status against configured staleness thresholds.

        Args:
            pipeline_configs_map: A dictionary where keys are pipeline names and values are
                                  dictionaries containing configuration for that pipeline,
                                  e.g., {'max_staleness_days': int}.

        Returns:
            A tuple: (overall_healthy_status, all_pipeline_statuses)
                     - overall_healthy_status (bool): True if all pipelines are healthy, False otherwise.
                     - all_pipeline_statuses (dict): Detailed status for each checked pipeline.
        """
        self.logger.info("Checking data pipeline health...")
        overall_healthy: bool = True
        all_statuses: Dict[str, Any] = {}

        for name, config in pipeline_configs_map.items():
            status: Dict[str, Any] = self._get_db_pipeline_status(name)
            all_statuses[name] = status
            max_staleness_days: int = config.get("max_staleness_days", 1)
            max_staleness_delta: timedelta = timedelta(days=max_staleness_days)

            last_run_utc: datetime = status.get(
                "last_success_run_utc", datetime.min.replace(tzinfo=timezone.utc)
            )
            current_pipeline_status: str = status.get("status", "UNKNOWN")

            if current_pipeline_status != "SUCCESS" or (datetime.now(timezone.utc) - last_run_utc) > max_staleness_delta:
                self.logger.error(
                    f"Pipeline '{name}' issue: Status={current_pipeline_status}, LastSuccessRun={last_run_utc}, MaxStaleness={max_staleness_delta}."
                )
                overall_healthy = False
            else:
                self.logger.info(
                    f"Pipeline '{name}' is healthy. Last run: {last_run_utc}"
                )

        if overall_healthy:
            self.logger.info("All data pipelines appear healthy.")
        else:
            self.logger.warning("One or more data pipelines have issues.")
        return overall_healthy, all_statuses

    def establish_drift_baselines(
        self, historical_X_full_df: pd.DataFrame, sector_id_col: str
    ) -> None:
        """
        Establishes baseline statistics (mean, std) for numeric features for each sector.
        These baselines are used later for data drift detection. Baselines are stored
        in `self.data_drift_baselines`.

        Args:
            historical_X_full_df: DataFrame containing historical feature data for all sectors.
                                  It must include the `sector_id_col`.
            sector_id_col: The name of the column in `historical_X_full_df` that
                           identifies the sector for each row.
        """
        self.logger.info("Establishing data drift baselines...")
        self.data_drift_baselines = {}  # Reset baselines

        if historical_X_full_df.empty or sector_id_col not in historical_X_full_df.columns:
            self.logger.error(
                "Cannot establish drift baselines: Historical data is empty or sector ID column missing."
            )
            return

        unique_sectors: np.ndarray = historical_X_full_df[sector_id_col].unique()
        feature_names_map: Dict[str, List[str]] = self.trained_models_info.get("feature_names", {})

        for sector in unique_sectors:
            if sector not in feature_names_map:
                self.logger.debug(
                    f"No trained feature names for sector '{sector}', cannot establish drift baseline."
                )
                continue

            relevant_features_for_sector: List[str] = feature_names_map[sector]
            X_sector: pd.DataFrame = historical_X_full_df[
                historical_X_full_df[sector_id_col] == sector
            ]

            X_sector_relevant_numeric: pd.DataFrame = X_sector[
                relevant_features_for_sector
            ].select_dtypes(include=np.number)

            if X_sector_relevant_numeric.empty:
                self.logger.warning(
                    f"No numeric data for baseline for sector '{sector}' with features: {relevant_features_for_sector}"
                )
                continue

            sector_baselines: Dict[str, Dict[str, float]] = X_sector_relevant_numeric.agg(["mean", "std"]).to_dict()
            self.data_drift_baselines[sector] = {
                feat: stats
                for feat, stats in sector_baselines.items()
                if pd.notna(stats.get("mean")) and pd.notna(stats.get("std"))
            }
            self.logger.debug(
                f"Baselines for sector '{sector}': {self.data_drift_baselines[sector]}"
            )
        self.logger.info(
            f"Data drift baselines established for {len(self.data_drift_baselines)} sectors."
        )

    def monitor_for_data_drift(
        self,
        current_features_per_sector_map: Dict[str, pd.DataFrame],
        prediction_generator: 'PredictionGenerator' # Type hint using forward reference string
    ) -> Dict[str, List[str]]:
        """
        Monitors for data drift by comparing current feature statistics against established baselines.
        Requires a PredictionGenerator instance to preprocess current features consistently with model training.

        Args:
            current_features_per_sector_map: A dictionary where keys are sector names and
                                             values are DataFrames of current raw features for that sector.
            prediction_generator: An instance of the `PredictionGenerator` class, used for its
                                 `_preprocess_for_prediction` method.
        Returns:
            A dictionary of drift alerts, where keys are sectors and values are lists
            of messages describing detected drifts for features in that sector.
        """
        self.logger.info("Monitoring for data drift...")
        drift_alerts: Dict[str, List[str]] = {}

        if not self.data_drift_baselines:
            self.logger.warning("No data drift baselines established. Cannot monitor for drift.")
            return drift_alerts

        for sector, current_raw_features_df in current_features_per_sector_map.items():
            if sector not in self.data_drift_baselines:
                self.logger.debug(f"No baseline for sector '{sector}'. Skipping drift check.")
                continue
            if current_raw_features_df.empty:
                self.logger.debug(f"Empty current features for sector '{sector}'. Skipping drift check.")
                continue

            try:
                current_processed_features_df: pd.DataFrame = (
                    prediction_generator._preprocess_for_prediction( # type: ignore
                        current_raw_features_df, sector
                    )
                )
            except ValueError as ve:
                self.logger.error(
                    f"Error preprocessing data for drift check in sector '{sector}': {ve}. Skipping drift check for this sector."
                )
                continue
            except Exception as e: # Catch any other unexpected error during preprocessing
                self.logger.error(f"Unexpected error during preprocessing for drift check, sector '{sector}': {e}")
                continue

            sector_drift_details: List[str] = []
            for feature, baseline_stats in self.data_drift_baselines[sector].items():
                if feature not in current_processed_features_df.columns:
                    self.logger.warning(
                        f"Feature '{feature}' from baseline not in current processed data for sector '{sector}'."
                    )
                    continue

                current_mean: float = current_processed_features_df[feature].mean()
                baseline_mean: Optional[float] = baseline_stats.get("mean")
                baseline_std: Optional[float] = baseline_stats.get("std")

                if pd.notna(current_mean) and pd.notna(baseline_mean) and baseline_mean is not None: # Ensure baseline_mean is not None for type checker
                    # Standard drift check: percentage change from baseline mean
                    if abs(baseline_mean) > 1e-9:
                        percentage_change: float = abs((current_mean - baseline_mean) / baseline_mean)
                        if percentage_change > self.drift_threshold_pct:
                            msg = (
                                f"Feature '{feature}': Mean changed by {percentage_change:.2%} "
                                f"(Baseline: {baseline_mean:.4f}, Current: {current_mean:.4f})"
                            )
                            sector_drift_details.append(msg)
                    # Heuristic for near-zero baseline mean:
                    # Compare absolute change to a fraction of baseline standard deviation.
                    elif baseline_std is not None and pd.notna(baseline_std) and abs(current_mean - baseline_mean) > (
                        self.drift_threshold_pct * (baseline_std if baseline_std > 1e-9 else 1.0)
                    ):
                        msg = (
                            f"Feature '{feature}' (near zero baseline mean): "
                            f"Absolute change {abs(current_mean - baseline_mean):.4f} is significant relative to baseline std dev {baseline_std:.4f}. "
                            f"Baseline Mean: {baseline_mean:.4f}, Current Mean: {current_mean:.4f}"
                        )
                        sector_drift_details.append(msg)

            if sector_drift_details:
                self.logger.warning(f"Data drift detected for sector '{sector}': {'; '.join(sector_drift_details)}")
                drift_alerts[sector] = sector_drift_details

        if not drift_alerts:
            self.logger.info("No significant data drift detected based on mean changes.")
        return drift_alerts

    def evaluate_recent_model_performance(
        self,
        recent_eval_data_map: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        prediction_generator: 'PredictionGenerator' # Type hint using forward reference string
    ) -> Dict[str, List[str]]:
        """
        Evaluates model performance on recent (hold-out) data for each sector.
        Requires a PredictionGenerator instance to preprocess features consistently.

        Args:
            recent_eval_data_map: A dictionary mapping sector names to tuples of
                                  (X_recent_raw_df, y_recent_series).
            prediction_generator: An instance of `PredictionGenerator` for preprocessing.

        Returns:
            A dictionary of performance issues, where keys are sector names and
            values are lists of messages describing any performance degradation.
        """
        self.logger.info("Evaluating model performance on recent data...")
        performance_issues: Dict[str, List[str]] = {}

        trained_sector_models: Dict[str, Dict[str, Any]] = self.trained_models_info.get("models", {})

        for sector, (X_raw_eval, y_eval) in recent_eval_data_map.items():
            if sector not in trained_sector_models or not trained_sector_models[sector]:
                self.logger.debug(f"No trained models for sector '{sector}' to evaluate performance.")
                continue
            if X_raw_eval.empty or y_eval.empty:
                self.logger.debug(f"No recent evaluation data for sector '{sector}'.")
                continue

            try:
                X_processed_eval: pd.DataFrame = prediction_generator._preprocess_for_prediction( # type: ignore
                    X_raw_eval, sector
                )

                # Evaluate the first model available for the sector (simplification)
                # A more complex system might evaluate an ensemble or a specific production model.
                model_to_eval_name: Optional[str] = next(iter(trained_sector_models[sector].keys()), None)
                if not model_to_eval_name:
                    self.logger.warning(f"No model found to evaluate for sector '{sector}' though entry exists.")
                    continue

                model_to_eval: Any = trained_sector_models[sector][model_to_eval_name]
                y_pred_eval: np.ndarray = model_to_eval.predict(X_processed_eval)

                mae: float = mean_absolute_error(y_eval, y_pred_eval)
                dir_acc: float = np.mean(
                    np.sign(y_eval.fillna(0).values) == np.sign(pd.Series(y_pred_eval).fillna(0).values)
                )

                if (mae > self.perf_thresholds["mae_max"] or
                    dir_acc < self.perf_thresholds["direction_accuracy_min"]):
                    msg = (
                        f"Performance Issue Alert for '{sector}' (model: {model_to_eval_name}): "
                        f"MAE={mae:.3f} (Threshold: >{self.perf_thresholds['mae_max']}), "
                        f"DirectionAccuracy={dir_acc:.2%} (Threshold: <{self.perf_thresholds['direction_accuracy_min']})"
                    )
                    self.logger.warning(msg)
                    performance_issues.setdefault(sector, []).append(msg)
                else:
                    self.logger.info(
                        f"Recent model performance for '{sector}' (model: {model_to_eval_name}) within thresholds. MAE={mae:.3f}, DirAcc={dir_acc:.2%}"
                    )

            except ValueError as ve:
                 self.logger.error(f"ValueError during performance evaluation for sector '{sector}': {ve}")
            except Exception as e:
                self.logger.error(f"Unexpected error evaluating recent model performance for sector '{sector}': {e}")

        if not performance_issues:
            self.logger.info("Recent model performance appears within thresholds for all checked sectors.")
        return performance_issues

    def suggest_retraining_trigger(self, sector_name: str, reason_message: str) -> None:
        """
        Logs a suggestion for model retraining for a specific sector.
        In a production system, this could trigger an automated retraining pipeline or alert.

        Args:
            sector_name: The name of the sector for which retraining is suggested.
            reason_message: The reason for the retraining suggestion.
        """
        self.logger.critical(
            f"RETRAINING SUGGESTED for sector '{sector_name}' due to: {reason_message}. "
            "Manual review and action may be required based on system design."
        )
