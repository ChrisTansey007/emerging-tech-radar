import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# joblib was imported but not used directly in InnovationPredictor snippet.
# It's often used for saving/loading models, which can be added later if needed.
# import joblib

class InnovationPredictor:
    def __init__(self, random_state=42):
        self.models_blueprints = {
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=random_state)
        }
        self.ensemble_weights = {'random_forest': 0.6, 'gradient_boosting': 0.4}
        self.sector_models = {}
        self.feature_importances = {}
        self.scaler = StandardScaler() # For _normalize_features

    def _temporal_alignment(self, patent_features, funding_features, research_features, targets_df):
        # This method's logic was mostly placeholder. True alignment is complex and data-dependent.
        # For now, it simulates a simple concat.
        df_list = []
        if not patent_features.empty: df_list.append(patent_features.add_suffix('_patent'))
        if not funding_features.empty: df_list.append(funding_features.add_suffix('_funding'))
        if not research_features.empty: df_list.append(research_features.add_suffix('_research'))
        if not df_list: return pd.DataFrame()
        aligned_data = pd.concat(df_list, axis=1)
        return aligned_data

    def _create_lagged_features(self, df, feature_columns, lags=[1, 3, 6, 12]):
        if df.empty: return pd.DataFrame()
        lagged_df = df.copy()
        for col in feature_columns:
            if col in lagged_df.columns:
                for lag in lags:
                    lagged_df[f'{col}_lag{lag}'] = lagged_df[col].shift(lag)
        return lagged_df

    def _calculate_innovation_indices(self, patent_features_norm, funding_features_norm, research_features_norm, config):
        # This method needs feature_config, which should be passed or accessed from a config module
        patent_score = 0
        if not patent_features_norm.empty:
            weighted_sum = sum(patent_features_norm.get(feat, 0) * weight for feat, weight in config['patent_weights'].items() if feat in patent_features_norm)
            total_weight = sum(weight for feat, weight in config['patent_weights'].items() if feat in patent_features_norm)
            patent_score = weighted_sum / total_weight if total_weight > 0 else 0

        funding_score = 0
        if not funding_features_norm.empty:
            weighted_sum = sum(funding_features_norm.get(feat, 0) * weight for feat, weight in config['funding_weights'].items() if feat in funding_features_norm)
            total_weight = sum(weight for feat, weight in config['funding_weights'].items() if feat in funding_features_norm)
            funding_score = weighted_sum / total_weight if total_weight > 0 else 0

        research_score = 0
        if not research_features_norm.empty:
            weighted_sum = sum(research_features_norm.get(feat, 0) * weight for feat, weight in config['research_weights'].items() if feat in research_features_norm)
            total_weight = sum(weight for feat, weight in config['research_weights'].items() if feat in research_features_norm)
            research_score = weighted_sum / total_weight if total_weight > 0 else 0

        innovation_index = (0.40 * patent_score + 0.35 * funding_score + 0.25 * research_score)
        commercial_readiness = (0.60 * funding_score + 0.40 * patent_score)
        research_momentum = (0.70 * research_score + 0.30 * patent_score)

        return pd.DataFrame({
            'innovation_index': [innovation_index],
            'commercial_readiness_index': [commercial_readiness],
            'research_momentum_index': [research_momentum]
        })

    def _normalize_features(self, features_df, method='z_score'):
        if features_df.empty: return features_df
        if method == 'z_score':
            # Ensure scaler is fitted only on training data.
            # For simplicity here, it might be fit_transformed.
            # Proper way: fit scaler on train, transform on train/test.
            return pd.DataFrame(self.scaler.fit_transform(features_df), columns=features_df.columns, index=features_df.index)
        elif method == 'min_max':
            min_max_scaler = MinMaxScaler()
            return pd.DataFrame(min_max_scaler.fit_transform(features_df), columns=features_df.columns, index=features_df.index)
        return features_df

    def prepare_training_data(self, historical_patent_features_list, historical_funding_features_list, historical_research_features_list, historical_targets_df, feature_engineering_config):
        print("Warning: prepare_training_data is highly dependent on specific data structures and ETL.")
        print("This example assumes X and y are pre-prepared for train_sector_models.")
        # This would involve calls to _temporal_alignment, _normalize_features, _calculate_innovation_indices, _create_lagged_features
        # Returning placeholders.
        return pd.DataFrame(), pd.Series()

    def _get_param_grid(self, model_name):
        if model_name == 'random_forest':
            return {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]} # Reduced for speed
        elif model_name == 'gradient_boosting':
            return {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5], 'subsample': [0.8, 1.0]} # Reduced
        return {}

    def train_sector_models(self, X_train_full, y_train_full, sectors_column='sector_label'):
        if X_train_full.empty or y_train_full.empty:
            print("Training data is empty. Skipping model training.")
            return {}
        tscv = TimeSeriesSplit(n_splits=3) # Reduced splits for speed
        unique_sectors = X_train_full[sectors_column].unique()
        for sector in unique_sectors:
            print(f"Training models for sector: {sector}...")
            sector_mask = X_train_full[sectors_column] == sector
            X_sector = X_train_full[sector_mask].drop(columns=[sectors_column])
            y_sector = y_train_full[sector_mask]
            X_sector_numeric = X_sector.select_dtypes(include=np.number).fillna(X_sector.select_dtypes(include=np.number).median())
            if len(X_sector_numeric) < 20: # Min samples
                print(f"Insufficient data for {sector}: {len(X_sector_numeric)} samples. Skipping.")
                continue
            self.sector_models[sector] = {}
            self.feature_importances[sector] = {}
            for model_name, model_blueprint in self.models_blueprints.items():
                model = model_blueprint
                param_grid = self._get_param_grid(model_name)
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
                try:
                    grid_search.fit(X_sector_numeric, y_sector)
                    best_model = grid_search.best_estimator_
                    self.sector_models[sector][model_name] = best_model
                    print(f"  {sector} - {model_name}: Best MAE = {-grid_search.best_score_:.4f}")
                    if hasattr(best_model, 'feature_importances_'):
                        importances = pd.Series(best_model.feature_importances_, index=X_sector_numeric.columns)
                        self.feature_importances[sector][model_name] = importances.sort_values(ascending=False)
                except Exception as e:
                    print(f"  Error training {model_name} for sector {sector}: {e}")
        return self.sector_models

    def validate_models(self, X_test_full, y_test_full, sectors_column='sector_label'):
        if X_test_full.empty or y_test_full.empty:
            print("Test data is empty. Skipping model validation.")
            return {}
        validation_results = {}
        unique_sectors = X_test_full[sectors_column].unique()
        for sector in unique_sectors:
            if sector not in self.sector_models or not self.sector_models[sector]:
                print(f"No trained models for sector '{sector}'. Skipping validation.")
                continue
            sector_mask = X_test_full[sectors_column] == sector
            X_sector_test = X_test_full[sector_mask].drop(columns=[sectors_column])
            y_sector_test = y_test_full[sector_mask]
            X_sector_test_numeric = X_sector_test.select_dtypes(include=np.number).fillna(X_sector_test.select_dtypes(include=np.number).median())
            if len(X_sector_test_numeric) == 0: continue
            sector_results = {}
            for model_name, model in self.sector_models[sector].items():
                try:
                    y_pred = model.predict(X_sector_test_numeric)
                    mae = mean_absolute_error(y_sector_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_sector_test, y_pred))
                    direction_accuracy = np.mean(np.sign(y_sector_test.fillna(0)) == np.sign(pd.Series(y_pred).fillna(0))) if len(y_sector_test) > 0 else 0
                    sector_results[model_name] = {'mae': mae, 'rmse': rmse, 'direction_accuracy': direction_accuracy, 'predictions_sample': y_pred[:3].tolist(), 'actuals_sample': y_sector_test[:3].tolist()}
                    print(f"  Validation {sector} - {model_name}: MAE={mae:.4f}, DirAcc={direction_accuracy:.2%}")
                except Exception as e:
                    print(f"  Error validating {model_name} for sector {sector}: {e}")
                    sector_results[model_name] = {'error': str(e)}
            validation_results[sector] = sector_results
        return validation_results
