import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from innovation_system.data_collection.collectors import PatentDataCollector, FundingDataCollector, ResearchDataCollector
from innovation_system.feature_engineering.engineer import FeatureEngineer
from innovation_system.model_development.predictor import InnovationPredictor
from innovation_system.prediction.generator import PredictionGenerator
from innovation_system.monitoring.monitor import SystemMonitor
from innovation_system.uncertainty_handling.manager import UncertaintyManager
from innovation_system.config.settings import (
    patent_config, funding_config, research_config,
    feature_config, model_config, prediction_config,
    monitoring_config, uncertainty_config
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # For mock models in main

# --- Main Execution / Example Usage (Conceptual) ---

# Imports will be updated later to reflect the new structure

# Temporarily include classes and configs here for the script to be runnable before full refactor.
# This is NOT the final state. These will be replaced by proper imports.
import argparse
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Innovation Prediction System CLI")

    parser.add_argument(
        "--sectors",
        type=str,
        default="AI,Biotech", # Default from design
        help='Comma-separated list of technology sectors to analyze (e.g., "AI,Biotech").'
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'), # Default: 90 days ago
        help="Start date for data collection in YYYY-MM-DD format. Defaults to 90 days ago."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'), # Default: today
        help="End date for data collection in YYYY-MM-DD format. Defaults to today."
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="6,12,24", # Default from design, or use model_config
        help='Comma-separated list of prediction horizons in months (e.g., "6,12,18").'
    )
    parser.add_argument(
        "--force-collect",
        action="store_true", # Creates a boolean flag, default is False
        help="Force data collection/generation even if Parquet files exist."
    )

    args = parser.parse_args()

    # Define Parquet file paths
    DATA_DIR = "data/raw" # For feature data
    MONITORING_DB_DIR = "data" # For monitoring database

    PATENTS_FILE = os.path.join(DATA_DIR, "patents.parquet")
    FUNDING_FILE = os.path.join(DATA_DIR, "funding.parquet")
    RESEARCH_FILE = os.path.join(DATA_DIR, "research_papers.parquet")

    # Ensure data directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    # MONITORING_DB_DIR ("data") is parent of DATA_DIR ("data/raw"), so it's implicitly created.
    if MONITORING_DB_DIR and not os.path.exists(MONITORING_DB_DIR): # Check explicitly if it's different or needs creation
         os.makedirs(MONITORING_DB_DIR, exist_ok=True)

    MONITORING_DB_FILE = os.path.join(MONITORING_DB_DIR, "monitoring.sqlite")

    # Process parsed arguments
    cli_sectors = [s.strip() for s in args.sectors.split(',')]
    try:
        cli_start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        cli_end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format.")
        exit(1)

    if cli_end_date < cli_start_date:
        print("Error: End date cannot be before start date.")
        exit(1)

    try:
        cli_horizons = [int(h.strip()) for h in args.horizons.split(',')]
    except ValueError:
        print("Error: Horizons must be comma-separated integers.")
        exit(1)

    print("--- Innovation Prediction System ---")
    print(f"Running analysis for sectors: {cli_sectors}")
    print(f"Data collection period: {cli_start_date.strftime('%Y-%m-%d')} to {cli_end_date.strftime('%Y-%m-%d')}")
    print(f"Prediction horizons: {cli_horizons} months")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s [%(module)s:%(lineno)d] - %(message)s',
                        filename=monitoring_config.get('log_file', 'innovation_main.log'),
                        filemode='a')

    # --- Phase 1: Data Collection (Conceptual Instantiation) ---
    # patent_collector = PatentDataCollector()
    # funding_collector = FundingDataCollector()
    # research_collector = ResearchDataCollector()

    print("\n--- Starting Data Simulation/Loading ---")
    patents_file_exists = os.path.exists(PATENTS_FILE)
    funding_file_exists = os.path.exists(FUNDING_FILE)
    research_file_exists = os.path.exists(RESEARCH_FILE)
    all_data_files_exist = patents_file_exists and funding_file_exists and research_file_exists

    if args.force_collect or not all_data_files_exist:
        print("\n--- Generating and Saving Source Data ---")
        if args.force_collect: print("Reason: --force-collect specified.")
        if not all_data_files_exist: print(f"Reason: One or more data files missing (Patents: {patents_file_exists}, Funding: {funding_file_exists}, Research: {research_file_exists}).")

        num_periods_calc = (cli_end_date.year - cli_start_date.year) * 12 + (cli_end_date.month - cli_start_date.month) + 1
        if num_periods_calc <= 0:
            print(f"Error: Date range results in {num_periods_calc} periods. Provide a valid range.")
            exit(1)

        dates_idx = pd.date_range(start=cli_start_date, end=cli_end_date, freq='MS')
        num_periods_actual = len(dates_idx)

        sector_list_for_mock = []
        if cli_sectors:
            base_len_per_sector = num_periods_actual // len(cli_sectors)
            remainder = num_periods_actual % len(cli_sectors)
            for i, sector in enumerate(cli_sectors):
                sector_len = base_len_per_sector + (1 if i < remainder else 0)
                sector_list_for_mock.extend([sector] * sector_len)
        else:
            sector_list_for_mock = ['DefaultSector'] * num_periods_actual

        # Enhanced Mock Patent Data Generation
        num_mock_patents = num_periods_actual
        mock_patent_data_list = []
        for i in range(num_mock_patents):
            current_filing_date = dates_idx[i]
            num_inventors = np.random.randint(1, 4)
            inventors = [f"Inventor Last {j}, Inventor First {j}" for j in range(num_inventors)]
            num_cpcs = np.random.randint(1, 3)
            cpc_prefixes = ["G06N", "H04L", "A61K", "C12Q", "H01L", "B60W", "F03D"] # Added more variety
            cpcs = [f"{np.random.choice(cpc_prefixes)}{np.random.randint(1,20)}/{np.random.randint(100,500)}" for _ in range(num_cpcs)]

            mock_patent_data_list.append({
                'patent_id': f"PG{np.random.randint(10000000, 99999999)}",
                'title': f"Mock Invention Title {i} for sector {sector_list_for_mock[i]}",
                'filing_date': current_filing_date.strftime('%Y-%m-%d'),
                'assignee': f"Mock Assignee Company {np.random.randint(1, 100)}",
                'inventors': inventors,
                'tech_class': cpcs[0] if num_cpcs == 1 else cpcs,
                'abstract': f"This is a detailed mock abstract for patent {i} concerning {sector_list_for_mock[i]}. It includes keywords like innovation, technology, and {sector_list_for_mock[i].lower()} breakthrough. " * 2,
                'citations': np.random.randint(0, 20),
                'source': 'USPTO_mock',
                'sector_label': sector_list_for_mock[i]
            })
        mock_patent_df = pd.DataFrame(mock_patent_data_list)
        # The FeatureEngineer expects 'filing_date' as a column, and will convert it.
        # Set index to dates_idx for alignment with other time-series mock data.
        mock_patent_df.index = dates_idx[:len(mock_patent_df)]


        mock_funding_columns = ['funding_deals_velocity_6m', 'funding_amount_velocity_6m_usd', 'avg_round_size_usd', 'seed_ratio', 'series_a_ratio']
        mock_funding_df = pd.DataFrame(np.random.rand(num_periods_actual, len(mock_funding_columns)), index=dates_idx, columns=mock_funding_columns)
        mock_funding_df['sector_label'] = sector_list_for_mock

        research_collector = ResearchDataCollector()
        research_df = pd.DataFrame()
        days_back_calc = (cli_end_date - cli_start_date).days
        actual_days_back = max(1, days_back_calc)

        arxiv_categories_to_try = []
        sector_to_arxiv_cat = { "AI": "cs.AI", "ML": "cs.LG", "CV": "cs.CV", "LG": "cs.LG", "RO": "cs.RO", "CS.AI": "cs.AI", "CS.LG": "cs.LG", "CS.CV": "cs.CV", "CS.RO": "cs.RO", "BIOTECH": "q-bio.BM", "Q-BIO.BM": "q-bio.BM", "QUANTUM": "quant-ph", "QUANT-PH": "quant-ph" }
        if cli_sectors:
            for sector in cli_sectors:
                cat = sector_to_arxiv_cat.get(sector, sector_to_arxiv_cat.get(sector.upper(), sector))
                arxiv_categories_to_try.append(cat)
        else: arxiv_categories_to_try.append("cs.AI")

        print(f"Attempting live arXiv data collection for categories: {arxiv_categories_to_try} covering {actual_days_back} days...")
        try:
            live_research_papers = research_collector.collect_arxiv_papers(categories=arxiv_categories_to_try, days_back=actual_days_back)
            if live_research_papers:
                research_df = pd.DataFrame(live_research_papers)
                if 'published_date' in research_df.columns: research_df['published_date'] = pd.to_datetime(research_df['published_date'])
                print(f"Successfully collected {len(research_df)} arXiv papers.")
            else: print("No papers collected from arXiv for the given criteria.")
        except Exception as e: print(f"Error during arXiv data collection: {e}")

        if research_df.empty:
            print("Falling back to mock research data generation.")
            num_mock_papers = 50
            mock_research_data_list = [{'paper_id': f'mockArxiv{i}', 'title': f'Mock Research Paper Title {i}', 'authors': [f'Author {j}' for j in range(np.random.randint(1,4))], 'abstract': f'This is a mock abstract for paper {i}. ' * 5, 'categories': [arxiv_categories_to_try[i % len(arxiv_categories_to_try)]] if arxiv_categories_to_try else ['cs.AI'], 'published_date': (cli_end_date - timedelta(days=np.random.randint(0, actual_days_back))) if actual_days_back > 0 else cli_end_date, 'pdf_url': f'http://mock.arxiv.org/pdf/mockArxiv{i}', 'source': 'arXiv_mock', 'citation_count': 0 } for i in range(num_mock_papers)]
            research_df = pd.DataFrame(mock_research_data_list)
            if 'published_date' in research_df.columns: research_df['published_date'] = pd.to_datetime(research_df['published_date'])
            print(f"Generated {len(research_df)} mock research papers.")

        print(f"Saving mock patents data to {PATENTS_FILE}...")
        mock_patent_df.to_parquet(PATENTS_FILE)
        print(f"Saving mock funding data to {FUNDING_FILE}...")
        mock_funding_df.to_parquet(FUNDING_FILE)
        print(f"Saving research data (live or mock) to {RESEARCH_FILE}...")
        research_df.to_parquet(RESEARCH_FILE)
        print("All source data saved.")
    else:
        print("\n--- Loading Source Data from Parquet Files ---")
        mock_patent_df = pd.read_parquet(PATENTS_FILE)
        mock_funding_df = pd.read_parquet(FUNDING_FILE)
        research_df = pd.read_parquet(RESEARCH_FILE)
        print("Source data loaded from Parquet files.")

    print("\n--- Generating Features for Demo Model Training ---")
    feature_eng = FeatureEngineer(config=feature_config)

    patent_features_df = feature_eng.create_patent_features(mock_patent_df) if not mock_patent_df.empty else pd.DataFrame()
    print(f"Patent features created. Columns: {patent_features_df.columns.tolist() if not patent_features_df.empty else 'N/A'}")
    funding_features_df = feature_eng.create_funding_features(mock_funding_df) if not mock_funding_df.empty else pd.DataFrame()
    print(f"Funding features created. Columns: {funding_features_df.columns.tolist() if not funding_features_df.empty else 'N/A'}")
    research_features_df = feature_eng.create_research_features(research_df) if not research_df.empty else pd.DataFrame()
    print(f"Research features created (including NLP). Columns: {research_features_df.columns.tolist() if not research_features_df.empty else 'N/A'}")

    predictor = InnovationPredictor(random_state=123)

    num_target_periods = len(mock_patent_df)
    dates_for_targets = mock_patent_df.index
    target_sector_list = []
    if cli_sectors:
        base_len_target = num_target_periods // len(cli_sectors)
        remainder_target = num_target_periods % len(cli_sectors)
        for i, sector in enumerate(cli_sectors):
            sector_len = base_len_target + (1 if i < remainder_target else 0)
            target_sector_list.extend([sector] * sector_len)
    else:
        target_sector_list = ['DefaultSector'] * num_target_periods
    mock_targets_df = pd.DataFrame({'target_growth_6m': np.random.randn(num_target_periods) * 0.1, 'sector_label': target_sector_list}, index=dates_for_targets)
    print("Mock targets data prepared in memory.")

    print("\n--- Combining Features for Demo Predictor ---")
    aligned_idx = mock_targets_df.index
    data_dict_for_predictor = {'sector_label': mock_targets_df['sector_label'].values }

    def add_features_to_data_dict(df_features, suffix, data_dict, common_idx):
        if df_features is not None and not df_features.empty:
            if len(df_features) == 1:
                for col in df_features.columns: data_dict[f"{col}{suffix}"] = np.repeat(df_features[col].iloc[0], len(common_idx))
            else:
                df_aligned = df_features.reindex(common_idx).fillna(method='ffill').fillna(method='bfill')
                for col in df_aligned.columns:
                    if col not in ['sector_label', 'target_growth_6m', 'observation_date', 'date', 'published_date', 'filing_date', 'patent_id', 'title', 'assignee', 'inventors', 'tech_class', 'abstract', 'source', 'paper_id', 'authors', 'categories', 'pdf_url', 'company_uuid', 'company_name', 'amount_usd', 'currency', 'stage', 'abstractText', 'patentApplicationNumber']: # Avoid re-adding raw source columns
                         data_dict[f"{col}{suffix}"] = df_aligned[col].values
        else: print(f"Warning: {suffix.replace('_','')} features DataFrame is empty or None. No features added.")

    add_features_to_data_dict(patent_features_df, "_patent", data_dict_for_predictor, aligned_idx)
    add_features_to_data_dict(funding_features_df, "_funding", data_dict_for_predictor, aligned_idx)
    add_features_to_data_dict(research_features_df, "_research", data_dict_for_predictor, aligned_idx)
    data_dict_for_predictor['target_growth_6m'] = mock_targets_df['target_growth_6m'].values

    historical_data_df = pd.DataFrame(data_dict_for_predictor, index=aligned_idx)
    historical_data_df = historical_data_df.fillna(historical_data_df.median(numeric_only=True))
    historical_data_df = historical_data_df.dropna(subset=['target_growth_6m'])
    feature_cols_for_dropna = [col for col in historical_data_df.columns if col not in ['sector_label', 'target_growth_6m']]
    if feature_cols_for_dropna: historical_data_df = historical_data_df.dropna(subset=feature_cols_for_dropna, how='all')

    X_train_demo, y_train_demo, X_test_demo, y_test_demo = pd.DataFrame(), pd.Series(dtype='float64'), pd.DataFrame(), pd.Series(dtype='float64')
    trained_models = {}
    if historical_data_df.empty:
        print("Error: historical_data_df is empty. Model training demo cannot proceed.")
    else:
        print(f"Combined historical_data_df created. Shape: {historical_data_df.shape}")
        train_size = int(len(historical_data_df) * 0.7)
        if train_size == 0 and len(historical_data_df) > 0: train_size = 1

        X_train_demo = historical_data_df.iloc[:train_size].drop(columns=['target_growth_6m'])
        y_train_demo = historical_data_df.iloc[:train_size]['target_growth_6m']
        X_test_demo = historical_data_df.iloc[train_size:].drop(columns=['target_growth_6m'])
        y_test_demo = historical_data_df.iloc[train_size:]['target_growth_6m']

        if X_train_demo.empty or y_train_demo.empty: print("Warning: Training data is empty after splitting.")
        if X_test_demo.empty or y_test_demo.empty: print("Warning: Test data is empty after splitting.")

    emerging_techs, investment_ops, sector_forecasts = [], [], {}
    if not X_train_demo.empty and not y_train_demo.empty:
        print(f"X_train_demo shape: {X_train_demo.shape}, y_train_demo shape: {y_train_demo.shape}")
        print(f"X_train_demo columns: {X_train_demo.columns.tolist()}")
        print(f"Sample sector_label in X_train_demo: {X_train_demo['sector_label'].value_counts(dropna=False)}")

        print("\n--- Starting Model Training & Validation ---")
        trained_models = predictor.train_sector_models(X_train_demo, y_train_demo, sectors_column='sector_label')
        print("Sector models trained.")
        if not X_test_demo.empty and not y_test_demo.empty:
            validation_results = predictor.validate_models(X_test_demo, y_test_demo, sectors_column='sector_label')
            print("Model validation complete.")
        else:
            print("Skipping model validation as test data is empty.")

        if not trained_models or all(not v for v in trained_models.values()):
            print("No models were trained successfully. Skipping prediction generation.")
        else:
            print("\n--- Starting Prediction Generation ---")
            pred_gen = PredictionGenerator(trained_models, predictor.ensemble_weights, prediction_config)
            current_features_for_prediction = X_test_demo.copy()
            if not current_features_for_prediction.empty:
                sector_forecasts = pred_gen.generate_sector_forecasts(current_features_all_sectors_df=current_features_for_prediction, sector_column_name='sector_label', horizons=cli_horizons)
                print(f"Sector forecasts generated for horizons: {cli_horizons} months.")
                try:
                    latest_features_by_sector = X_test_demo.groupby('sector_label').last().reset_index()
                    emerging_techs = pred_gen.identify_emerging_technologies(all_features_df=latest_features_by_sector, sector_col='sector_label', indicators_config=prediction_config['emergence_indicators'], threshold_percentile=50)
                    print("Emerging technologies identified.")
                    mock_market_data = {sector: {'market_size_usd': np.random.randint(50, 200) * 1e9} for sector in cli_sectors}
                    investment_ops = pred_gen.create_investment_opportunities(sector_forecasts, emerging_techs, mock_market_data, prediction_config['investment_ranking_criteria'])
                    print("Investment opportunities created.")
                except Exception as e:
                    print(f"Error in post-training prediction generation (emerging tech/investment ops): {e}")
            else:
                print("Skipping prediction generation as test data is empty.")
    else:
        print("Skipping model training, validation, and prediction generation as training data is empty.")

    print("\n--- Results Summary ---")
    if emerging_techs:
        print("\nTop Emerging Technologies:")
        for i, tech in enumerate(emerging_techs[:3]): print(f"  {i+1}. {tech['technology']} (Score: {tech['emergence_score']:.2f})")
    else: print("\nNo emerging technologies identified based on current criteria.")
    if sector_forecasts:
        print("\nSector Forecasts (Growth Rate - Sample Horizon):")
        summary_horizon_key_template = "{}m_growth_rate"
        summary_horizon_to_check = cli_horizons[0]
        if 12 in cli_horizons: summary_horizon_to_check = 12
        summary_horizon_key = summary_horizon_key_template.format(summary_horizon_to_check)
        for sector, forecast_data in sector_forecasts.items():
            if summary_horizon_key in forecast_data: details = forecast_data[summary_horizon_key]; print(f"  - {sector} ({summary_horizon_key.split('_')[0]}): {details['prediction']:.3f} (Quality: {details['quality_score']:.2f})")
            elif forecast_data: first_avail_key = next(iter(forecast_data)); details = forecast_data[first_avail_key]; horizon_val_str = first_avail_key.split('m_')[0] if 'm_' in first_avail_key else first_avail_key; print(f"  - {sector} ({horizon_val_str}): {details['prediction']:.3f} (Quality: {details['quality_score']:.2f})")
    else: print("\nNo sector forecasts were generated.")
    if investment_ops:
        print("\nTop Investment Opportunity:")
        top_op = investment_ops[0]
        print(f"  - Sector/Tech: {top_op['sector']}\n    Type: {top_op['type']}\n    Recommendation: {top_op['recommended_action']} (Score: {top_op['attractiveness_score']:.2f})")
    else: print("\nNo investment opportunities identified.")

    print("\n--- Monitoring & Maintenance (Conceptual) ---")
    monitor = None
    monitor_instance_for_status_only = False
    if 'trained_models' in locals() and trained_models and any(v for v in trained_models.values()):
        monitor = SystemMonitor(trained_models, model_config, monitoring_config, db_path=MONITORING_DB_FILE)
    else:
        monitor = SystemMonitor({}, model_config, monitoring_config, db_path=MONITORING_DB_FILE)
        monitor_instance_for_status_only = True
        print("No trained models available for full monitoring setup, but pipeline status will be tracked.")

    if monitor:
        status_detail_start = (f"Run started. Sectors: {cli_sectors}, Start: {args.start_date}, End: {args.end_date}, Horizons: {cli_horizons}, ForceCollect: {args.force_collect}")
        monitor.update_pipeline_status("main_run", "STARTED", status_detail_start)
    else: print("CRITICAL: SystemMonitor could not be instantiated. Pipeline status will not be tracked.")

    if not monitor_instance_for_status_only and monitor and 'X_train_demo' in locals() and not X_train_demo.empty :
        baseline_features_df = X_train_demo.drop(columns=['sector_label']).select_dtypes(include=np.number)
        if not baseline_features_df.empty:
            monitor.set_data_drift_baseline(baseline_features_df)
            print("Data drift baseline set using a sample of training data.")
            if 'X_test_demo' in locals() and not X_test_demo.empty:
                 drift_check_features_df = X_test_demo.drop(columns=['sector_label']).select_dtypes(include=np.number).tail(10)
                 if not drift_check_features_df.empty:
                    current_ai_features_for_drift = X_test_demo[X_test_demo['sector_label']=='AI'].drop(columns=['sector_label']).select_dtypes(include=np.number).tail(5) if 'AI' in X_test_demo['sector_label'].unique() else pd.DataFrame()
                    if not current_ai_features_for_drift.empty:
                        drift_detected, _ = monitor.monitor_data_drift(current_ai_features_for_drift)
                        print(f"Data drift detected for AI sector (sample check on X_test_demo): {drift_detected}")
                 else: print("Test features for drift check are empty.")
            else: print("X_test_demo is empty or not defined, skipping drift check on test data.")
        else: print("Baseline features DataFrame for drift monitoring is empty.")

        if 'X_test_demo' in locals() and 'y_test_demo' in locals() and not X_test_demo.empty and not y_test_demo.empty:
            for sector_to_evaluate in cli_sectors:
                if sector_to_evaluate in X_test_demo['sector_label'].unique():
                    X_recent_sector_eval = X_test_demo[X_test_demo['sector_label'] == sector_to_evaluate]
                    y_recent_sector_eval = y_test_demo[y_test_demo.index.isin(X_recent_sector_eval.index)]
                    if not X_recent_sector_eval.empty and not y_recent_sector_eval.empty:
                        performance_ok = monitor.evaluate_model_performance_on_recent_data(sector_to_evaluate, X_recent_sector_eval.drop(columns=['sector_label']).select_dtypes(include=np.number), y_recent_sector_eval)
                        print(f"Model performance for {sector_to_evaluate} sector on recent data (X_test_demo) is OK: {performance_ok}")
                    else: print(f"No data for {sector_to_evaluate} sector in X_test_demo to evaluate performance.")
                else: print(f"Sector {sector_to_evaluate} not found in test data for performance evaluation.")
        else:
             print("Skipping model performance evaluation on recent data as X_test_demo or y_test_demo is empty or not defined.")
    elif monitor_instance_for_status_only:
        print("Skipping detailed monitoring tasks (drift, performance) as models were not trained.")

    print("\n--- Uncertainty Handling (Conceptual) ---")
    uncertainty_mgr = UncertaintyManager(uncertainty_config)
    if cli_sectors:
        example_sector_for_uncertainty = cli_sectors[0]
        adj1, msg1 = uncertainty_mgr.assess_data_completeness_impact("USPTO", 0.75, example_sector_for_uncertainty)
        print(msg1 if msg1 else f"Completeness OK for USPTO in {example_sector_for_uncertainty}")
        adj2, msg2, _ = uncertainty_mgr.handle_conflicting_signals(example_sector_for_uncertainty, {'pat':0.7, 'fund':-0.6})
        print(msg2)
        if 'sector_forecasts' in locals() and sector_forecasts and example_sector_for_uncertainty in sector_forecasts:
            sector_forecast_sample = next(iter(sector_forecasts.get(example_sector_for_uncertainty, {}).values()), None)
            if sector_forecast_sample:
                final_out = uncertainty_mgr.format_final_prediction_output(sector_forecast_sample['prediction'], sector_forecast_sample['quality_score'], [adj1, adj2], sector_forecast_sample.get('confidence_intervals'))
                print(f"\nFinal Prediction Output with Uncertainty ({example_sector_for_uncertainty} Sample):"); [print(f"  {k}: {v}") for k,v in final_out.items()]
            else: print(f"\nNo {example_sector_for_uncertainty} forecast details for uncertainty demo.")
        else: print(f"\nNo sector_forecasts available for {example_sector_for_uncertainty} for uncertainty demo.")
    else: print("\nNo CLI sectors provided, skipping uncertainty handling demo.")

    if monitor:
        monitor.update_pipeline_status("main_run", "COMPLETED", "Run finished successfully.")
    else: print("CRITICAL: SystemMonitor not available to log run completion status.")

    print("\n--- System Run Finished (Conceptual) ---")
