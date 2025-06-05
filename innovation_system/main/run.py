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

    args = parser.parse_args()

    # Process parsed arguments
    cli_sectors = [s.strip() for s in args.sectors.split(',')]
    try:
        cli_start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        cli_end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format.")
        exit(1) # Or raise an error

    if cli_end_date < cli_start_date:
        print("Error: End date cannot be before start date.")
        exit(1) # Or raise an error

    try:
        cli_horizons = [int(h.strip()) for h in args.horizons.split(',')]
    except ValueError:
        print("Error: Horizons must be comma-separated integers.")
        exit(1) # Or raise an error

    # Print confirmation of parsed arguments (useful for feedback)
    print("--- Innovation Prediction System ---")
    print(f"Running analysis for sectors: {cli_sectors}")
    print(f"Data collection period: {cli_start_date.strftime('%Y-%m-%d')} to {cli_end_date.strftime('%Y-%m-%d')}")
    print(f"Prediction horizons: {cli_horizons} months")

    # Setup basic logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s [%(module)s:%(lineno)d] - %(message)s',
                        filename=monitoring_config.get('log_file', 'innovation_main.log'), # Use config
                        filemode='a')

    # --- Phase 1: Data Collection (Conceptual Instantiation) ---
    # patent_collector = PatentDataCollector()
    # funding_collector = FundingDataCollector(crunchbase_key="YOUR_CRUNCHBASE_KEY")
    # research_collector = ResearchDataCollector()
    # research_collector.pubmed_api_key = "YOUR_PUBMED_KEY"

    # Example collection (simplified, adapted for CLI args)
    # Note: The actual collection calls would likely involve looping through cli_sectors for some collectors,
    # or passing the list if the collector method supports it.
    # For USPTO, which often takes a single category, we might pick the first or loop.

    # For patent_collector.collect_uspto_patents:
    # Example: Collect for the first specified sector, or a default if none.
    # uspto_category_example = cli_sectors[0] if cli_sectors else "G06N" # Example: use first sector or default
    # collected_patents_df = pd.DataFrame(patent_collector.collect_uspto_patents(cli_start_date, cli_end_date, uspto_category_example))
    # print(f"Collected {len(collected_patents_df)} patents for category {uspto_category_example} (sample).")

    # For funding_collector.collect_funding_rounds:
    # Assumes collect_funding_rounds can take a list of sector names (keywords).
    # collected_funding_df = pd.DataFrame(funding_collector.collect_funding_rounds(cli_start_date.strftime('%Y-%m-%d'), cli_sectors))
    # print(f"Collected {len(collected_funding_df)} funding rounds for sectors {cli_sectors} (sample).")

    # For research_collector (ArXiv and PubMed):
    # The days_back parameter should be calculated from cli_start_date and cli_end_date.
    # research_days_back = (cli_end_date - cli_start_date).days
    # collected_arxiv_papers_df = pd.DataFrame(research_collector.collect_arxiv_papers(cli_sectors, days_back=research_days_back))
    # collected_pubmed_papers_df = pd.DataFrame(research_collector.collect_pubmed_papers(cli_sectors, days_back=research_days_back))
    # print(f"Collected {len(collected_arxiv_papers_df)} ArXiv papers and {len(collected_pubmed_papers_df)} PubMed papers for sectors {cli_sectors} (sample).")
    # combined_research_df = pd.concat([collected_arxiv_papers_df, collected_pubmed_papers_df])
    # print(f"Collected {len(combined_research_df)} total research papers (sample).")

    # --- Phase 2: Feature Engineering (Conceptual) ---
    # feature_eng = FeatureEngineer()
    # patent_features = feature_eng.create_patent_features(ai_patents_df) if not ai_patents_df.empty else pd.DataFrame()
    # funding_features = feature_eng.create_funding_features(ai_funding_df) if not ai_funding_df.empty else pd.DataFrame()
    # research_features = feature_eng.create_research_features(ai_research_df) if not ai_research_df.empty else pd.DataFrame()
    # print("Patent Features (sample): ", patent_features.head())

    # --- Phase 3: Predictive Model Development (Conceptual) ---
    # print("\n--- Model Training & Prediction (Conceptual) ---") # Original generic print
    predictor = InnovationPredictor(random_state=123)

    print("\n--- Starting Data Simulation and Feature Preparation ---")
    # Adapt mock data generation based on CLI args
    # Calculate number of monthly periods between cli_start_date and cli_end_date
    num_periods = (cli_end_date.year - cli_start_date.year) * 12 + (cli_end_date.month - cli_start_date.month) + 1
    if num_periods <= 0:
        print(f"Error: The start date {cli_start_date} and end date {cli_end_date} result in {num_periods} periods. Please provide a valid date range.")
        exit(1)

    dates = pd.date_range(start=cli_start_date, end=cli_end_date, freq='MS') # MS for month start
    num_periods = len(dates) # Update num_periods to actual length of dates range

    # Distribute sectors among the periods
    num_cli_sectors = len(cli_sectors)
    sector_labels = []
    if num_cli_sectors > 0:
        base_len_per_sector = num_periods // num_cli_sectors
        remainder = num_periods % num_cli_sectors
        for i, sector in enumerate(cli_sectors):
            sector_len = base_len_per_sector + (1 if i < remainder else 0)
            sector_labels.extend([sector] * sector_len)
    else: # Default if no sectors provided (though argparse has a default)
        sector_labels = ['DefaultSector'] * num_periods


    mock_patent_df = pd.DataFrame(
        np.random.rand(num_periods, 5),
        index=dates[:num_periods], # Ensure index matches data length if num_periods was adjusted
        columns=['filing_rate_3m', 'filing_rate_12m', 'tech_diversity_shannon', 'unique_inventor_count', 'citation_velocity_avg']
    )
    mock_patent_df['sector_label'] = sector_labels

    mock_funding_df = pd.DataFrame(
        np.random.rand(num_periods, 5),
        index=dates[:num_periods],
        columns=['funding_deals_velocity_6m', 'funding_amount_velocity_6m_usd', 'avg_round_size_usd', 'seed_ratio', 'series_a_ratio']
    )
    mock_funding_df['sector_label'] = sector_labels

    mock_research_df = pd.DataFrame(
        np.random.rand(num_periods, 4),
        index=dates[:num_periods],
        columns=['publication_rate_6m', 'avg_citation_count', 'avg_authors_per_paper', 'category_diversity_shannon']
    )
    mock_research_df['sector_label'] = sector_labels

    mock_targets_df = pd.DataFrame(
        np.random.randn(num_periods, 1) * 0.1,
        index=dates[:num_periods],
        columns=['target_growth_6m']
    )
    mock_targets_df['sector_label'] = sector_labels
    print("Mock data generated successfully.")

    # Call prepare_training_data
    # Pass DataFrames without 'sector_label' for features, as predictor doesn't handle it internally during alignment.
    # Target DF should only contain the target column.
    print(f"Preparing training data for sectors: {cli_sectors}...")
    X_prepared, y_prepared = predictor.prepare_training_data(
        mock_patent_df.drop(columns=['sector_label']),
        mock_funding_df.drop(columns=['sector_label']),
        mock_research_df.drop(columns=['sector_label']),
        mock_targets_df[['target_growth_6m']] # Ensure only target column is passed
    )
    print("Training data prepared.")

    if X_prepared.empty or y_prepared.empty:
        print("Error: X_prepared or y_prepared is empty after data preparation. Exiting.")
        # Exit or handle error appropriately
    else:
        # Re-integrate Sector Label and Split Data
        # Align sector labels with the index of X_prepared (which might have changed due to alignment and NaN handling)
        X_prepared['sector_label'] = mock_targets_df.loc[X_prepared.index, 'sector_label']

        # Handle any NaNs in sector_label that might arise if X_prepared.index has dates not in mock_targets_df
        # (though inner merge in _temporal_alignment on target should prevent this if target_df is source of truth for dates)
        # or if some original sector labels were NaN.
        X_prepared = X_prepared.dropna(subset=['sector_label'])
        y_prepared = y_prepared.loc[X_prepared.index] # Ensure y aligns with X after potential dropna

        if X_prepared.empty:
            print("Error: X_prepared is empty after re-integrating sector_label and NaN handling. Exiting.")
        else:
            train_size = int(len(X_prepared) * 0.7)
            X_train_demo = X_prepared.iloc[:train_size]
            y_train_demo = y_prepared.iloc[:train_size]
            X_test_demo = X_prepared.iloc[train_size:]
            y_test_demo = y_prepared.iloc[train_size:]

            print(f"X_train_demo shape: {X_train_demo.shape}, y_train_demo shape: {y_train_demo.shape}")
            print(f"X_test_demo shape: {X_test_demo.shape}, y_test_demo shape: {y_test_demo.shape}")
            print(f"X_train_demo columns: {X_train_demo.columns}")
            print(f"Sample sector_label in X_train_demo: {X_train_demo['sector_label'].value_counts()}")

            print("\n--- Starting Model Training & Validation ---")
            # Execute Training and Validation
            trained_models = predictor.train_sector_models(X_train_demo, y_train_demo, sectors_column='sector_label')
            print("Sector models trained.")
            validation_results = predictor.validate_models(X_test_demo, y_test_demo, sectors_column='sector_label')
            print("Model validation complete.")
            # print("Sample Validation Results:", validation_results) # Keep or remove detailed print as desired

            print("\n--- Starting Prediction Generation ---")
            # --- Phase 4: Prediction Generation (Conceptual) ---
            # Ensure trained_models is not empty before proceeding
            # Initialize emerging_techs and investment_ops to handle cases where they might not be created
            emerging_techs = []
            investment_ops = []
            sector_forecasts = {}

            if not trained_models or all(not v for v in trained_models.values()): # Check if dict is empty or all sector models are empty
                print("No models were trained successfully. Skipping prediction generation.")
            else:
                pred_gen = PredictionGenerator(trained_models, predictor.ensemble_weights, prediction_config)
                current_features_for_prediction = X_test_demo.copy()

                sector_forecasts = pred_gen.generate_sector_forecasts(current_features_all_sectors_df=current_features_for_prediction, sector_column_name='sector_label', horizons=cli_horizons) # Use cli_horizons
                # print(f"\nSector Forecasts (sample for {len(sector_forecasts)} sectors):") # Detailed print
                # for sector, forecasts_data in sector_forecasts.items():
                #     print(f"  Sector: {sector}")
                #     for horizon, details in forecasts_data.items(): print(f"    {horizon}: Pred={details['prediction']:.3f}, Qual={details['quality_score']:.2f}")
                print(f"Sector forecasts generated for horizons: {cli_horizons} months.")

                # The following section for identifying emerging technologies and investment opportunities
                # might require significant adjustments based on the new feature names in X_test_demo
                # (e.g., 'filing_rate_3m_patent' instead of 'pfeat1').
                # For this subtask, focusing on training pipeline, we can simplify or comment out if errors occur.

                try:
                    latest_features_by_sector = X_test_demo.groupby('sector_label').last().reset_index()
                    # Attempt to dynamically map features for emergence score if possible, or use known suffixes
                    # This is a placeholder, actual mapping might be complex if original names are not easily reconstructible
                    emergence_feature_map_auto = {
                        col: col.replace('_patent','').replace('_funding','').replace('_research','')
                        for col in latest_features_by_sector.columns if '_patent' in col or '_funding' in col or '_research' in col
                    }
                    # This mapping is very basic and might not match `prediction_config['emergence_indicators']` structure.
                    # For a robust solution, `identify_emerging_technologies` might need to be aware of suffixed names
                    # or expect a pre-mapping to a canonical form.

                    # If `prediction_config['emergence_indicators']` expects specific names like 'filing_rate_3m',
                    # and `latest_features_by_sector` has 'filing_rate_3m_patent', a rename is needed.
                    # Example: if indicators expect 'filing_rate_3m', and X_test_demo has 'filing_rate_3m_patent',
                    # 'filing_rate_3m_funding', etc. then `identify_emerging_technologies` needs adjustment or
                    # a specific mapping.
                    # For now, let's assume `identify_emerging_technologies` might need an update or this part might be skipped.
                    # To make it runnable, let's try to pass it as is, but this is a known potential issue.

                    # print("\nAttempting to identify emerging technologies (may require feature name alignment)...") # Original print
                    emerging_techs = pred_gen.identify_emerging_technologies(
                        all_features_df=latest_features_by_sector, # Pass as is, let the function handle it or error
                        sector_col='sector_label',
                        indicators_config=prediction_config['emergence_indicators'],
                        threshold_percentile=50
                    )
                    # print(f"\nEmerging Technologies Identified (sample, {len(emerging_techs)}):") # Detailed print
                    # for tech_item in emerging_techs[:1]: print(f"  Tech: {tech_item['technology']}, Score: {tech_item['emergence_score']:.2f}")
                    print("Emerging technologies identified.")

                    mock_market_data = {sector: {'market_size_usd': np.random.randint(50, 200) * 1e9} for sector in cli_sectors} # Dynamic mock market data
                    investment_ops = pred_gen.create_investment_opportunities(sector_forecasts, emerging_techs, mock_market_data, prediction_config['investment_ranking_criteria'])
                    # print(f"\nInvestment Opportunities (Top 1 sample, {len(investment_ops)} total):") # Detailed print
                    # for op_item in investment_ops[:1]: print(f"  Sector/Tech: {op_item['sector']}, Action: {op_item['recommended_action']}, Score: {op_item['attractiveness_score']:.2f}")
                    print("Investment opportunities created.")

                except Exception as e:
                    print(f"Error in post-training prediction generation (emerging tech/investment ops): {e}")
                    print("This may be due to feature name changes from alignment. Further refactoring of this section may be needed.")

            # --- Results Summary ---
            print("\n--- Results Summary ---")
            if emerging_techs:
                print("\nTop Emerging Technologies:")
                for i, tech in enumerate(emerging_techs[:3]): # Print top 3
                    print(f"  {i+1}. {tech['technology']} (Score: {tech['emergence_score']:.2f})")
            else:
                print("\nNo emerging technologies identified based on current criteria.")

            if sector_forecasts:
                print("\nSector Forecasts (Growth Rate - Sample Horizon):")
                summary_horizon_key_template = "{}m_growth_rate"
                summary_horizon_to_check = cli_horizons[0] # Default to first requested horizon
                if 12 in cli_horizons: summary_horizon_to_check = 12 # Prefer 12m if requested

                summary_horizon_key = summary_horizon_key_template.format(summary_horizon_to_check)

                for sector, forecast_data in sector_forecasts.items():
                    if summary_horizon_key in forecast_data:
                        details = forecast_data[summary_horizon_key]
                        print(f"  - {sector} ({summary_horizon_key.split('_')[0]}): {details['prediction']:.3f} (Quality: {details['quality_score']:.2f})")
                    elif forecast_data: # Print first available if preferred not found
                        first_avail_key = next(iter(forecast_data))
                        details = forecast_data[first_avail_key]
                        # Ensure key is in expected format before splitting
                        horizon_val_str = first_avail_key.split('m_')[0] if 'm_' in first_avail_key else first_avail_key
                        print(f"  - {sector} ({horizon_val_str}): {details['prediction']:.3f} (Quality: {details['quality_score']:.2f})")
            else:
                print("\nNo sector forecasts were generated.")

            if investment_ops:
                print("\nTop Investment Opportunity:")
                top_op = investment_ops[0] # Already sorted by attractiveness_score
                print(f"  - Sector/Tech: {top_op['sector']}")
                print(f"    Type: {top_op['type']}")
                print(f"    Recommendation: {top_op['recommended_action']} (Score: {top_op['attractiveness_score']:.2f})")
            else:
                print("\nNo investment opportunities identified.")

    # --- Monitoring and Maintenance (Conceptual) ---
    # ... (rest of the script, potentially needs similar updates for feature names) ...
    # For now, the focus is on the training data pipeline.
    # The monitoring and uncertainty sections would also need X_test_demo with new feature names.
    print("\n--- Monitoring & Maintenance (Conceptual) ---") # Moved this print to be after summary
    # Ensure trained_models exists and is not empty. If it was skipped, monitor can't use it.
    if 'trained_models' in locals() and trained_models and any(v for v in trained_models.values()):
        monitor = SystemMonitor(trained_models, model_config, monitoring_config)

        # Baseline for data drift: Use a subset of X_train_demo (numeric features only, no sector label)
        # This requires features to be in a format that monitor.set_data_drift_baseline expects.
        # If `monitoring_config['data_drift_thresholds']` uses specific names (e.g. 'filing_rate_3m_patent'),
        # then X_train_demo should already have these.
        baseline_features_df = X_train_demo.drop(columns=['sector_label']).select_dtypes(include=np.number)
        if not baseline_features_df.empty:
            monitor.set_data_drift_baseline(baseline_features_df)
            print("Data drift baseline set using a sample of training data.")

            # Drift check: Use a subset of X_test_demo
            # Ensure column names align with what `monitor_data_drift` expects (i.e., names in baseline)
            drift_check_features_df = X_test_demo.drop(columns=['sector_label']).select_dtypes(include=np.number).tail(10)
            if not drift_check_features_df.empty:
                # Example: Check for AI sector. Assumes X_test_demo has 'sector_label'
                current_ai_features_for_drift = X_test_demo[X_test_demo['sector_label']=='AI'].drop(columns=['sector_label']).select_dtypes(include=np.number).tail(5)
                if not current_ai_features_for_drift.empty:
                    drift_detected, _ = monitor.monitor_data_drift(current_ai_features_for_drift)
                    print(f"Data drift detected for AI sector (sample check on X_test_demo): {drift_detected}")
        else:
            print("Baseline features DataFrame for drift monitoring is empty.")

        # Model performance evaluation on recent data
        # Example: Evaluate AI model on its test data slice
        X_recent_ai_eval = X_test_demo[X_test_demo['sector_label']=='AI'] # Keep sector_label if evaluate_model... expects it
        y_recent_ai_eval = y_test_demo[y_test_demo.index.isin(X_recent_ai_eval.index)] # Align y

        # The evaluate_model_performance_on_recent_data in monitor.py expects X_recent to be numeric and without sector label
        # It also gets model directly using sector name.
        # This loop needs to be dynamic based on cli_sectors
        for sector_to_evaluate in cli_sectors:
            if sector_to_evaluate in X_test_demo['sector_label'].unique():
                X_recent_sector_eval = X_test_demo[X_test_demo['sector_label'] == sector_to_evaluate]
                y_recent_sector_eval = y_test_demo[y_test_demo.index.isin(X_recent_sector_eval.index)]

                if not X_recent_sector_eval.empty and not y_recent_sector_eval.empty:
                    performance_ok = monitor.evaluate_model_performance_on_recent_data(
                        sector_to_evaluate,
                        X_recent_sector_eval.drop(columns=['sector_label']).select_dtypes(include=np.number),
                        y_recent_sector_eval
                    )
                    print(f"Model performance for {sector_to_evaluate} sector on recent data (X_test_demo) is OK: {performance_ok}")
                else:
                    print(f"No data for {sector_to_evaluate} sector in X_test_demo to evaluate performance.")
            else:
                print(f"Sector {sector_to_evaluate} not found in test data for performance evaluation.")
    else:
        print("No trained models available for monitoring setup.")


    # --- Uncertainty Handling (Conceptual) ---
    print("\n--- Uncertainty Handling (Conceptual) ---")
    uncertainty_mgr = UncertaintyManager(uncertainty_config)
    # Make uncertainty handling example dynamic with first CLI sector
    if cli_sectors:
        example_sector_for_uncertainty = cli_sectors[0]
        adj1, msg1 = uncertainty_mgr.assess_data_completeness_impact("USPTO", 0.75, example_sector_for_uncertainty)
        print(msg1 if msg1 else f"Completeness OK for USPTO in {example_sector_for_uncertainty}")
        adj2, msg2, _ = uncertainty_mgr.handle_conflicting_signals(example_sector_for_uncertainty, {'pat':0.7, 'fund':-0.6})
        print(msg2)

        if 'sector_forecasts' in locals() and sector_forecasts and example_sector_for_uncertainty in sector_forecasts:
            sector_forecast_sample = next(iter(sector_forecasts.get(example_sector_for_uncertainty, {}).values()), None)
            if sector_forecast_sample:
                final_out = uncertainty_mgr.format_final_prediction_output(
                    sector_forecast_sample['prediction'],
                    sector_forecast_sample['quality_score'],
                    [adj1, adj2],
                    sector_forecast_sample.get('confidence_intervals')
                )
                print(f"\nFinal Prediction Output with Uncertainty ({example_sector_for_uncertainty} Sample):"); [print(f"  {k}: {v}") for k,v in final_out.items()]
            else: print(f"\nNo {example_sector_for_uncertainty} forecast details for uncertainty demo.")
        else: print(f"\nNo sector_forecasts available for {example_sector_for_uncertainty} for uncertainty demo.")
    else:
        print("\nNo CLI sectors provided, skipping uncertainty handling demo.")

    print("\n--- System Run Finished (Conceptual) ---")
