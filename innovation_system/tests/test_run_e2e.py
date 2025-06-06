import pytest
import subprocess
import os
import tempfile
from datetime import datetime, timedelta
import sqlite3
import pandas as pd # To check parquet files

# Define the path to the run.py script relative to the tests directory or project root
# Assuming tests are in innovation_system/tests and run.py is in innovation_system/main
PATH_TO_RUN_PY = os.path.join(os.path.dirname(__file__), "..", "main", "run.py")


def check_parquet_file(file_path, min_rows=1, allow_empty=False): # Added allow_empty
    assert os.path.exists(file_path), f"Parquet file {file_path} was not created."
    if allow_empty and not os.path.getsize(file_path) > 0 : # Check if file has content before reading if allowed empty
        print(f"Warning: Parquet file {file_path} is empty (as allowed for this check).")
        return None # Or an empty DataFrame: pd.DataFrame()

    df = pd.read_parquet(file_path)
    if not allow_empty:
        assert not df.empty, f"Parquet file {file_path} is empty."
        assert len(df) >= min_rows, f"Parquet file {file_path} has {len(df)} rows, expected at least {min_rows}."
    elif not df.empty: # If allow_empty is true, but file is not empty, still check min_rows
         assert len(df) >= min_rows, f"Parquet file {file_path} has {len(df)} rows, expected at least {min_rows} (when not empty)."
    return df


def check_sqlite_db(db_path):
    assert os.path.exists(db_path), f"SQLite DB {db_path} was not created."
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Check if pipeline_status table exists and has entries for main_run
        cursor.execute("SELECT COUNT(*) FROM pipeline_status WHERE pipeline_name = 'main_run'")
        count = cursor.fetchone()[0]
        assert count > 0, "No 'main_run' entries found in pipeline_status table."

        cursor.execute("SELECT status FROM pipeline_status WHERE pipeline_name = 'main_run' ORDER BY last_run_timestamp DESC LIMIT 1")
        final_status_row = cursor.fetchone()
        assert final_status_row is not None, "Could not fetch final status for 'main_run'."
        final_status = final_status_row[0]
        assert final_status == "COMPLETED", f"Final status for 'main_run' was '{final_status}', expected 'COMPLETED'."

    finally:
        conn.close()


@pytest.mark.e2e
def test_e2e_basic_run_with_force_collect():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_raw_path = os.path.join(tmpdir, "data", "raw")
        # run.py creates data/raw inside its CWD if it doesn't exist.
        # os.makedirs(data_raw_path, exist_ok=True) # Not needed if run.py handles it

        monitoring_db_file = os.path.join(tmpdir, "data", "monitoring.sqlite") # As per run.py default
        patents_file = os.path.join(data_raw_path, "patents.parquet")
        funding_file = os.path.join(data_raw_path, "funding.parquet")
        research_file = os.path.join(data_raw_path, "research_papers.parquet")
        main_log_file = os.path.join(tmpdir, "innovation_main.log")

        sectors = "TestSectorA,TestSectorB"
        # Adjust start_date for a longer period to ensure data generation
        start_date = (datetime.now() - timedelta(days=70)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        horizons = "6,12"

        command = [
            "python", PATH_TO_RUN_PY,
            "--sectors", sectors,
            "--start-date", start_date,
            "--end-date", end_date,
            "--horizons", horizons,
            "--force-collect"
            # Add --config-file if a specific test config is needed
            # Add --log-level DEBUG for more verbose output if helpful
        ]

        process = subprocess.Popen(command, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(f"run.py script timed out after 300s. STDOUT:\n{stdout}\nSTDERR:\n{stderr}")


        print("--- run.py STDOUT ---")
        print(stdout)
        print("--- run.py STDERR ---")
        print(stderr)

        assert process.returncode == 0, f"run.py script failed with error code {process.returncode}. STDERR:\n{stderr}"

        check_parquet_file(patents_file, min_rows=1)
        check_parquet_file(funding_file, min_rows=1)
        # research_file can be empty if live arXiv call fails and mock fallback is also minimal/empty
        check_parquet_file(research_file, min_rows=0, allow_empty=True)

        check_sqlite_db(monitoring_db_file)

        assert "--- Innovation Prediction System ---" in stdout
        assert "Generating and Saving Source Data" in stdout

        # Check if training was attempted or skipped. With 70 days, it should generate enough data.
        # The mock data in run.py uses freq='MS', so 70 days = ~2-3 data points.
        # This might still be too few for some operations (e.g. if lags are long).
        # Predictor's prepare_training_data drops NaNs after lagging (max lag 12 months default).
        # So, if we only have 2-3 months of data, X_prepared might become empty.

        # Let's look for specific logs that indicate data presence or absence for training.
        if "Training data is empty after splitting" in stdout or \
           "X_prepared is empty after processing" in stdout or \
           "No data available for training after processing features and targets." in stdout or \
           "Skipping model training, validation, and prediction generation due to empty data." in stdout: # Added this check from run.py
            print("Warning: Model training was skipped due to empty data, E2E test assertions adjusted.")
            assert "Skipping model training, validation, and prediction generation due to empty data." in stdout
        else:
            assert "Starting Model Training & Validation" in stdout
            # These messages might only appear if training actually proceeds for a sector
            # assert "Sector models trained." in stdout # This is logged by predictor.train_sector_models if models are trained
            assert "Starting Prediction Generation" in stdout
            assert "Results Summary" in stdout # This is logged by run.py after predictions

        assert "System Run Finished" in stdout
        assert os.path.exists(main_log_file), f"Main log file {main_log_file} not found."


@pytest.mark.e2e
def test_e2e_run_loading_from_existing_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_raw_path = os.path.join(tmpdir, "data", "raw")
        # monitoring_db_path_dir = os.path.join(tmpdir, "data") # Not directly used, db file path is sufficient
        monitoring_db_file = os.path.join(tmpdir, "data", "monitoring.sqlite")
        # Other file paths for checking can be defined if needed

        # --- Step 1: Generate data files first using --force-collect ---
        sectors_gen = "GenSector"
        start_date_gen = (datetime.now() - timedelta(days=70)).strftime('%Y-%m-%d')
        end_date_gen = datetime.now().strftime('%Y-%m-%d')
        horizons_gen = "6"

        gen_command = [
            "python", PATH_TO_RUN_PY,
            "--sectors", sectors_gen,
            "--start-date", start_date_gen,
            "--end-date", end_date_gen,
            "--horizons", horizons_gen,
            "--force-collect"
        ]

        process_gen = subprocess.Popen(gen_command, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout_gen, stderr_gen = process_gen.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            process_gen.kill()
            stdout_gen, stderr_gen = process_gen.communicate()
            pytest.fail(f"Data generation step for E2E test timed out. STDERR:\n{stderr_gen}")


        assert process_gen.returncode == 0, f"Data generation step failed for E2E test. STDERR:\n{stderr_gen}"
        print("--- Data Generation STDOUT (for test_e2e_run_loading_from_existing_files) ---")
        print(stdout_gen)
        assert "Generating and Saving Source Data" in stdout_gen
        assert os.path.exists(os.path.join(data_raw_path, "patents.parquet")), "Patents.parquet not generated in pre-step."


        # --- Step 2: Run the script again, this time WITHOUT --force-collect ---
        sectors_load = "LoadSector"
        start_date_load = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        end_date_load = datetime.now().strftime('%Y-%m-%d')
        horizons_load = "12,24"

        load_command = [
            "python", PATH_TO_RUN_PY,
            "--sectors", sectors_load,
            "--start-date", start_date_load,
            "--end-date", end_date_load,
            "--horizons", horizons_load
        ]

        process_load = subprocess.Popen(load_command, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout_load, stderr_load = process_load.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            process_load.kill()
            stdout_load, stderr_load = process_load.communicate()
            pytest.fail(f"run.py script (loading) timed out. STDERR:\n{stderr_load}")


        print("--- run.py STDOUT (Loading from files) ---")
        print(stdout_load)
        print("--- run.py STDERR (Loading from files) ---")
        print(stderr_load)

        assert process_load.returncode == 0, f"run.py script (loading) failed. STDERR:\n{stderr_load}"

        assert "--- Loading Source Data from Parquet Files ---" in stdout_load
        assert "Generating and Saving Source Data" not in stdout_load

        assert "Generating Features for Demo Model Training" in stdout_load

        if "Training data is empty after splitting" in stdout_load or \
           "X_prepared is empty after processing" in stdout_load or \
           "No data available for training after processing features and targets." in stdout_load or \
           "Skipping model training, validation, and prediction generation due to empty data." in stdout_load:
            print("Warning: Model training was skipped during 'load' run due to data, E2E test assertions adjusted.")
            assert "Skipping model training, validation, and prediction generation due to empty data." in stdout_load
        else:
            assert "Starting Model Training & Validation" in stdout_load
            assert "Starting Prediction Generation" in stdout_load

        assert "System Run Finished" in stdout_load

        check_sqlite_db(monitoring_db_file)

```
