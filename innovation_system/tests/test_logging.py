import unittest
import os
import subprocess
import re
import shutil

# Define the project root and log file path
# Assuming this test script is in /app/innovation_system/tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "system_monitor.log")
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "innovation_system", "main", "run.py")
BACKUP_LOG_FILE_PATH = LOG_FILE_PATH + ".bak"

class TestLogging(unittest.TestCase):

    def setUp(self):
        """
        Back up and remove the existing log file before each test.
        """
        if os.path.exists(LOG_FILE_PATH):
            shutil.copy(LOG_FILE_PATH, BACKUP_LOG_FILE_PATH)
            os.remove(LOG_FILE_PATH)

    def tearDown(self):
        """
        Clean up: remove the log file created by the test and restore the backup.
        """
        if os.path.exists(LOG_FILE_PATH):
            os.remove(LOG_FILE_PATH)
        if os.path.exists(BACKUP_LOG_FILE_PATH):
            shutil.move(BACKUP_LOG_FILE_PATH, LOG_FILE_PATH)

    def test_log_file_creation_and_content(self):
        """
        Tests if the log file is created by run.py and contains expected content.
        """
        # Ensure the script path is correct
        self.assertTrue(os.path.exists(SCRIPT_PATH), f"Script not found at {SCRIPT_PATH}")

        # Set PYTHONPATH for the subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = PROJECT_ROOT

        # Execute the main script
        process = subprocess.run(
            ["python", SCRIPT_PATH],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT  # Ensure script runs from project root
        )

        # Assert that the log file was created
        self.assertTrue(os.path.exists(LOG_FILE_PATH), "Log file was not created.")

        # Assert that the log file is not empty
        log_file_size = os.path.getsize(LOG_FILE_PATH)
        self.assertTrue(log_file_size > 0, "Log file is empty.")

        # Read the log file content
        log_content = []
        if log_file_size > 0:
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                log_content = f.readlines()

        # Optional: Check the format of some log lines
        # Full corrected regex pattern
        log_pattern = re.compile(
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (INFO|WARNING|ERROR|DEBUG|CRITICAL) \[[^\]]+:\d+\] - .+$"
        )

        matched_lines = 0
        unmatched_lines_sample = []
        for i, line in enumerate(log_content):
            line_stripped = line.strip()
            if log_pattern.match(line_stripped):
                matched_lines += 1
            elif len(unmatched_lines_sample) < 5: # Collect a few samples of non-matching lines
                unmatched_lines_sample.append(line_stripped) # Keep sample collection for potential future fails

            if matched_lines >= 2: # Check if we found at least 2 matching lines
                break

        self.assertTrue(matched_lines >= 2,
            f"Could not find at least 2 log lines matching the expected format. Found {matched_lines}.\n"
            f"Process exit code: {process.returncode}. Stderr: {process.stderr}\n"
            f"Sample of non-matching lines from log: {unmatched_lines_sample[:5]}" # Show only up to 5 samples
        )

if __name__ == "__main__":
    unittest.main()
