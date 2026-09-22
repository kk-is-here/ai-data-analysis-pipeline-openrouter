import argparse
import json
from pathlib import Path

import pandas as pd

from src.profiling.extract_metadata import (
    load_data,
    extract_metadata,
    save_metadata,
)

from src.profiling.ai_profiler import (
    profile_dataset,
)

from src.analysis.statistical_engine import (
    run_all_tests,
)

from src.visualization.visualization_engine import (
    generate_visualizations,
)

from src.interpretation.interpretation_engine import (
    generate_all_interpretations,
)

from src.reporting.report_generator import (
    generate_html_report,
    save_report,
)

from src.ingestion.google_sheets import (
    load_google_sheet,
    save_data,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("output")

METADATA_FILE = OUTPUT_DIR / "metadata.json"
DICTIONARY_FILE = OUTPUT_DIR / "data_dictionary.json"
STATISTICS_FILE = OUTPUT_DIR / "statistical_results.json"
INTERPRETATIONS_FILE = OUTPUT_DIR / "interpretations.json"
VISUALIZATIONS_FILE = OUTPUT_DIR / "visualizations.json"
REPORT_FILE = OUTPUT_DIR / "analysis_report.html"

GOOGLE_SHEET_OUTPUT = Path("data/google_sheet_data.csv")


# ============================================================
# JSON HELPERS
# ============================================================

def save_json(data, file_path):
    """Save Python data to JSON."""

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str
        )


# ============================================================
# GOOGLE SHEET DETECTION
# ============================================================

def is_google_sheet_input(input_value):
    """
    Determine whether the supplied input is a Google Sheet URL
    or Google Sheet ID.
    """

    input_value = str(input_value).strip()

    if "docs.google.com/spreadsheets" in input_value:
        return True

    if (
        len(input_value) >= 30
        and "/" not in input_value
        and "\\" not in input_value
        and "." not in input_value
    ):
        return True

    return False


# ============================================================
# PREPARE INPUT DATASET
# ============================================================

def prepare_input(input_value):
    """
    Prepare the user input for the pipeline.

    Local CSV/XLSX/XLS files are returned unchanged.

    Google Sheet URLs/IDs are downloaded and saved as CSV.
    """

    input_value = str(input_value).strip()

    # Google Sheet
    if is_google_sheet_input(input_value):

        print()
        print("Google Sheet detected.")
        print("Downloading Google Sheet data...")

        data = load_google_sheet(input_value)

        if data.empty:
            raise ValueError(
                "The Google Sheet contains no rows."
            )

        save_data(
            data,
            GOOGLE_SHEET_OUTPUT
        )

        print(
            f"Google Sheet converted to: "
            f"{GOOGLE_SHEET_OUTPUT}"
        )

        return GOOGLE_SHEET_OUTPUT

    # Local file
    input_path = Path(input_value)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset was not found: {input_path}"
        )

    if not input_path.is_file():
        raise ValueError(
            f"Input path is not a file: {input_path}"
        )

    supported_extensions = {
        ".csv",
        ".xlsx",
        ".xls"
    }

    if input_path.suffix.lower() not in supported_extensions:
        raise ValueError(
            "Unsupported file format. "
            "Supported formats are: CSV, XLSX and XLS."
        )

    return input_path


# ============================================================
# DATASET INFORMATION
# ============================================================

def create_dataset_info(data, input_path):
    """Create dataset information for the report."""

    return {
        "file_name": input_path.name,
        "file_path": str(input_path),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "column_names": [
            str(column)
            for column in data.columns
        ]
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(input_value):

    input_path = prepare_input(input_value)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("AI DATA ANALYSIS PIPELINE")
    print("=" * 70)

    print()
    print(f"Input dataset: {input_value}")
    print(f"Working dataset: {input_path}")

    # ========================================================
    # STEP 1 — LOAD DATA
    # ========================================================

    print()
    print("[1/7] Loading dataset...")

    data = load_data(input_path)

    if data.empty:
        raise ValueError(
            "The dataset contains no rows."
        )

    print(
        f"Loaded successfully: "
        f"{data.shape[0]} rows × "
        f"{data.shape[1]} columns"
    )

    dataset_info = create_dataset_info(
        data,
        input_path
    )

    # ========================================================
    # STEP 2 — METADATA EXTRACTION
    # ========================================================

    print()
    print("[2/7] Extracting metadata...")

    metadata = extract_metadata(input_path)

    save_metadata(
        metadata,
        METADATA_FILE
    )

    print(
        f"Metadata saved to: "
        f"{METADATA_FILE}"
    )

    # ========================================================
    # STEP 3 — AI DATA PROFILING
    # ========================================================

    print()
    print("[3/7] Running AI data profiler...")

    data_dictionary = profile_dataset(
        metadata_file=METADATA_FILE,
        output_file=DICTIONARY_FILE
    )

    print(
        f"AI data dictionary created: "
        f"{len(data_dictionary)} variables"
    )

    # ========================================================
    # STEP 4 — STATISTICAL ANALYSIS
    # ========================================================

    print()
    print("[4/7] Running statistical analysis...")

    statistical_results = run_all_tests(
        data,
        data_dictionary
    )

    save_json(
        statistical_results,
        STATISTICS_FILE
    )

    print(
        f"Statistical analyses generated: "
        f"{len(statistical_results)}"
    )

    # ========================================================
    # STEP 5 — VISUALIZATION
    # ========================================================

    print()
    print("[5/7] Generating visualizations...")

    visualizations = generate_visualizations(
        data,
        data_dictionary
    )

    save_json(
        visualizations,
        VISUALIZATIONS_FILE
    )

    print(
        f"Visualizations generated: "
        f"{len(visualizations)}"
    )

    # ========================================================
    # STEP 6 — INTERPRETATION
    # ========================================================

    print()
    print("[6/7] Generating statistical interpretations...")

    interpretations = generate_all_interpretations(
        statistical_results
    )

    save_json(
        interpretations,
        INTERPRETATIONS_FILE
    )

    print(
        f"Interpretations generated: "
        f"{len(interpretations)}"
    )

    # ========================================================
    # STEP 7 — REPORT
    # ========================================================

    print()
    print("[7/7] Generating HTML report...")

    html = generate_html_report(
        dataset_info,
        data_dictionary,
        statistical_results,
        interpretations,
        visualizations
    )

    save_report(html)

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)

    print()
    print(f"Dataset        : {input_value}")
    print(f"Working file   : {input_path}")
    print(
        f"Rows × Columns : "
        f"{data.shape[0]} × {data.shape[1]}"
    )
    print(
        f"Variables      : "
        f"{len(data_dictionary)}"
    )
    print(
        f"Analyses       : "
        f"{len(statistical_results)}"
    )
    print(
        f"Interpretations: "
        f"{len(interpretations)}"
    )
    print(
        f"Visualizations : "
        f"{len(visualizations)}"
    )
    print(
        f"Report         : "
        f"{REPORT_FILE}"
    )

    print()
    print("All pipeline steps completed successfully.")

    return {
        "dataset": dataset_info,
        "data_dictionary": data_dictionary,
        "statistical_results": statistical_results,
        "interpretations": interpretations,
        "visualizations": visualizations,
        "report": str(REPORT_FILE)
    }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AI-powered automated data analysis pipeline "
            "for CSV, Excel and Google Sheets datasets."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to CSV/XLSX/XLS file, "
            "Google Sheet URL, or Google Sheet ID."
        )
    )

    args = parser.parse_args()

    try:

        run_pipeline(args.input)

    except Exception as error:

        print()
        print("=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)

        print()
        print(f"Error: {error}")

        print()

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()