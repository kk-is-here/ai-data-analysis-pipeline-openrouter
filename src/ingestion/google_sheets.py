import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_FILE = Path("data/survey_data.csv")


def extract_sheet_id(sheet_input):
    """
    Accept either:
    1. A Google Sheet ID
    2. A full Google Sheet URL

    Returns the Google Sheet ID.
    """

    sheet_input = sheet_input.strip()

    # If a full Google Sheets URL is provided
    match = re.search(
        r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)",
        sheet_input
    )

    if match:
        return match.group(1)

    # Otherwise assume the input itself is the Sheet ID
    if re.fullmatch(r"[a-zA-Z0-9-_]+", sheet_input):
        return sheet_input

    raise ValueError(
        "Invalid Google Sheet input. Provide a Google Sheet URL or Sheet ID."
    )


def load_google_sheet(sheet_input):
    """
    Load a publicly accessible Google Sheet as a pandas DataFrame.
    """

    sheet_id = extract_sheet_id(sheet_input)

    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{sheet_id}/export?format=csv"
    )

    print("Loading Google Sheet...")
    print(f"Sheet ID: {sheet_id}")

    try:
        df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(
            "Could not load the Google Sheet. "
            "Make sure the sheet is publicly accessible or published "
            "and that the provided URL/Sheet ID is correct."
        ) from e

    return df


def save_data(df, output_file=DEFAULT_OUTPUT_FILE):
    """
    Save the Google Sheet data as CSV.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)

    print(f"Data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Load a Google Sheet and save it as CSV."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Google Sheet URL or Google Sheet ID"
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output CSV file path"
    )

    args = parser.parse_args()

    try:
        df = load_google_sheet(args.input)

        print("Google Sheet loaded successfully.")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        save_data(df, args.output)

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()