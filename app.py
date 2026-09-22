import streamlit as st
from pathlib import Path
import subprocess
import sys
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Analysis Pipeline",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

REPORT_FILE = OUTPUT_DIR / "analysis_report.html"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📊 AI Data Analysis Pipeline")

st.markdown(
    """
    Upload a dataset or provide a Google Sheet URL and let the
    automated pipeline perform data profiling, statistical analysis,
    visualization, interpretation, and report generation.
    """
)


# ============================================================
# INPUT SOURCE
# ============================================================

st.subheader("1. Select Data Source")

input_type = st.radio(
    "Choose your data source:",
    [
        "Upload CSV / Excel",
        "Google Sheet"
    ],
    horizontal=True
)


# ============================================================
# FILE UPLOAD
# ============================================================

input_value = None

if input_type == "Upload CSV / Excel":

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=[
            "csv",
            "xlsx",
            "xls"
        ]
    )

    if uploaded_file is not None:

        uploaded_path = DATA_DIR / uploaded_file.name

        with open(
            uploaded_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        input_value = uploaded_path

        st.success(
            f"Dataset selected: {uploaded_file.name}"
        )


# ============================================================
# GOOGLE SHEET
# ============================================================

else:

    google_sheet_url = st.text_input(
        "Google Sheet URL",
        placeholder=(
            "https://docs.google.com/spreadsheets/d/..."
        )
    )

    if google_sheet_url.strip():

        input_value = google_sheet_url.strip()

        st.info(
            "Make sure the Google Sheet is publicly accessible "
            "or published to the web."
        )


# ============================================================
# RUN PIPELINE
# ============================================================

st.subheader("2. Run Analysis")

run_button = st.button(
    "🚀 Run Analysis",
    type="primary",
    use_container_width=True
)


if run_button:

    if input_value is None:

        st.warning(
            "Please select a dataset or enter a Google Sheet URL."
        )

        st.stop()

    # --------------------------------------------------------
    # Clear previous report
    # --------------------------------------------------------

    if REPORT_FILE.exists():

        try:
            REPORT_FILE.unlink()
        except Exception:
            pass

    # --------------------------------------------------------
    # Progress display
    # --------------------------------------------------------

    progress = st.progress(0)

    status = st.empty()

    status.info(
        "Starting AI Data Analysis Pipeline..."
    )

    progress.progress(10)

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    command = [
        sys.executable,
        str(BASE_DIR / "run_pipeline.py"),
        "--input",
        str(input_value)
    ]

    try:

        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        progress.progress(100)

        # ----------------------------------------------------
        # Pipeline success
        # ----------------------------------------------------

        if result.returncode == 0:

            status.success(
                "Analysis completed successfully!"
            )

            st.success(
                "Your dataset has been analyzed successfully."
            )

            # ------------------------------------------------
            # Pipeline details
            # ------------------------------------------------

            with st.expander(
                "View Pipeline Details"
            ):

                st.code(
                    result.stdout,
                    language="text"
                )

            # ------------------------------------------------
            # DISPLAY REPORT
            # ------------------------------------------------

            if REPORT_FILE.exists():

                st.subheader(
                    "3. Analysis Report"
                )

                st.success(
                    "Analysis dashboard generated successfully."
                )

                # --------------------------------------------
                # Read complete HTML report
                # --------------------------------------------

                with open(
                    REPORT_FILE,
                    "r",
                    encoding="utf-8"
                ) as report:

                    html_report = report.read()

                # --------------------------------------------
                # Display HTML dashboard inside Streamlit
                # --------------------------------------------

                components.html(
                    html_report,
                    height=900,
                    scrolling=True
                )

                # --------------------------------------------
                # Download report
                # --------------------------------------------

                st.download_button(
                    label="⬇️ Download HTML Report",
                    data=html_report,
                    file_name="analysis_report.html",
                    mime="text/html",
                    use_container_width=True
                )

            else:

                st.warning(
                    "Pipeline completed, but the HTML report "
                    "was not found."
                )

        # ----------------------------------------------------
        # Pipeline failure
        # ----------------------------------------------------

        else:

            status.error(
                "Pipeline failed."
            )

            st.error(
                "The analysis pipeline encountered an error."
            )

            with st.expander(
                "View Error Details",
                expanded=True
            ):

                st.code(
                    result.stdout
                    + "\n"
                    + result.stderr,
                    language="text"
                )

    except Exception as error:

        progress.progress(100)

        status.error(
            "Could not start the analysis pipeline."
        )

        st.exception(
            error
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Data Analysis Pipeline • Automated statistical analysis "
    "and visualization"
)