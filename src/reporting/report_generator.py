import argparse
import json
import base64
import mimetypes
from pathlib import Path
from datetime import datetime

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATA_FILE = Path(
    "data/test_dataset.xlsx"
)

DEFAULT_DICTIONARY_FILE = Path(
    "output/data_dictionary.json"
)

DEFAULT_STATISTICS_FILE = Path(
    "output/statistical_results.json"
)

DEFAULT_INTERPRETATIONS_FILE = Path(
    "output/interpretations.json"
)

DEFAULT_VISUALIZATIONS_FILE = Path(
    "output/visualizations.json"
)

DEFAULT_OUTPUT_DIR = Path(
    "output"
)

DEFAULT_REPORT_FILE = (
    DEFAULT_OUTPUT_DIR / "analysis_report.html"
)


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(data_file=DEFAULT_DATA_FILE):
    """
    Load CSV or Excel dataset.

    The dataset path is supplied explicitly by the pipeline.
    """

    data_file = Path(data_file)

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}"
        )

    extension = data_file.suffix.lower()

    try:

        if extension == ".csv":
            data = pd.read_csv(data_file)

        elif extension in [".xlsx", ".xls"]:
            data = pd.read_excel(data_file)

        else:
            raise ValueError(
                f"Unsupported dataset format: {extension}"
            )

        return data

    except Exception as error:

        raise RuntimeError(
            f"Could not load dataset '{data_file}': {error}"
        ) from error


# ============================================================
# LOAD JSON
# ============================================================

def load_json(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {file_path}"
        )

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as error:

        raise RuntimeError(
            f"Could not read {file_path}: {error}"
        ) from error


# ============================================================
# LOAD DATASET INFORMATION
# ============================================================

def load_dataset_info(
    data_file=DEFAULT_DATA_FILE
):

    data_file = Path(data_file)

    data = load_dataset(
        data_file
    )

    return {

        "rows":
            data.shape[0],

        "columns":
            data.shape[1],

        "column_names":
            list(data.columns),

        "file_name":
            data_file.name,

        "file_path":
            str(data_file)
    }


# ============================================================
# HTML ESCAPING
# ============================================================

def escape_html(value):

    if value is None:
        return ""

    text = str(value)

    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# ============================================================
# GET VARIABLES FROM RESULT
# ============================================================

def get_result_variables(result):

    variables = result.get("variables")

    if isinstance(
        variables,
        list
    ) and variables:

        return variables

    variable1 = result.get(
        "variable1"
    )

    variable2 = result.get(
        "variable2"
    )

    result_variables = []

    if variable1:
        result_variables.append(
            variable1
        )

    if variable2:
        result_variables.append(
            variable2
        )

    if result_variables:
        return result_variables

    x = result.get("x")
    y = result.get("y")

    result_variables = []

    if x:
        result_variables.append(x)

    if y:
        result_variables.append(y)

    return result_variables


# ============================================================
# GET TEST NAME
# ============================================================

def get_test_name(result):

    return (
        result.get("test")
        or result.get("test_name")
        or result.get("method")
        or result.get("analysis")
        or ""
    )


# ============================================================
# GET STATISTIC VALUE
# ============================================================

def get_statistic_value(result):

    test_name = str(
        get_test_name(result)
    ).lower()

    # Spearman correlation
    if "spearman" in test_name:

        rho = result.get("rho")

        if rho is not None:

            return (
                f"rho = {float(rho):.4f}"
            )

        correlation = result.get(
            "correlation"
        )

        if correlation is not None:

            return (
                f"rho = {float(correlation):.4f}"
            )

    # Pearson correlation
    if "pearson" in test_name:

        r = result.get("r")

        if r is not None:

            return (
                f"r = {float(r):.4f}"
            )

        correlation = result.get(
            "correlation"
        )

        if correlation is not None:

            return (
                f"r = {float(correlation):.4f}"
            )

    # Chi-square
    if "chi" in test_name:

        chi2 = result.get("chi2")

        if chi2 is not None:

            return (
                f"χ² = {float(chi2):.4f}"
            )

        statistic = result.get(
            "statistic"
        )

        if statistic is not None:

            return (
                f"χ² = {float(statistic):.4f}"
            )

    # ANOVA
    if "anova" in test_name:

        statistic = result.get(
            "statistic"
        )

        if statistic is not None:

            return (
                f"F = {float(statistic):.4f}"
            )

        f_statistic = result.get(
            "f_statistic"
        )

        if f_statistic is not None:

            return (
                f"F = {float(f_statistic):.4f}"
            )

    # Kruskal-Wallis
    if "kruskal" in test_name:

        statistic = result.get(
            "statistic"
        )

        if statistic is not None:

            return (
                f"H = {float(statistic):.4f}"
            )

        h_statistic = result.get(
            "h_statistic"
        )

        if h_statistic is not None:

            return (
                f"H = {float(h_statistic):.4f}"
            )

    # Mann-Whitney
    if "mann" in test_name:

        statistic = result.get(
            "statistic"
        )

        if statistic is not None:

            return (
                f"U = {float(statistic):.4f}"
            )

    # t-test
    if (
        "t-test" in test_name
        or "t test" in test_name
    ):

        statistic = result.get(
            "statistic"
        )

        if statistic is not None:

            return (
                f"t = {float(statistic):.4f}"
            )

        t_statistic = result.get(
            "t_statistic"
        )

        if t_statistic is not None:

            return (
                f"t = {float(t_statistic):.4f}"
            )

    # Generic fallback
    statistic = result.get(
        "statistic"
    )

    if statistic is not None:

        try:

            return (
                f"{float(statistic):.4f}"
            )

        except (
            ValueError,
            TypeError
        ):

            return str(statistic)

    return "—"


# ============================================================
# GET P-VALUE
# ============================================================

def get_p_value(result):

    value = result.get(
        "p_value"
    )

    if value is None:

        value = result.get(
            "pvalue"
        )

    if value is None:

        value = result.get(
            "p"
        )

    if value is None:
        return "—"

    try:

        value = float(value)

        if value < 0.0001:
            return "< 0.0001"

        return f"{value:.4f}"

    except (
        ValueError,
        TypeError
    ):

        return str(value)


# ============================================================
# GET RAW P-VALUE
# ============================================================

def get_raw_p_value(result):

    value = result.get(
        "p_value"
    )

    if value is None:

        value = result.get(
            "pvalue"
        )

    if value is None:

        value = result.get(
            "p"
        )

    try:

        return float(value)

    except (
        ValueError,
        TypeError,
    ):

        return None


# ============================================================
# GET ADJUSTED P-VALUES
# ============================================================

def get_fdr_p_value(result):

    value = result.get(
        "p_value_fdr"
    )

    if value is None:
        value = result.get(
            "fdr_p_value"
        )

    try:
        return float(value)

    except (
        ValueError,
        TypeError
    ):
        return None


def get_bonferroni_p_value(result):

    value = result.get(
        "p_value_bonferroni"
    )

    if value is None:
        value = result.get(
            "bonferroni_p_value"
        )

    try:
        return float(value)

    except (
        ValueError,
        TypeError
    ):
        return None


def format_numeric_p_value(value):

    if value is None:
        return "—"

    if value < 0.0001:
        return "< 0.0001"

    return f"{value:.4f}"


# ============================================================
# GET SAMPLE SIZE
# ============================================================

def get_sample_size(result):

    value = (
        result.get("n")
        or result.get("sample_size")
        or result.get("N")
    )

    if value is None:
        return "—"

    return str(value)


# ============================================================
# GET INTERPRETATION TEXT
# ============================================================

def get_interpretation_text(item):

    """
    Interpretation engine output may be either:

    {
        "interpretation": "text"
    }

    or:

    {
        "interpretation": {
            "finding": "text"
        }
    }
    """

    possible_fields = [

        "interpretation",
        "interpretation_text",
        "message",
        "text",
        "description",
        "conclusion",
        "finding",
        "result"
    ]

    for field in possible_fields:

        value = item.get(field)

        if not value:
            continue

        if isinstance(
            value,
            dict
        ):

            finding = value.get(
                "finding"
            )

            if finding:
                return str(finding)

            text = value.get(
                "interpretation"
            )

            if text:
                return str(text)

            conclusion = value.get(
                "conclusion"
            )

            if conclusion:
                return str(conclusion)

            continue

        return str(value)

    return ""


# ============================================================
# GET WARNING TEXT
# ============================================================

def get_warning_text(item):

    possible_fields = [

        "warning",
        "warnings",
        "caution",
        "limitation",
        "limitations"
    ]

    warnings = []

    for field in possible_fields:

        value = item.get(field)

        if not value:
            continue

        if isinstance(
            value,
            list
        ):

            warnings.extend(
                str(x)
                for x in value
                if x
            )

        else:

            warnings.append(
                str(value)
            )

    # Interpretation engine may store warnings
    # inside the nested interpretation object.

    nested = item.get(
        "interpretation"
    )

    if isinstance(
        nested,
        dict
    ):

        nested_warnings = nested.get(
            "warnings"
        )

        if nested_warnings:

            if isinstance(
                nested_warnings,
                list
            ):

                warnings.extend(
                    str(x)
                    for x in nested_warnings
                    if x
                )

            else:

                warnings.append(
                    str(nested_warnings)
                )

    # Remove duplicates
    unique_warnings = []

    for warning in warnings:

        if warning not in unique_warnings:

            unique_warnings.append(
                warning
            )

    return " ".join(
        unique_warnings
    )


# ============================================================
# GET SIGNIFICANCE
# ============================================================

def get_significance(item):

    possible_fields = [

        "significance",
        "significance_label",
        "significance_status",
        "status"
    ]

    for field in possible_fields:

        value = item.get(field)

        if value:
            return str(value)

    nested = item.get(
        "interpretation"
    )

    if isinstance(
        nested,
        dict
    ):

        value = nested.get(
            "significance"
        )

        if value:
            return str(value)

    # Derive from p-value if necessary

    p_value = get_raw_p_value(
        item
    )

    if p_value is not None:

        if p_value < 0.05:

            return (
                "Statistically significant"
            )

        return (
            "Not statistically significant"
        )

    return ""


# ============================================================
# SIGNIFICANCE CLASS
# ============================================================

def get_significance_class(item):

    significance = (
        get_significance(item)
        .lower()
    )

    if (
        "significant" in significance
        and "not" not in significance
        and "non" not in significance
    ):

        return "significant"

    if (
        "not" in significance
        or "non" in significance
    ):

        return "not-significant"

    p_value = get_raw_p_value(
        item
    )

    if p_value is not None:

        if p_value < 0.05:
            return "significant"

        return "not-significant"

    return ""


# ============================================================
# VARIABLE TABLE
# ============================================================

def generate_variable_table(
    data_dictionary,
    dataset_info=None
):

    rows = []

    if not isinstance(
        data_dictionary,
        list
    ):
        data_dictionary = []

    dictionary_by_name = {
        item.get("column_name"): item
        for item in data_dictionary
        if isinstance(item, dict)
    }

    dataset_columns = []

    if isinstance(
        dataset_info,
        dict
    ):

        dataset_columns = dataset_info.get(
            "column_names",
            []
        ) or []

    # Use actual dataset column order when available.
    ordered_columns = (
        list(dataset_columns)
        if dataset_columns
        else list(dictionary_by_name.keys())
    )

    # IMPORTANT:
    # Do not add dictionary-only columns here.
    #
    # data_dictionary.json can contain variables from a previous
    # dataset/run. The current dataset metadata is the authoritative
    # source for which variables belong in this report.
    #
    # Therefore, when dataset_columns is available, only those
    # columns are displayed. Missing dictionary entries for current
    # columns are still shown as "Not classified" below.

    for name in ordered_columns:

        item = dictionary_by_name.get(
            name
        )

        if item is None:

            rows.append(
                f"""
                <tr>
                    <td>{escape_html(name)}</td>
                    <td>Not classified</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
                """
            )

            continue

        column_name = escape_html(
            item.get(
                "column_name",
                name
            )
        )

        semantic_type = escape_html(
            item.get(
                "semantic_type",
                ""
            )
        )

        measurement_scale = escape_html(
            item.get(
                "measurement_scale",
                ""
            )
        )

        subtype = escape_html(
            item.get(
                "semantic_subtype",
                ""
            )
        )

        confidence = item.get(
            "confidence",
            ""
        )

        if isinstance(
            confidence,
            float
        ):
            confidence = f"{confidence:.2f}"

        ordered = item.get(
            "ordered"
        )

        ordered_text = (
            "Yes"
            if ordered is True
            else "No"
            if ordered is False
            else "—"
        )

        rows.append(
            f"""
            <tr>
                <td>{column_name}</td>
                <td>{semantic_type}</td>
                <td>{measurement_scale}</td>
                <td>{subtype}</td>
                <td>{escape_html(confidence)}</td>
                <td>{ordered_text}</td>
            </tr>
            """
        )

    return "\n".join(rows)


# ============================================================
# STATISTICAL RESULTS TABLE
# ============================================================

def generate_statistics_table(
    statistical_results
):

    rows = []

    if not isinstance(
        statistical_results,
        list
    ):
        return ""

    for result in statistical_results:

        variables = get_result_variables(
            result
        )

        variable_text = (
            " × ".join(
                str(variable)
                for variable in variables
            )
            if variables
            else "—"
        )

        test = get_test_name(
            result
        )

        statistic = get_statistic_value(
            result
        )

        raw_p = get_raw_p_value(
            result
        )

        fdr_p = get_fdr_p_value(
            result
        )

        bonf_p = get_bonferroni_p_value(
            result
        )

        sample_size = get_sample_size(
            result
        )

        raw_class = (
            "significant"
            if raw_p is not None
            and raw_p < 0.05
            else "not-significant"
            if raw_p is not None
            else ""
        )

        fdr_class = (
            "significant"
            if fdr_p is not None
            and fdr_p < 0.05
            else "not-significant"
            if fdr_p is not None
            else ""
        )

        result_label = (
            "Raw p < 0.05"
            if raw_p is not None
            and raw_p < 0.05
            else "Not significant"
            if raw_p is not None
            else "Not available"
        )

        rows.append(
            f"""
            <tr>
                <td>{escape_html(variable_text)}</td>
                <td>{escape_html(test)}</td>
                <td>{escape_html(statistic)}</td>
                <td class="{raw_class}">
                    {escape_html(
                        format_numeric_p_value(raw_p)
                    )}
                </td>
                <td class="{fdr_class}">
                    {escape_html(
                        format_numeric_p_value(fdr_p)
                    )}
                </td>
                <td>
                    {escape_html(
                        format_numeric_p_value(bonf_p)
                    )}
                </td>
                <td>{escape_html(sample_size)}</td>
                <td>{escape_html(result_label)}</td>
            </tr>
            """
        )

    return "\n".join(rows)


# ============================================================
# INTERPRETATION CARDS
# ============================================================

def generate_interpretation_cards(
    interpretations
):

    cards = []

    if not isinstance(
        interpretations,
        list
    ):

        return ""

    for item in interpretations:

        variables = (
            get_result_variables(
                item
            )
        )

        if variables:

            variable_text = " × ".join(
                str(variable)
                for variable in variables
            )

        else:

            variable_text = (
                item.get(
                    "title",
                    "Statistical finding"
                )
            )

        interpretation = (
            get_interpretation_text(
                item
            )
        )

        warning = (
            get_warning_text(
                item
            )
        )

        significance = (
            get_significance(
                item
            )
        )

        significance_class = (
            get_significance_class(
                item
            )
        )

        test_name = escape_html(
            get_test_name(
                item
            )
        )

        card = f"""
        <div class="interpretation-card">

            <div class="interpretation-header">

                <div>

                    <h3>
                        {escape_html(
                            variable_text
                        )}
                    </h3>

                    <div class="small-muted">
                        {test_name}
                    </div>

                </div>

                {
                    f'<span class="result-badge {significance_class}">'
                    f'{escape_html(significance)}'
                    f'</span>'
                    if significance
                    else ''
                }

            </div>
        """

        if interpretation:

            card += f"""
            <p class="interpretation-text">
                {escape_html(
                    interpretation
                )}
            </p>
            """

        else:

            card += """
            <p class="interpretation-text">
                No interpretation text was returned
                for this analysis.
            </p>
            """

        if warning:

            card += f"""
            <div class="warning">
                <strong>Note:</strong>
                {escape_html(
                    warning
                )}
            </div>
            """

        card += """
        </div>
        """

        cards.append(
            card
        )

    return "\n".join(cards)


# ============================================================
# VISUALIZATION SECTION
# ============================================================

def generate_visualization_section(
    visualizations,
    report_output_file=DEFAULT_REPORT_FILE
):
    """
    Generate visualization cards with images embedded directly into the HTML.

    The final report is self-contained: PNG/JPEG/SVG images are converted
    into Base64 data URIs. This means the downloaded HTML does not depend
    on a separate visualizations/ directory.

    Path resolution is deliberately robust because visualization paths can
    be relative, absolute, POSIX-style, Windows-style, or only filenames.
    """

    report_output_file = Path(report_output_file).resolve()

    def normalize_file_path(file_path):
        """
        Normalize path strings coming from JSON.

        Handles Windows paths even when the pipeline is running on Linux.
        """
        if file_path is None:
            return ""

        value = str(file_path).strip()

        if not value:
            return ""

        # Convert Windows separators to the current OS separator.
        value = value.replace("\\", "/")

        return value

    def resolve_image_path(file_path):
        """
        Resolve a visualization image against several sensible roots.
        """

        normalized = normalize_file_path(file_path)

        if not normalized:
            return None

        raw_path = Path(normalized)

        # If an absolute path is still valid, use it directly.
        if raw_path.is_absolute():
            try:
                if raw_path.exists() and raw_path.is_file():
                    return raw_path.resolve()
            except OSError:
                pass

        # Roots that commonly apply to this project.
        project_root = Path.cwd().resolve()
        report_dir = report_output_file.parent
        report_parent = report_dir.parent

        candidates = [
            # Path exactly as stored.
            raw_path,

            # Relative to current working directory.
            project_root / raw_path,

            # Relative to the report directory.
            report_dir / raw_path,

            # Relative to the report's parent directory.
            report_parent / raw_path,

            # Standard pipeline visualization directory.
            project_root / "output" / "visualizations" / raw_path.name,

            # Report directory's visualization folder.
            report_dir / "visualizations" / raw_path.name,

            # Project output visualization folder.
            project_root / "output" / "visualizations" / raw_path.name,

            # Filename only.
            project_root / raw_path.name,
        ]

        # Remove duplicates while preserving order.
        unique_candidates = []
        seen = set()

        for candidate in candidates:
            try:
                candidate = candidate.resolve()
                key = str(candidate)
            except OSError:
                continue

            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        for candidate in unique_candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate
            except OSError:
                continue

        return None

    def image_to_data_uri(file_path):
        """
        Read an image and convert it to a browser-safe Base64 data URI.
        """

        image_path = resolve_image_path(file_path)

        if image_path is None:
            return None

        try:
            mime_type, _ = mimetypes.guess_type(
                image_path.name
            )

            if mime_type is None:
                suffix = image_path.suffix.lower()

                mime_type = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                    ".svg": "image/svg+xml",
                }.get(
                    suffix,
                    "application/octet-stream"
                )

            image_bytes = image_path.read_bytes()

            if not image_bytes:
                return None

            encoded_image = base64.b64encode(
                image_bytes
            ).decode("ascii")

            return (
                f"data:{mime_type};base64,"
                f"{encoded_image}"
            )

        except (OSError, ValueError):
            return None

    cards = []

    if not isinstance(visualizations, list):
        return ""

    for item in visualizations:

        if not isinstance(item, dict):
            continue

        visualization_type = item.get(
            "type",
            "Visualization"
        )

        variable = item.get(
            "variable",
            ""
        )

        variables = item.get(
            "variables",
            []
        )

        file_path = item.get(
            "file",
            ""
        )

        # --------------------------------------------------------
        # Determine chart title
        # --------------------------------------------------------

        if isinstance(variables, list) and variables:
            title = " × ".join(
                str(value)
                for value in variables
            )

        elif variable:
            title = str(variable)

        else:
            title = visualization_type

        # --------------------------------------------------------
        # Embed image
        # --------------------------------------------------------

        image_data_uri = image_to_data_uri(
            file_path
        )

        if image_data_uri:

            image_html = f"""
                <img
                    src="{image_data_uri}"
                    alt="{escape_html(title)}"
                    style="width:100%;height:auto;display:block;"
                >
            """

            error_html = ""

        else:

            image_html = ""

            error_html = """
                <div
                    class="image-error-message"
                    style="display:block;"
                >
                    Visualization image could not be loaded.
                </div>
            """

        # --------------------------------------------------------
        # Visualization card
        # --------------------------------------------------------

        cards.append(
            f"""
            <div class="chart-card">

                <h3>
                    {escape_html(title)}
                </h3>

                <p class="chart-type">
                    {escape_html(visualization_type)}
                </p>

                {image_html}

                {error_html}

            </div>
            """
        )

    return "\n".join(cards)

# ============================================================
# REPORT STATISTICS
# ============================================================

def calculate_report_statistics(
    statistical_results
):

    if not isinstance(
        statistical_results,
        list
    ):
        statistical_results = []

    total = len(
        statistical_results
    )

    raw_significant = 0
    fdr_significant = 0
    bonferroni_significant = 0
    raw_non_significant = 0
    errors = 0

    for result in statistical_results:

        test_name = str(
            get_test_name(result)
        ).lower()

        status = str(
            result.get(
                "status",
                ""
            )
        ).lower()

        if (
            "error" in test_name
            or "error" in status
        ):

            errors += 1

            continue

        raw_p = get_raw_p_value(
            result
        )

        fdr_p = get_fdr_p_value(
            result
        )

        bonf_p = get_bonferroni_p_value(
            result
        )

        if raw_p is not None:

            if raw_p < 0.05:
                raw_significant += 1

            else:
                raw_non_significant += 1

        if (
            fdr_p is not None
            and fdr_p < 0.05
        ):

            fdr_significant += 1

        if (
            bonf_p is not None
            and bonf_p < 0.05
        ):

            bonferroni_significant += 1

    return {
        "total":
            total,

        "raw_significant":
            raw_significant,

        "fdr_significant":
            fdr_significant,

        "bonferroni_significant":
            bonferroni_significant,

        "raw_non_significant":
            raw_non_significant,

        "errors":
            errors
    }


# ============================================================
# GENERATE DATASET COLUMN LIST
# ============================================================

def generate_column_list(
    dataset_info
):

    columns = (
        dataset_info.get(
            "column_names",
            []
        )
    )

    if not columns:
        return "No columns detected."

    items = []

    for column in columns:

        items.append(
            f"""
            <span class="column-chip">
                {escape_html(column)}
            </span>
            """
        )

    return "\n".join(items)


# ============================================================
# REPORT HTML
# ============================================================

def generate_html_report(
    dataset_info,
    data_dictionary,
    statistical_results,
    interpretations,
    visualizations
):

    generated_time = (
        datetime.now().strftime(
            "%d %B %Y, %H:%M"
        )
    )

    variable_table = (
        generate_variable_table(
            data_dictionary,
            dataset_info
        )
    )

    statistics_table = (
        generate_statistics_table(
            statistical_results
        )
    )

    interpretation_cards = (
        generate_interpretation_cards(
            interpretations
        )
    )

    visualization_cards = (
        generate_visualization_section(
            visualizations
        )
    )

    report_statistics = (
        calculate_report_statistics(
            statistical_results
        )
    )

    column_list = (
        generate_column_list(
            dataset_info
        )
    )

    html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
    Automated Data Analysis Report
</title>


<style>

/* ========================================================
   GLOBAL
======================================================== */

* {{
    box-sizing: border-box;
}}

html {{
    scroll-behavior: smooth;
}}

body {{

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #f4f6f8;

    color: #1f2937;

    line-height: 1.6;
}}


.container {{

    max-width: 1500px;

    margin: auto;

    padding: 30px;
}}


/* ========================================================
   HEADER
======================================================== */

.header {{

    background:
        linear-gradient(
            135deg,
            #111827,
            #374151
        );

    color: white;

    padding: 35px;

    border-radius: 16px;

    margin-bottom: 25px;

    box-shadow:
        0 8px 25px
        rgba(0, 0, 0, 0.12);
}}


.header h1 {{

    margin: 0 0 10px 0;

    font-size: 34px;
}}


.header p {{

    margin: 5px 0;

    color: #d1d5db;
}}


.header .dataset-name {{

    margin-top: 18px;

    display: inline-block;

    padding: 7px 12px;

    border-radius: 20px;

    background: rgba(255,255,255,0.12);

    color: white;

    font-size: 13px;
}}


/* ========================================================
   NAVIGATION
======================================================== */

.navbar {{

    background: white;

    border-radius: 12px;

    padding: 14px 18px;

    margin-bottom: 25px;

    box-shadow:
        0 2px 8px
        rgba(0,0,0,0.06);

    display: flex;

    flex-wrap: wrap;

    gap: 8px;
}}


.navbar a {{

    text-decoration: none;

    color: #374151;

    padding: 8px 13px;

    border-radius: 7px;

    font-size: 14px;
}}


.navbar a:hover {{

    background: #f3f4f6;
}}


/* ========================================================
   SECTIONS
======================================================== */

.section {{

    background: white;

    padding: 25px;

    border-radius: 14px;

    margin-bottom: 25px;

    box-shadow:
        0 2px 8px
        rgba(0, 0, 0, 0.06);
}}


.section h2 {{

    margin-top: 0;

    margin-bottom: 20px;

    font-size: 24px;
}}


.section-description {{

    color: #6b7280;

    margin-top: -10px;

    margin-bottom: 20px;
}}


/* ========================================================
   SUMMARY
======================================================== */

.summary-grid {{

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(180px, 1fr)
        );

    gap: 18px;
}}


.summary-card {{

    padding: 22px;

    border-radius: 12px;

    background: #f8fafc;

    border: 1px solid #e5e7eb;
}}


.summary-card .number {{

    font-size: 30px;

    font-weight: bold;

    margin-bottom: 5px;
}}


.summary-card .label {{

    color: #6b7280;

    font-size: 14px;
}}


.summary-card.highlight {{

    background: #f0fdf4;

    border-color: #bbf7d0;
}}


.summary-card.warning-card {{

    background: #fff7ed;

    border-color: #fed7aa;
}}


/* ========================================================
   COLUMNS
======================================================== */

.column-list {{

    display: flex;

    flex-wrap: wrap;

    gap: 8px;
}}


.column-chip {{

    display: inline-block;

    padding: 6px 10px;

    background: #f3f4f6;

    border: 1px solid #e5e7eb;

    border-radius: 7px;

    font-size: 13px;
}}


/* ========================================================
   TABLES
======================================================== */

.table-container {{

    overflow-x: auto;
}}


table {{

    width: 100%;

    border-collapse: collapse;
}}


th,
td {{

    padding: 12px;

    border-bottom:
        1px solid #e5e7eb;

    text-align: left;

    vertical-align: top;
}}


th {{

    background: #f8fafc;

    font-weight: 600;

    white-space: nowrap;
}}


tr:hover {{

    background: #f9fafb;
}}


td.significant {{

    font-weight: 700;

    color: #166534;

    background: #f0fdf4;
}}


td.not-significant {{

    color: #6b7280;
}}


/* ========================================================
   INTERPRETATION
======================================================== */

.interpretation-card {{

    padding: 20px;

    margin-bottom: 15px;

    border:
        1px solid #e5e7eb;

    border-radius: 12px;

    background: #fafafa;
}}


.interpretation-header {{

    display: flex;

    justify-content: space-between;

    align-items: flex-start;

    gap: 15px;

    margin-bottom: 12px;
}}


.interpretation-card h3 {{

    margin: 0;

    font-size: 18px;
}}


.interpretation-text {{

    line-height: 1.7;

    margin-top: 0;

    margin-bottom: 15px;
}}


.small-muted {{

    color: #6b7280;

    font-size: 13px;

    margin-top: 3px;
}}


.result-badge {{

    display: inline-block;

    padding: 5px 10px;

    border-radius: 20px;

    font-size: 12px;

    white-space: nowrap;
}}


.result-badge.significant {{

    background: #dcfce7;

    color: #166534;
}}


.result-badge.not-significant {{

    background: #f3f4f6;

    color: #6b7280;
}}


.warning {{

    margin-top: 12px;

    padding: 12px;

    background: #fff7ed;

    border-left:
        4px solid #f97316;

    color: #9a3412;

    line-height: 1.5;

    border-radius: 4px;
}}


/* ========================================================
   CHARTS
======================================================== */

.chart-grid {{

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(400px, 1fr)
        );

    gap: 25px;
}}


.chart-card {{

    border:
        1px solid #e5e7eb;

    border-radius: 12px;

    padding: 15px;

    background: white;

    overflow: hidden;
}}


.chart-card h3 {{

    margin-top: 0;

    margin-bottom: 5px;

    font-size: 18px;
}}


.chart-type {{

    color: #6b7280;

    font-size: 13px;

    margin-top: 0;
}}


.chart-card img {{

    width: 100%;

    height: auto;

    display: block;

    margin-top: 10px;

    border-radius: 6px;
}}


.image-error-message {{

    display: none;

    color: #b91c1c;

    background: #fef2f2;

    padding: 10px;

    margin-top: 10px;

    border-radius: 6px;

    font-size: 13px;
}}


.chart-card.image-error
.image-error-message {{

    display: block;
}}


/* ========================================================
   FOOTER
======================================================== */

.footer {{

    text-align: center;

    color: #6b7280;

    font-size: 13px;

    padding: 25px;
}}


/* ========================================================
   MOBILE
======================================================== */

@media (max-width: 700px) {{

    .container {{

        padding: 15px;
    }}

    .section {{

        padding: 18px;
    }}

    .chart-grid {{

        grid-template-columns: 1fr;
    }}

    .interpretation-header {{

        flex-direction: column;
    }}

    .header h1 {{

        font-size: 27px;
    }}
}}


</style>

</head>


<body>


<div class="container">


<!-- =====================================================
     HEADER
====================================================== -->

<div class="header">

    <h1>
        Automated Data Analysis Report
    </h1>

    <p>
        AI-assisted statistical analysis,
        interpretation and visualization
    </p>

    <p>
        Generated on
        {escape_html(generated_time)}
    </p>

    <span class="dataset-name">
        Dataset:
        {escape_html(
            dataset_info.get(
                "file_name",
                "Unknown"
            )
        )}
    </span>

</div>


<!-- =====================================================
     NAVIGATION
====================================================== -->

<div class="navbar">

    <a href="#overview">
        Overview
    </a>

    <a href="#variables">
        Variables
    </a>

    <a href="#statistics">
        Statistical Analysis
    </a>

    <a href="#interpretations">
        Interpretations
    </a>

    <a href="#visualizations">
        Visualizations
    </a>

</div>


<!-- =====================================================
     DATASET OVERVIEW
====================================================== -->

<div
    class="section"
    id="overview"
>

    <h2>
        Dataset Overview
    </h2>

    <p class="section-description">
        Summary of the dataset and the automated
        analysis performed by the pipeline.
    </p>

    <div class="summary-grid">


        <div class="summary-card">

            <div class="number">
                {dataset_info["rows"]}
            </div>

            <div class="label">
                Rows / Observations
            </div>

        </div>


        <div class="summary-card">

            <div class="number">
                {dataset_info["columns"]}
            </div>

            <div class="label">
                Variables
            </div>

        </div>


        <div class="summary-card">

            <div class="number">
                {report_statistics["total"]}
            </div>

            <div class="label">
                Statistical Analyses
            </div>

        </div>


        <div class="summary-card highlight">

            <div class="number">
                {report_statistics["raw_significant"]}
            </div>

            <div class="label">
                Raw p &lt; 0.05
            </div>

        </div>


        <div class="summary-card">

            <div class="number">
                {report_statistics["fdr_significant"]}
            </div>

            <div class="label">
                Significant after FDR
            </div>

        </div>


        <div class="summary-card">

            <div class="number">
                {report_statistics["bonferroni_significant"]}
            </div>

            <div class="label">
                Significant after Bonferroni
            </div>

        </div>


        <div class="summary-card">

            <div class="number">
                {len(interpretations)}
            </div>

            <div class="label">
                Interpretations
            </div>

        </div>


        <div class="summary-card">

            <div class="number">
                {len(visualizations)}
            </div>

            <div class="label">
                Visualizations
            </div>

        </div>


        <div class="summary-card warning-card">

            <div class="number">
                {report_statistics["errors"]}
            </div>

            <div class="label">
                Analysis Errors
            </div>

        </div>


    </div>

</div>


<!-- =====================================================
     DATASET VARIABLES
====================================================== -->

<div class="section">

    <h2>
        Dataset Variables
    </h2>

    <div class="column-list">

        {column_list}

    </div>

</div>


<!-- =====================================================
     VARIABLE PROFILE
====================================================== -->

<div
    class="section"
    id="variables"
>

    <h2>
        Variable Profile
    </h2>

    <p class="section-description">
        AI-assisted classification of variables
        based on their semantic type and
        measurement scale. Variables present in
        the dataset but missing from the AI data
        dictionary are shown as “Not classified”.
    </p>

    <div class="table-container">

        <table>

            <thead>

                <tr>

                    <th>
                        Variable
                    </th>

                    <th>
                        Semantic Type
                    </th>

                    <th>
                        Measurement Scale
                    </th>

                    <th>
                        Subtype
                    </th>

                    <th>
                        Confidence
                    </th>

                    <th>
                        Ordered
                    </th>

                </tr>

            </thead>


            <tbody>

                {variable_table}

            </tbody>

        </table>

    </div>

</div>


<!-- =====================================================
     STATISTICAL RESULTS
====================================================== -->

<div
    class="section"
    id="statistics"
>

    <h2>
        Statistical Analysis
    </h2>

    <p class="section-description">
        Statistical tests automatically selected
        based on the detected characteristics
        of the variables. Raw p-values are shown
        alongside Benjamini–Hochberg FDR and
        Bonferroni-adjusted p-values.
    </p>

    <div class="table-container">

        <table>

            <thead>

                <tr>

                    <th>
                        Variables
                    </th>

                    <th>
                        Test
                    </th>

                    <th>
                        Statistic
                    </th>

                    <th>
                        Raw p-value
                    </th>

                    <th>
                        FDR-adjusted p
                    </th>

                    <th>
                        Bonferroni-adjusted p
                    </th>

                    <th>
                        N
                    </th>

                    <th>
                        Raw Result
                    </th>

                </tr>

            </thead>


            <tbody>

                {statistics_table}

            </tbody>

        </table>

    </div>

</div>


<!-- =====================================================
     INTERPRETATIONS
====================================================== -->

<div
    class="section"
    id="interpretations"
>

    <h2>
        Statistical Interpretation
    </h2>

    <p class="section-description">
        Automated interpretation of the statistical
        results, including significance and
        important analytical warnings.
    </p>

    {interpretation_cards}

</div>


<!-- =====================================================
     VISUALIZATIONS
====================================================== -->

<div
    class="section"
    id="visualizations"
>

    <h2>
        Visualizations
    </h2>

    <p class="section-description">
        Automatically generated visualizations
        based on the detected variable types
        and relationships.
    </p>

    <div class="chart-grid">

        {visualization_cards}

    </div>

</div>


<!-- =====================================================
     FOOTER
====================================================== -->

<div class="footer">

    Automated Data Analysis Pipeline
    <br>
    AI-assisted data profiling, statistical
    analysis, interpretation and visualization

</div>


</div>


</body>

</html>
"""

    return html


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    html,
    output_file=DEFAULT_REPORT_FILE
):

    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(html)

    print(
        f"HTML report saved to: "
        f"{output_file}"
    )

    return output_file


# ============================================================
# MAIN
# ============================================================

def main(
    data_file=DEFAULT_DATA_FILE,
    dictionary_file=DEFAULT_DICTIONARY_FILE,
    statistics_file=DEFAULT_STATISTICS_FILE,
    interpretations_file=DEFAULT_INTERPRETATIONS_FILE,
    visualizations_file=DEFAULT_VISUALIZATIONS_FILE,
    output_file=DEFAULT_REPORT_FILE
):

    print(
        "Starting report generator..."
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_info = (
        load_dataset_info(
            data_file
        )
    )

    # --------------------------------------------------------
    # JSON outputs
    # --------------------------------------------------------

    data_dictionary = load_json(
        dictionary_file
    )

    statistical_results = load_json(
        statistics_file
    )

    interpretations = load_json(
        interpretations_file
    )

    visualizations = load_json(
        visualizations_file
    )

    # --------------------------------------------------------
    # Pipeline summary
    # --------------------------------------------------------

    print(
        f"Dataset: "
        f"{dataset_info['rows']} rows × "
        f"{dataset_info['columns']} columns"
    )

    print(
        f"Dataset file: "
        f"{dataset_info['file_name']}"
    )

    print(
        f"Variables: "
        f"{len(dataset_info.get('column_names', []))}"
    )

    print(
        f"Statistical results: "
        f"{len(statistical_results)}"
    )

    print(
        f"Interpretations: "
        f"{len(interpretations)}"
    )

    print(
        f"Visualizations: "
        f"{len(visualizations)}"
    )

    # --------------------------------------------------------
    # Generate HTML
    # --------------------------------------------------------

    html = generate_html_report(
        dataset_info,
        data_dictionary,
        statistical_results,
        interpretations,
        visualizations
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_report(
        html,
        output_file
    )

    print(
        "Report generation completed successfully."
    )

    return html


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate an automated HTML "
            "data analysis report."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to the source dataset "
            "(CSV or Excel)."
        )
    )

    parser.add_argument(
        "--dictionary",
        default=str(
            DEFAULT_DICTIONARY_FILE
        ),
        help=(
            "Path to data_dictionary.json."
        )
    )

    parser.add_argument(
        "--statistics",
        default=str(
            DEFAULT_STATISTICS_FILE
        ),
        help=(
            "Path to statistical_results.json."
        )
    )

    parser.add_argument(
        "--interpretations",
        default=str(
            DEFAULT_INTERPRETATIONS_FILE
        ),
        help=(
            "Path to interpretations.json."
        )
    )

    parser.add_argument(
        "--visualizations",
        default=str(
            DEFAULT_VISUALIZATIONS_FILE
        ),
        help=(
            "Path to visualizations.json."
        )
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_REPORT_FILE
        ),
        help=(
            "Path for the generated HTML report."
        )
    )

    args = parser.parse_args()

    try:

        main(
            data_file=args.input,
            dictionary_file=args.dictionary,
            statistics_file=args.statistics,
            interpretations_file=args.interpretations,
            visualizations_file=args.visualizations,
            output_file=args.output
        )

    except Exception as error:

        print(
            "\nReport generation failed."
        )

        print(
            f"Error: {error}"
        )

        raise