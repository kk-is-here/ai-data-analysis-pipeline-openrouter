import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATA_FILE = Path(
    "data/test_dataset.xlsx"
)

DEFAULT_DICTIONARY_FILE = Path(
    "output/data_dictionary.json"
)

DEFAULT_OUTPUT_DIR = Path(
    "output/visualizations"
)

DEFAULT_METADATA_FILE = Path(
    "output/visualizations.json"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_data(data_file=DEFAULT_DATA_FILE):
    """
    Load CSV or Excel dataset.

    The dataset path can be supplied by the pipeline, allowing
    this module to work with any supported dataset.
    """

    data_file = Path(data_file)

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}"
        )

    extension = data_file.suffix.lower()

    if extension == ".csv":

        data = pd.read_csv(
            data_file
        )

    elif extension in [
        ".xlsx",
        ".xls"
    ]:

        data = pd.read_excel(
            data_file
        )

    else:

        raise ValueError(
            f"Unsupported file format: "
            f"{extension}. "
            "Supported formats: CSV, XLSX, XLS."
        )

    print(
        f"Dataset loaded from: "
        f"{data_file}"
    )

    return data


def load_data_dictionary(
    dictionary_file=DEFAULT_DICTIONARY_FILE
):

    dictionary_file = Path(
        dictionary_file
    )

    if not dictionary_file.exists():
        raise FileNotFoundError(
            f"Data dictionary not found: "
            f"{dictionary_file}"
        )

    with open(
        dictionary_file,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

def create_output_directory(
    output_dir=DEFAULT_OUTPUT_DIR
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return output_dir


def clear_old_visualizations(
    output_dir=DEFAULT_OUTPUT_DIR
):

    """
    Remove old PNG files so that visualizations
    from previous datasets do not remain in the
    report.
    """

    output_dir = Path(
        output_dir
    )

    if not output_dir.exists():
        return

    removed = 0

    for file in output_dir.iterdir():

        if (
            file.is_file()
            and file.suffix.lower() == ".png"
        ):

            file.unlink()

            removed += 1

    if removed:

        print(
            f"Removed {removed} old visualization files."
        )


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(variable_name):

    invalid_characters = '<>:"/\\|?*'

    filename = str(
        variable_name
    )

    for character in invalid_characters:

        filename = filename.replace(
            character,
            "_"
        )

    filename = filename.replace(
        " ",
        "_"
    )

    return filename[:150]


# ============================================================
# NUMERIC DISTRIBUTION
# ============================================================

def plot_numeric_distribution(
    data,
    variable,
    output_dir=DEFAULT_OUTPUT_DIR
):

    """
    Histogram for continuous numeric variables.
    """

    output_dir = Path(
        output_dir
    )

    series = pd.to_numeric(
        data[variable],
        errors="coerce"
    ).dropna()

    if series.empty:
        return None

    plt.figure()

    series.plot(
        kind="hist",
        bins=10
    )

    plt.title(
        f"Distribution of {variable}"
    )

    plt.xlabel(
        variable
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    filename = (
        output_dir
        / f"{safe_filename(variable)}_distribution.png"
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return str(filename)


# ============================================================
# ORDINAL NUMERIC DISTRIBUTION
# ============================================================

def plot_ordinal_distribution(
    data,
    variable,
    category_order=None,
    output_dir=DEFAULT_OUTPUT_DIR
):

    """
    Ordered bar chart for ordinal numeric variables.
    """

    output_dir = Path(
        output_dir
    )

    series = data[
        variable
    ].dropna()

    if series.empty:
        return None

    counts = series.value_counts()

    # --------------------------------------------------------
    # Apply AI-generated category order
    # --------------------------------------------------------

    if category_order:

        ordered_values = []

        for category in category_order:

            if category in counts.index:

                ordered_values.append(
                    category
                )

        remaining_values = [
            value
            for value in counts.index
            if value not in ordered_values
        ]

        ordered_values.extend(
            remaining_values
        )

        counts = counts.reindex(
            ordered_values,
            fill_value=0
        )

    else:

        try:

            counts = counts.sort_index()

        except Exception:

            pass

    plt.figure()

    counts.plot(
        kind="bar"
    )

    plt.title(
        f"Distribution of {variable}"
    )

    plt.xlabel(
        variable
    )

    plt.ylabel(
        "Count"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    filename = (
        output_dir
        / f"{safe_filename(variable)}_ordinal_distribution.png"
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return str(filename)


# ============================================================
# CATEGORICAL DISTRIBUTION
# ============================================================

def plot_categorical_distribution(
    data,
    variable,
    category_order=None,
    output_dir=DEFAULT_OUTPUT_DIR
):

    output_dir = Path(
        output_dir
    )

    series = data[
        variable
    ].dropna()

    if series.empty:
        return None

    counts = series.value_counts()

    # --------------------------------------------------------
    # Apply AI category ordering
    # --------------------------------------------------------

    if category_order:

        ordered_categories = [
            category
            for category in category_order
            if category in counts.index
        ]

        remaining_categories = [
            category
            for category in counts.index
            if category not in ordered_categories
        ]

        ordered_categories.extend(
            remaining_categories
        )

        counts = counts.reindex(
            ordered_categories,
            fill_value=0
        )

    plt.figure()

    counts.plot(
        kind="bar"
    )

    plt.title(
        f"Distribution of {variable}"
    )

    plt.xlabel(
        variable
    )

    plt.ylabel(
        "Count"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    filename = (
        output_dir
        / f"{safe_filename(variable)}_distribution.png"
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return str(filename)


# ============================================================
# DATETIME TIME SERIES
# ============================================================

def plot_datetime_distribution(
    data,
    variable,
    output_dir=DEFAULT_OUTPUT_DIR
):

    """
    Create a time-series visualization for
    datetime variables.

    The chart shows the number of observations
    over time.
    """

    output_dir = Path(
        output_dir
    )

    series = pd.to_datetime(
        data[variable],
        errors="coerce"
    ).dropna()

    if series.empty:
        return None

    counts = (
        series
        .dt.date
        .value_counts()
        .sort_index()
    )

    if counts.empty:
        return None

    plt.figure()

    plt.plot(
        counts.index,
        counts.values
    )

    plt.title(
        f"Observations Over Time: {variable}"
    )

    plt.xlabel(
        variable
    )

    plt.ylabel(
        "Number of observations"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    filename = (
        output_dir
        / f"{safe_filename(variable)}_time_series.png"
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return str(filename)


# ============================================================
# NUMERIC VS NUMERIC
# ============================================================

def plot_numeric_vs_numeric(
    data,
    variable1,
    variable2,
    output_dir=DEFAULT_OUTPUT_DIR
):

    output_dir = Path(
        output_dir
    )

    subset = data[
        [variable1, variable2]
    ].copy()

    subset[variable1] = pd.to_numeric(
        subset[variable1],
        errors="coerce"
    )

    subset[variable2] = pd.to_numeric(
        subset[variable2],
        errors="coerce"
    )

    subset = subset.dropna()

    if len(subset) < 2:
        return None

    plt.figure()

    plt.scatter(
        subset[variable1],
        subset[variable2]
    )

    plt.title(
        f"{variable1} vs {variable2}"
    )

    plt.xlabel(
        variable1
    )

    plt.ylabel(
        variable2
    )

    plt.tight_layout()

    filename = (
        output_dir
        / f"{safe_filename(variable1)}_vs_"
        f"{safe_filename(variable2)}.png"
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return str(filename)


# ============================================================
# CATEGORICAL / ORDINAL VS NUMERIC
# ============================================================

def plot_grouped_numeric(
    data,
    grouping_variable,
    numeric_variable,
    category_order=None,
    output_dir=DEFAULT_OUTPUT_DIR
):

    """
    Create a box plot for a numeric variable across
    categorical or ordinal groups.
    """

    output_dir = Path(
        output_dir
    )

    subset = data[
        [
            grouping_variable,
            numeric_variable
        ]
    ].copy()

    subset[numeric_variable] = pd.to_numeric(
        subset[numeric_variable],
        errors="coerce"
    )

    subset = subset.dropna()

    if subset.empty:
        return None

    unique_categories = (
        subset[grouping_variable]
        .dropna()
        .unique()
    )

    # --------------------------------------------------------
    # Determine category order
    # --------------------------------------------------------

    if category_order:

        categories = [
            category
            for category in category_order
            if category in unique_categories
        ]

        remaining_categories = [
            category
            for category in unique_categories
            if category not in categories
        ]

        categories.extend(
            remaining_categories
        )

    else:

        categories = list(
            unique_categories
        )

    groups = []
    valid_categories = []

    for category in categories:

        values = subset.loc[
            subset[grouping_variable] == category,
            numeric_variable
        ].dropna().values

        if len(values) > 0:

            groups.append(
                values
            )

            valid_categories.append(
                category
            )

    if not groups:
        return None

    plt.figure()

    plt.boxplot(
        groups,
        tick_labels=valid_categories
    )

    plt.title(
        f"{numeric_variable} by "
        f"{grouping_variable}"
    )

    plt.xlabel(
        grouping_variable
    )

    plt.ylabel(
        numeric_variable
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    filename = (
        output_dir
        / f"{safe_filename(numeric_variable)}_by_"
        f"{safe_filename(grouping_variable)}.png"
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return str(filename)


# ============================================================
# GENERATE VISUALIZATIONS
# ============================================================

def generate_visualizations(
    data,
    data_dictionary,
    output_dir=DEFAULT_OUTPUT_DIR
):

    output_dir = create_output_directory(
        output_dir
    )

    visualizations = []

    # ========================================================
    # INDIVIDUAL VARIABLE VISUALIZATIONS
    # ========================================================

    for item in data_dictionary:

        variable = item[
            "column_name"
        ]

        semantic_type = item.get(
            "semantic_type"
        )

        measurement_scale = item.get(
            "measurement_scale"
        )

        category_order = item.get(
            "category_order",
            []
        )

        ordered = item.get(
            "ordered",
            False
        )

        # ----------------------------------------------------
        # Numeric
        # ----------------------------------------------------

        if semantic_type == "numeric":

            # Ordinal numeric
            if (
                measurement_scale
                == "ordinal"
                or ordered
            ):

                filename = (
                    plot_ordinal_distribution(
                        data,
                        variable,
                        category_order,
                        output_dir
                    )
                )

                if filename:

                    visualizations.append(
                        {
                            "type":
                                "ordered_bar_chart",

                            "variable":
                                variable,

                            "measurement_scale":
                                measurement_scale,

                            "file":
                                filename
                        }
                    )

            # Continuous numeric
            else:

                filename = (
                    plot_numeric_distribution(
                        data,
                        variable,
                        output_dir
                    )
                )

                if filename:

                    visualizations.append(
                        {
                            "type":
                                "histogram",

                            "variable":
                                variable,

                            "measurement_scale":
                                measurement_scale,

                            "file":
                                filename
                        }
                    )

        # ----------------------------------------------------
        # Categorical / Binary
        # ----------------------------------------------------

        elif semantic_type in [
            "categorical",
            "binary"
        ]:

            filename = (
                plot_categorical_distribution(
                    data,
                    variable,
                    category_order,
                    output_dir
                )
            )

            if filename:

                visualizations.append(
                    {
                        "type":
                            "bar_chart",

                        "variable":
                            variable,

                        "measurement_scale":
                            measurement_scale,

                        "file":
                            filename
                    }
                )

        # ----------------------------------------------------
        # Datetime
        # ----------------------------------------------------

        elif semantic_type == "datetime":

            filename = (
                plot_datetime_distribution(
                    data,
                    variable,
                    output_dir
                )
            )

            if filename:

                visualizations.append(
                    {
                        "type":
                            "time_series",

                        "variable":
                            variable,

                        "file":
                            filename
                    }
                )

    # ========================================================
    # PAIRWISE VISUALIZATIONS
    # ========================================================

    for i, item1 in enumerate(
        data_dictionary
    ):

        for item2 in data_dictionary[
            i + 1:
        ]:

            type1 = item1.get(
                "semantic_type"
            )

            type2 = item2.get(
                "semantic_type"
            )

            scale1 = item1.get(
                "measurement_scale"
            )

            scale2 = item2.get(
                "measurement_scale"
            )

            variable1 = item1[
                "column_name"
            ]

            variable2 = item2[
                "column_name"
            ]

            order1 = item1.get(
                "category_order",
                []
            )

            order2 = item2.get(
                "category_order",
                []
            )

            ordered1 = item1.get(
                "ordered",
                False
            )

            ordered2 = item2.get(
                "ordered",
                False
            )

            # ------------------------------------------------
            # Continuous Numeric + Continuous Numeric
            # ------------------------------------------------

            if (
                type1 == "numeric"
                and type2 == "numeric"
                and scale1 != "ordinal"
                and scale2 != "ordinal"
                and not ordered1
                and not ordered2
            ):

                filename = (
                    plot_numeric_vs_numeric(
                        data,
                        variable1,
                        variable2,
                        output_dir
                    )
                )

                if filename:

                    visualizations.append(
                        {
                            "type":
                                "scatter_plot",

                            "variables": [
                                variable1,
                                variable2
                            ],

                            "file":
                                filename
                        }
                    )

            # ------------------------------------------------
            # Ordinal Numeric + Continuous Numeric
            # ------------------------------------------------

            elif (
                type1 == "numeric"
                and type2 == "numeric"
                and scale1 == "ordinal"
                and scale2 != "ordinal"
            ):

                filename = (
                    plot_grouped_numeric(
                        data,
                        variable1,
                        variable2,
                        order1,
                        output_dir
                    )
                )

                if filename:

                    visualizations.append(
                        {
                            "type":
                                "box_plot",

                            "variables": [
                                variable1,
                                variable2
                            ],

                            "relationship":
                                "ordinal_numeric_vs_continuous",

                            "file":
                                filename
                        }
                    )

            elif (
                type1 == "numeric"
                and type2 == "numeric"
                and scale2 == "ordinal"
                and scale1 != "ordinal"
            ):

                filename = (
                    plot_grouped_numeric(
                        data,
                        variable2,
                        variable1,
                        order2,
                        output_dir
                    )
                )

                if filename:

                    visualizations.append(
                        {
                            "type":
                                "box_plot",

                            "variables": [
                                variable2,
                                variable1
                            ],

                            "relationship":
                                "continuous_vs_ordinal_numeric",

                            "file":
                                filename
                        }
                    )

            # ------------------------------------------------
            # Categorical / Binary + Numeric
            # ------------------------------------------------

            elif (
                type1 in [
                    "categorical",
                    "binary"
                ]
                and type2 == "numeric"
                and scale2 != "ordinal"
            ):

                filename = (
                    plot_grouped_numeric(
                        data,
                        variable1,
                        variable2,
                        order1,
                        output_dir
                    )
                )

                if filename:

                    visualizations.append(
                        {
                            "type":
                                "box_plot",

                            "variables": [
                                variable1,
                                variable2
                            ],

                            "file":
                                filename
                        }
                    )

            elif (
                type2 in [
                    "categorical",
                    "binary"
                ]
                and type1 == "numeric"
                and scale1 != "ordinal"
            ):

                filename = (
                    plot_grouped_numeric(
                        data,
                        variable2,
                        variable1,
                        order2,
                        output_dir
                    )
                )

                if filename:

                    visualizations.append(
                        {
                            "type":
                                "box_plot",

                            "variables": [
                                variable2,
                                variable1
                            ],

                            "file":
                                filename
                        }
                    )

            # ------------------------------------------------
            # Categorical / Binary + Ordinal Numeric
            # ------------------------------------------------

            elif (
                type1 in [
                    "categorical",
                    "binary"
                ]
                and type2 == "numeric"
                and scale2 == "ordinal"
            ):

                filename = (
                    plot_grouped_numeric(
                        data,
                        variable1,
                        variable2,
                        order1,
                        output_dir
                    )
                )

                if filename:

                    visualizations.append(
                        {
                            "type":
                                "box_plot",

                            "variables": [
                                variable1,
                                variable2
                            ],

                            "relationship":
                                "categorical_vs_ordinal",

                            "file":
                                filename
                        }
                    )

            elif (
                type2 in [
                    "categorical",
                    "binary"
                ]
                and type1 == "numeric"
                and scale1 == "ordinal"
            ):

                filename = (
                    plot_grouped_numeric(
                        data,
                        variable2,
                        variable1,
                        order2,
                        output_dir
                    )
                )

                if filename:

                    visualizations.append(
                        {
                            "type":
                                "box_plot",

                            "variables": [
                                variable2,
                                variable1
                            ],

                            "relationship":
                                "categorical_vs_ordinal",

                            "file":
                                filename
                        }
                    )

    return visualizations


# ============================================================
# SAVE VISUALIZATION METADATA
# ============================================================

def save_visualization_metadata(
    visualizations,
    metadata_file=DEFAULT_METADATA_FILE
):

    metadata_file = Path(
        metadata_file
    )

    metadata_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            visualizations,
            file,
            indent=2
        )

    print(
        f"Visualization metadata saved to: "
        f"{metadata_file}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate automated visualizations."
    )

    parser.add_argument(
        "--input",
        required=False,
        default=str(DEFAULT_DATA_FILE),
        help="Path to CSV or Excel dataset."
    )

    parser.add_argument(
        "--dictionary",
        required=False,
        default=str(DEFAULT_DICTIONARY_FILE),
        help="Path to AI data dictionary JSON."
    )

    parser.add_argument(
        "--output-dir",
        required=False,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated visualization files."
    )

    parser.add_argument(
        "--metadata-output",
        required=False,
        default=str(DEFAULT_METADATA_FILE),
        help="Path for visualization metadata JSON."
    )

    args = parser.parse_args()

    data_file = Path(
        args.input
    )

    dictionary_file = Path(
        args.dictionary
    )

    output_dir = Path(
        args.output_dir
    )

    metadata_file = Path(
        args.metadata_output
    )

    print(
        "Starting visualization engine..."
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    create_output_directory(
        output_dir
    )

    # --------------------------------------------------------
    # Remove visualizations from previous runs
    # --------------------------------------------------------

    clear_old_visualizations(
        output_dir
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = load_data(
        data_file
    )

    print(
        f"Dataset dimensions: "
        f"{data.shape[0]} rows × "
        f"{data.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Load AI data dictionary
    # --------------------------------------------------------

    data_dictionary = (
        load_data_dictionary(
            dictionary_file
        )
    )

    print(
        "AI data dictionary loaded."
    )

    print(
        f"Variables available: "
        f"{len(data_dictionary)}"
    )

    # --------------------------------------------------------
    # Generate visualizations
    # --------------------------------------------------------

    visualizations = (
        generate_visualizations(
            data,
            data_dictionary,
            output_dir
        )
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_visualization_metadata(
        visualizations,
        metadata_file
    )

    print(
        f"Generated "
        f"{len(visualizations)} visualizations."
    )

    print(
        "Visualization engine completed successfully."
    )