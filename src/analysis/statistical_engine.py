import json
from pathlib import Path

import pandas as pd
import numpy as np

from scipy.stats import (
    pearsonr,
    spearmanr,
    chi2_contingency,
    ttest_ind,
    mannwhitneyu,
    f_oneway,
    kruskal,
    shapiro,
    levene
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DICTIONARY_FILE = Path("output/data_dictionary.json")
DEFAULT_DATA_FILE = Path("data/test_dataset.xlsx")
DEFAULT_OUTPUT_FILE = Path("output/statistical_results.json")

ALPHA = 0.05


# ============================================================
# DATA LOADING
# ============================================================

def load_data_dictionary(dictionary_file=DEFAULT_DICTIONARY_FILE):
    """
    Load the AI-generated data dictionary.

    The path can be supplied by the pipeline, so this module
    does not depend on a specific dataset.
    """

    dictionary_file = Path(dictionary_file)

    if not dictionary_file.exists():
        raise FileNotFoundError(
            f"Data dictionary not found: {dictionary_file}"
        )

    with open(
        dictionary_file,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_data(data_file=DEFAULT_DATA_FILE):
    """
    Load a CSV or Excel dataset.

    This function accepts an explicit dataset path so the
    statistical engine can work with any supported dataset.
    """

    data_file = Path(data_file)

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}"
        )

    suffix = data_file.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(data_file)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(data_file)

    raise ValueError(
        f"Unsupported data file format: {suffix}. "
        "Supported formats: CSV, XLSX, XLS."
    )


# ============================================================
# VARIABLE CLASSIFICATION
# ============================================================

def classify_variables(data_dictionary):

    variables = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "identifier": [],
        "geographic": [],
        "binary": []
    }

    for item in data_dictionary:

        semantic_type = item.get(
            "semantic_type"
        )

        column_name = item.get(
            "column_name"
        )

        if semantic_type in variables:
            variables[semantic_type].append(
                column_name
            )

    return variables


# ============================================================
# ANALYSIS PAIR GENERATION
# ============================================================

def generate_analysis_pairs(data_dictionary):
    """
    Automatically determine appropriate statistical
    analyses from the AI-generated data dictionary.
    """

    pairs = []

    for i, var1 in enumerate(
        data_dictionary
    ):

        for var2 in data_dictionary[i + 1:]:

            name1 = var1["column_name"]
            name2 = var2["column_name"]

            type1 = var1.get(
                "semantic_type"
            )

            type2 = var2.get(
                "semantic_type"
            )

            scale1 = var1.get(
                "measurement_scale"
            )

            scale2 = var2.get(
                "measurement_scale"
            )

            ordered1 = bool(
                var1.get(
                    "ordered",
                    False
                )
            )

            ordered2 = bool(
                var2.get(
                    "ordered",
                    False
                )
            )

            # ------------------------------------------------
            # NUMERIC + NUMERIC
            # ------------------------------------------------

            if (
                type1 == "numeric"
                and type2 == "numeric"
            ):

                if (
                    scale1 == "ordinal"
                    or scale2 == "ordinal"
                ):

                    pairs.append({
                        "variables": [
                            name1,
                            name2
                        ],
                        "suggested_tests": [
                            "Spearman correlation"
                        ]
                    })

                else:

                    pairs.append({
                        "variables": [
                            name1,
                            name2
                        ],
                        "suggested_tests": [
                            "Pearson correlation",
                            "Spearman correlation"
                        ]
                    })

            # ------------------------------------------------
            # CATEGORICAL/BINARY + NUMERIC
            # ------------------------------------------------

            elif (
                type1 in [
                    "categorical",
                    "binary"
                ]
                and type2 == "numeric"
            ):

                if ordered1:

                    pairs.append({
                        "variables": [
                            name1,
                            name2
                        ],
                        "suggested_tests": [
                            "Spearman correlation"
                        ]
                    })

                else:

                    pairs.append({
                        "variables": [
                            name1,
                            name2
                        ],
                        "suggested_tests": [
                            "Group comparison"
                        ]
                    })

            # ------------------------------------------------
            # NUMERIC + CATEGORICAL/BINARY
            # ------------------------------------------------

            elif (
                type1 == "numeric"
                and type2 in [
                    "categorical",
                    "binary"
                ]
            ):

                if ordered2:

                    pairs.append({
                        "variables": [
                            name1,
                            name2
                        ],
                        "suggested_tests": [
                            "Spearman correlation"
                        ]
                    })

                else:

                    pairs.append({
                        "variables": [
                            name1,
                            name2
                        ],
                        "suggested_tests": [
                            "Group comparison"
                        ]
                    })

            # ------------------------------------------------
            # ORDINAL + ORDINAL
            # ------------------------------------------------

            elif (
                type1 == "categorical"
                and type2 == "categorical"
                and ordered1
                and ordered2
            ):

                pairs.append({
                    "variables": [
                        name1,
                        name2
                    ],
                    "suggested_tests": [
                        "Spearman correlation"
                    ]
                })

            # ------------------------------------------------
            # CATEGORICAL/BINARY + CATEGORICAL/BINARY
            # ------------------------------------------------

            elif (
                type1 in [
                    "categorical",
                    "binary"
                ]
                and type2 in [
                    "categorical",
                    "binary"
                ]
            ):

                pairs.append({
                    "variables": [
                        name1,
                        name2
                    ],
                    "suggested_tests": [
                        "Chi-square test"
                    ]
                })

    return pairs


# ============================================================
# CATEGORY MAPPING FOR ORDINAL VARIABLES
# ============================================================

def build_category_mapping(
    metadata
):

    category_order = metadata.get(
        "category_order",
        []
    )

    if not category_order:
        return None

    return {
        value: rank
        for rank, value in enumerate(
            category_order,
            start=1
        )
    }


# ============================================================
# PEARSON CORRELATION
# ============================================================

def run_pearson(
    data,
    variable1,
    variable2
):

    subset = data[
        [
            variable1,
            variable2
        ]
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

    if len(subset) < 3:

        return {
            "test": "Pearson correlation",
            "variable1": variable1,
            "variable2": variable2,
            "correlation": None,
            "p_value": None,
            "sample_size": len(subset),
            "status": "insufficient_data"
        }

    if (
        subset[variable1].nunique() < 2
        or subset[variable2].nunique() < 2
    ):

        return {
            "test": "Pearson correlation",
            "variable1": variable1,
            "variable2": variable2,
            "correlation": None,
            "p_value": None,
            "sample_size": len(subset),
            "status": "zero_variance"
        }

    correlation, p_value = pearsonr(
        subset[variable1],
        subset[variable2]
    )

    return {
        "test": "Pearson correlation",
        "variable1": variable1,
        "variable2": variable2,
        "correlation": float(correlation),
        "p_value": float(p_value),
        "sample_size": len(subset)
    }


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

def run_spearman(
    data,
    variable1,
    variable2,
    data_dictionary
):

    subset = data[
        [
            variable1,
            variable2
        ]
    ].copy()

    dictionary = {
        item["column_name"]: item
        for item in data_dictionary
    }

    for variable in [
        variable1,
        variable2
    ]:

        metadata = dictionary.get(
            variable,
            {}
        )

        category_order = metadata.get(
            "category_order",
            []
        )

        if category_order:

            category_map = {
                value: rank
                for rank, value
                in enumerate(
                    category_order,
                    start=1
                )
            }

            subset[variable] = (
                subset[variable]
                .map(category_map)
            )

        else:

            subset[variable] = pd.to_numeric(
                subset[variable],
                errors="coerce"
            )

    subset = subset.dropna()

    if len(subset) < 3:

        return {
            "test": "Spearman correlation",
            "variable1": variable1,
            "variable2": variable2,
            "correlation": None,
            "p_value": None,
            "sample_size": len(subset),
            "status": "insufficient_data"
        }

    if (
        subset[variable1].nunique() < 2
        or subset[variable2].nunique() < 2
    ):

        return {
            "test": "Spearman correlation",
            "variable1": variable1,
            "variable2": variable2,
            "correlation": None,
            "p_value": None,
            "sample_size": len(subset),
            "status": "zero_variance"
        }

    correlation, p_value = spearmanr(
        subset[variable1],
        subset[variable2]
    )

    return {
        "test": "Spearman correlation",
        "variable1": variable1,
        "variable2": variable2,
        "correlation": float(correlation),
        "p_value": float(p_value),
        "sample_size": len(subset)
    }


# ============================================================
# CHI-SQUARE TEST
# ============================================================

def run_chi_square(
    data,
    variable1,
    variable2
):

    subset = data[
        [
            variable1,
            variable2
        ]
    ].dropna()

    if len(subset) == 0:

        return {
            "test":
                "Chi-square test of independence",
            "variable1": variable1,
            "variable2": variable2,
            "status": "no_data"
        }

    contingency_table = pd.crosstab(
        subset[variable1],
        subset[variable2]
    )

    if (
        contingency_table.shape[0] < 2
        or contingency_table.shape[1] < 2
    ):

        return {
            "test":
                "Chi-square test of independence",
            "variable1": variable1,
            "variable2": variable2,
            "sample_size": len(subset),
            "status":
                "insufficient_categories"
        }

    (
        chi2,
        p_value,
        degrees_of_freedom,
        expected
    ) = chi2_contingency(
        contingency_table
    )

    expected_counts = expected.flatten()

    if (
        expected_counts < 5
    ).any():

        assumption_warning = (
            "Some expected cell counts "
            "are below 5."
        )

    else:

        assumption_warning = None

    rows, columns = (
        contingency_table.shape
    )

    min_dimension = min(
        rows - 1,
        columns - 1
    )

    if (
        min_dimension > 0
        and len(subset) > 0
    ):

        cramers_v = np.sqrt(
            chi2
            / (
                len(subset)
                * min_dimension
            )
        )

    else:

        cramers_v = None

    return {
        "test":
            "Chi-square test of independence",
        "variable1": variable1,
        "variable2": variable2,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "cramers_v": (
            float(cramers_v)
            if cramers_v is not None
            else None
        ),
        "degrees_of_freedom":
            int(degrees_of_freedom),
        "sample_size":
            int(len(subset)),
        "assumption_warning":
            assumption_warning
    }


# ============================================================
# RESOLVE GROUP AND NUMERIC VARIABLES
# ============================================================

def resolve_group_variables(
    variables,
    data_dictionary
):
    """
    Determine which variable is categorical/binary
    and which variable is numeric.
    """

    if len(variables) != 2:

        raise ValueError(
            "Group comparison requires exactly "
            "two variables."
        )

    metadata_lookup = {
        item["column_name"]: item
        for item in data_dictionary
    }

    variable1 = variables[0]
    variable2 = variables[1]

    metadata1 = metadata_lookup.get(
        variable1
    )

    metadata2 = metadata_lookup.get(
        variable2
    )

    if (
        metadata1 is None
        or metadata2 is None
    ):

        raise ValueError(
            "Metadata not found for "
            f"{variable1} or {variable2}."
        )

    type1 = metadata1.get(
        "semantic_type"
    )

    type2 = metadata2.get(
        "semantic_type"
    )

    if (
        type1 in [
            "categorical",
            "binary"
        ]
        and type2 == "numeric"
    ):

        return variable1, variable2

    if (
        type2 in [
            "categorical",
            "binary"
        ]
        and type1 == "numeric"
    ):

        return variable2, variable1

    raise ValueError(
        "Invalid group comparison pair: "
        f"{variable1} ({type1}) + "
        f"{variable2} ({type2}). "
        "Expected one categorical/binary "
        "variable and one numeric variable."
    )


# ============================================================
# GROUP ASSUMPTION CHECKS
# ============================================================

def check_group_assumptions(
    groups
):

    normality_results = []

    for label, group in groups:

        group = np.asarray(
            group,
            dtype=float
        )

        if len(group) >= 3:

            try:

                statistic, p_value = (
                    shapiro(group)
                )

                normality_results.append({
                    "group":
                        str(label),
                    "sample_size":
                        int(len(group)),
                    "shapiro_statistic":
                        float(statistic),
                    "shapiro_p_value":
                        float(p_value),
                    "normal":
                        bool(
                            p_value > ALPHA
                        )
                })

            except Exception as error:

                normality_results.append({
                    "group":
                        str(label),
                    "sample_size":
                        int(len(group)),
                    "shapiro_statistic":
                        None,
                    "shapiro_p_value":
                        None,
                    "normal":
                        None,
                    "warning":
                        str(error)
                })

        else:

            normality_results.append({
                "group":
                    str(label),
                "sample_size":
                    int(len(group)),
                "shapiro_statistic":
                    None,
                "shapiro_p_value":
                    None,
                "normal":
                    None,
                "warning":
                    "Less than 3 observations; "
                    "normality could not be tested."
            })

    # --------------------------------------------------------
    # Levene's test
    # --------------------------------------------------------

    arrays = [
        np.asarray(
            group,
            dtype=float
        )
        for _, group in groups
    ]

    if len(arrays) >= 2:

        try:

            (
                levene_statistic,
                levene_p_value
            ) = levene(
                *arrays
            )

            equal_variances = bool(
                levene_p_value > ALPHA
            )

            levene_warning = None

        except Exception as error:

            levene_statistic = None
            levene_p_value = None
            equal_variances = None
            levene_warning = str(error)

    else:

        levene_statistic = None
        levene_p_value = None
        equal_variances = None
        levene_warning = None

    result = {
        "normality":
            normality_results,
        "levene_statistic":
            (
                float(levene_statistic)
                if levene_statistic is not None
                else None
            ),
        "levene_p_value":
            (
                float(levene_p_value)
                if levene_p_value is not None
                else None
            ),
        "equal_variances":
            equal_variances
    }

    if levene_warning is not None:

        result["levene_warning"] = (
            levene_warning
        )

    return result


# ============================================================
# COHEN'S D
# ============================================================

def calculate_cohens_d(
    group1,
    group2
):

    group1 = np.asarray(
        group1,
        dtype=float
    )

    group2 = np.asarray(
        group2,
        dtype=float
    )

    n1 = len(group1)
    n2 = len(group2)

    if (
        n1 < 2
        or n2 < 2
    ):

        return None

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    var1 = np.var(
        group1,
        ddof=1
    )

    var2 = np.var(
        group2,
        ddof=1
    )

    denominator = (
        n1 + n2 - 2
    )

    if denominator <= 0:
        return None

    pooled_variance = (
        (
            (n1 - 1) * var1
            + (n2 - 1) * var2
        )
        / denominator
    )

    pooled_std = np.sqrt(
        pooled_variance
    )

    if pooled_std == 0:
        return None

    return float(
        (mean1 - mean2)
        / pooled_std
    )


# ============================================================
# ETA SQUARED
# ============================================================

def calculate_eta_squared(
    groups
):

    arrays = [
        np.asarray(
            group,
            dtype=float
        )
        for _, group in groups
    ]

    all_values = np.concatenate(
        arrays
    )

    grand_mean = np.mean(
        all_values
    )

    between_group_ss = sum(
        len(group)
        * (
            np.mean(group)
            - grand_mean
        ) ** 2
        for group in arrays
    )

    total_ss = np.sum(
        (
            all_values
            - grand_mean
        ) ** 2
    )

    if total_ss == 0:
        return None

    return float(
        between_group_ss
        / total_ss
    )


# ============================================================
# EPSILON SQUARED
# ============================================================

def calculate_epsilon_squared(
    groups,
    statistic
):

    n = sum(
        len(group)
        for _, group in groups
    )

    k = len(groups)

    if (
        n <= k
        or statistic is None
    ):

        return None

    value = (
        statistic
        - k
        + 1
    ) / (
        n - k
    )

    value = max(
        0.0,
        value
    )

    return float(value)


# ============================================================
# GROUP COMPARISON
# ============================================================

def run_group_comparison(
    data,
    categorical_variable,
    numeric_variable
):
    """
    Compare a numeric variable across groups defined
    by a categorical/binary variable.
    """

    if categorical_variable not in data.columns:

        raise ValueError(
            f"Grouping variable "
            f"'{categorical_variable}' "
            "does not exist in the dataset."
        )

    if numeric_variable not in data.columns:

        raise ValueError(
            f"Numeric variable "
            f"'{numeric_variable}' "
            "does not exist in the dataset."
        )

    subset = data[
        [
            categorical_variable,
            numeric_variable
        ]
    ].copy()

    # Only the numeric variable is converted.
    subset[numeric_variable] = pd.to_numeric(
        subset[numeric_variable],
        errors="coerce"
    )

    subset = subset.dropna()

    if len(subset) == 0:

        return {
            "status": "no_valid_data",
            "variable1":
                categorical_variable,
            "variable2":
                numeric_variable
        }

    grouped = []

    for label, group in (
        subset.groupby(
            categorical_variable,
            sort=False
        )
    ):

        values = (
            group[numeric_variable]
            .astype(float)
            .to_numpy()
        )

        if len(values) > 0:

            grouped.append(
                (
                    label,
                    values
                )
            )

    number_of_groups = len(
        grouped
    )

    if number_of_groups < 2:

        return {
            "status":
                "insufficient_groups",
            "variable1":
                categorical_variable,
            "variable2":
                numeric_variable,
            "number_of_groups":
                number_of_groups,
            "sample_size":
                len(subset)
        }

    assumptions = (
        check_group_assumptions(
            grouped
        )
    )

    normality_values = [
        item["normal"]
        for item
        in assumptions["normality"]
        if item["normal"] is not None
    ]

    all_normal = (
        len(normality_values)
        == number_of_groups
        and all(normality_values)
    )

    equal_variances = (
        assumptions[
            "equal_variances"
        ]
    )

    arrays = [
        values
        for _, values in grouped
    ]

    # --------------------------------------------------------
    # TWO GROUPS
    # --------------------------------------------------------

    if number_of_groups == 2:

        if all_normal:

            statistic, p_value = (
                ttest_ind(
                    arrays[0],
                    arrays[1],
                    equal_var=(
                        equal_variances
                        if equal_variances
                        is not None
                        else False
                    )
                )
            )

            if equal_variances:

                test_name = (
                    "Student's t-test"
                )

            else:

                test_name = (
                    "Welch's t-test"
                )

        else:

            statistic, p_value = (
                mannwhitneyu(
                    arrays[0],
                    arrays[1],
                    alternative="two-sided"
                )
            )

            test_name = (
                "Mann–Whitney U test"
            )

        cohens_d = (
            calculate_cohens_d(
                arrays[0],
                arrays[1]
            )
        )

        return {
            "test":
                test_name,
            "variable1":
                categorical_variable,
            "variable2":
                numeric_variable,
            "statistic":
                float(statistic),
            "p_value":
                float(p_value),
            "cohens_d":
                cohens_d,
            "number_of_groups":
                number_of_groups,
            "sample_size":
                len(subset),
            "group_labels": [
                str(label)
                for label, _
                in grouped
            ],
            "group_sizes": {
                str(label):
                    len(values)
                for label, values
                in grouped
            },
            "assumptions":
                assumptions
        }

    # --------------------------------------------------------
    # THREE OR MORE GROUPS
    # --------------------------------------------------------

    if number_of_groups >= 3:

        eta_squared = None
        epsilon_squared = None

        if (
            all_normal
            and equal_variances
        ):

            statistic, p_value = (
                f_oneway(
                    *arrays
                )
            )

            test_name = (
                "One-way ANOVA"
            )

            eta_squared = (
                calculate_eta_squared(
                    grouped
                )
            )

        else:

            statistic, p_value = (
                kruskal(
                    *arrays
                )
            )

            test_name = (
                "Kruskal–Wallis test"
            )

            epsilon_squared = (
                calculate_epsilon_squared(
                    grouped,
                    statistic
                )
            )

        return {
            "test":
                test_name,
            "variable1":
                categorical_variable,
            "variable2":
                numeric_variable,
            "statistic":
                float(statistic),
            "p_value":
                float(p_value),
            "eta_squared":
                eta_squared,
            "epsilon_squared":
                epsilon_squared,
            "number_of_groups":
                number_of_groups,
            "sample_size":
                len(subset),
            "group_labels": [
                str(label)
                for label, _
                in grouped
            ],
            "group_sizes": {
                str(label):
                    len(values)
                for label, values
                in grouped
            },
            "assumptions":
                assumptions
        }

    return {
        "status": "not_implemented",
        "variable1":
            categorical_variable,
        "variable2":
            numeric_variable
    }


# ============================================================
# RESOLVE GROUP COMPARISON
# ============================================================

def run_resolved_group_comparison(
    data,
    variables,
    data_dictionary
):

    (
        categorical_variable,
        numeric_variable
    ) = resolve_group_variables(
        variables,
        data_dictionary
    )

    return run_group_comparison(
        data,
        categorical_variable,
        numeric_variable
    )


# ============================================================
# RUN SELECTED TEST
# ============================================================

def run_selected_test(
    data,
    analysis,
    data_dictionary
):

    variables = analysis[
        "variables"
    ]

    tests = analysis[
        "suggested_tests"
    ]

    # --------------------------------------------------------
    # Pearson
    # --------------------------------------------------------

    if (
        "Pearson correlation"
        in tests
    ):

        return run_pearson(
            data,
            variables[0],
            variables[1]
        )

    # --------------------------------------------------------
    # Spearman
    # --------------------------------------------------------

    if (
        "Spearman correlation"
        in tests
    ):

        return run_spearman(
            data,
            variables[0],
            variables[1],
            data_dictionary
        )

    # --------------------------------------------------------
    # Chi-square
    # --------------------------------------------------------

    if (
        "Chi-square test"
        in tests
    ):

        return run_chi_square(
            data,
            variables[0],
            variables[1]
        )

    # --------------------------------------------------------
    # Group comparison
    # --------------------------------------------------------

    if (
        "Group comparison"
        in tests
    ):

        return run_resolved_group_comparison(
            data,
            variables,
            data_dictionary
        )

    return {
        "status":
            "not_implemented",
        "variables":
            variables,
        "suggested_tests":
            tests
    }


# ============================================================
# BENJAMINI-HOCHBERG FDR CORRECTION
# ============================================================

def apply_benjamini_hochberg(
    results
):
    """
    Apply Benjamini-Hochberg false discovery rate
    correction to all valid p-values.
    """

    valid_indices = []
    p_values = []

    for index, result in enumerate(
        results
    ):

        p_value = result.get(
            "p_value"
        )

        if p_value is None:
            continue

        try:

            p_value = float(
                p_value
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        if not np.isfinite(
            p_value
        ):

            continue

        valid_indices.append(
            index
        )

        p_values.append(
            p_value
        )

    m = len(
        p_values
    )

    if m == 0:
        return results

    order = np.argsort(
        p_values
    )

    sorted_p_values = np.array(
        p_values
    )[order]

    adjusted_sorted = (
        sorted_p_values
        * m
        / np.arange(
            1,
            m + 1
        )
    )

    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.minimum(
        adjusted_sorted,
        1.0
    )

    adjusted = np.empty(
        m
    )

    adjusted[order] = (
        adjusted_sorted
    )

    for position, result_index in enumerate(
        valid_indices
    ):

        raw_p = p_values[
            position
        ]

        fdr_p = adjusted[
            position
        ]

        result = results[
            result_index
        ]

        result[
            "p_value_fdr"
        ] = float(fdr_p)

        result[
            "fdr_significant"
        ] = bool(
            fdr_p < ALPHA
        )

        result[
            "raw_significant"
        ] = bool(
            raw_p < ALPHA
        )

    return results


# ============================================================
# BONFERRONI CORRECTION
# ============================================================

def apply_bonferroni(
    results
):

    valid_indices = []
    p_values = []

    for index, result in enumerate(
        results
    ):

        p_value = result.get(
            "p_value"
        )

        if p_value is None:
            continue

        try:

            p_value = float(
                p_value
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        if not np.isfinite(
            p_value
        ):

            continue

        valid_indices.append(
            index
        )

        p_values.append(
            p_value
        )

    m = len(
        p_values
    )

    if m == 0:
        return results

    for position, result_index in enumerate(
        valid_indices
    ):

        raw_p = p_values[
            position
        ]

        adjusted_p = min(
            raw_p * m,
            1.0
        )

        results[
            result_index
        ][
            "p_value_bonferroni"
        ] = float(
            adjusted_p
        )

        results[
            result_index
        ][
            "bonferroni_significant"
        ] = bool(
            adjusted_p < ALPHA
        )

    return results


# ============================================================
# ADD MULTIPLE TESTING CORRECTIONS
# ============================================================

def apply_multiple_testing_corrections(
    results
):

    results = (
        apply_benjamini_hochberg(
            results
        )
    )

    results = (
        apply_bonferroni(
            results
        )
    )

    return results


# ============================================================
# RUN ALL TESTS
# ============================================================

def run_all_tests(
    data,
    data_dictionary
):

    analyses = (
        generate_analysis_pairs(
            data_dictionary
        )
    )

    results = []

    for analysis in analyses:

        try:

            result = run_selected_test(
                data,
                analysis,
                data_dictionary
            )

            if result is None:

                result = {
                    "status":
                        "empty_result"
                }

            result[
                "analysis_pair"
            ] = analysis

            results.append(
                result
            )

        except Exception as error:

            results.append({
                "status":
                    "analysis_error",
                "variables":
                    analysis[
                        "variables"
                    ],
                "suggested_tests":
                    analysis[
                        "suggested_tests"
                    ],
                "error":
                    str(error)
            })

    # --------------------------------------------------------
    # Multiple-testing corrections
    # --------------------------------------------------------

    results = (
        apply_multiple_testing_corrections(
            results
        )
    )

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    output_file=DEFAULT_OUTPUT_FILE
):
    """
    Save statistical results to JSON.

    The output path can be supplied by the pipeline.
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            default=str
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run automated statistical analysis."
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
        "--output",
        required=False,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path for statistical results JSON."
    )

    args = parser.parse_args()

    data_file = Path(args.input)
    dictionary_file = Path(args.dictionary)
    output_file = Path(args.output)

    print(
        "Starting statistical engine..."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    dictionary = (
        load_data_dictionary(
            dictionary_file
        )
    )

    data = load_data(
        data_file
    )

    print(
        f"Dataset loaded: "
        f"{data.shape[0]} rows × "
        f"{data.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    variables = (
        classify_variables(
            dictionary
        )
    )

    print(
        "\nVariable classification:"
    )

    for category, names in (
        variables.items()
    ):

        print(
            f"  {category}: "
            f"{len(names)}"
        )

    # --------------------------------------------------------
    # Generate analyses
    # --------------------------------------------------------

    analyses = (
        generate_analysis_pairs(
            dictionary
        )
    )

    print(
        "\nGenerating automatic analyses..."
    )

    print(
        f"Analyses generated: "
        f"{len(analyses)}"
    )

    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    print(
        "\nRunning statistical tests..."
    )

    results = run_all_tests(
        data,
        dictionary
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_tests = len(
        results
    )

    error_count = sum(
        1
        for result in results
        if result.get(
            "status"
        ) == "analysis_error"
    )

    valid_results = [
        result
        for result in results
        if result.get(
            "p_value"
        ) is not None
    ]

    raw_significant = sum(
        1
        for result
        in valid_results
        if result.get(
            "raw_significant",
            False
        )
    )

    fdr_significant = sum(
        1
        for result
        in valid_results
        if result.get(
            "fdr_significant",
            False
        )
    )

    bonferroni_significant = sum(
        1
        for result
        in valid_results
        if result.get(
            "bonferroni_significant",
            False
        )
    )

    print(
        "\nStatistical analysis completed."
    )

    print(
        f"Total tests: "
        f"{total_tests}"
    )

    print(
        f"Raw p < 0.05: "
        f"{raw_significant}"
    )

    print(
        f"FDR significant: "
        f"{fdr_significant}"
    )

    print(
        f"Bonferroni significant: "
        f"{bonferroni_significant}"
    )

    print(
        f"Analysis errors: "
        f"{error_count}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        output_file
    )

    print(
        "\nResults saved to: "
        f"{output_file}"
    )