import argparse
import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INPUT_FILE = Path(
    "output/statistical_results.json"
)

DEFAULT_OUTPUT_FILE = Path(
    "output/interpretations.json"
)

ALPHA = 0.05


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results(input_file=DEFAULT_INPUT_FILE):
    """
    Load statistical analysis results from JSON.
    """

    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Statistical results file not found: {input_file}"
        )

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as file:

        results = json.load(file)

    if not isinstance(results, list):
        raise ValueError(
            "Statistical results JSON must contain a list."
        )

    return results


# ============================================================
# FORMATTING
# ============================================================

def format_p_value(value):

    if value is None:
        return "not available"

    try:
        value = float(value)

    except (
        TypeError,
        ValueError
    ):

        return str(value)

    if value < 0.0001:
        return "< 0.0001"

    return f"{value:.4f}"


def format_effect(
    value,
    symbol=""
):

    if value is None:
        return None

    try:

        return (
            f"{symbol}{float(value):.3f}"
        )

    except (
        TypeError,
        ValueError
    ):

        return str(value)


# ============================================================
# CORRELATION STRENGTH
# ============================================================

def correlation_strength(
    correlation
):

    if correlation is None:
        return "unknown"

    absolute = abs(
        float(correlation)
    )

    if absolute < 0.10:
        return "negligible"

    if absolute < 0.30:
        return "weak"

    if absolute < 0.50:
        return "moderate"

    if absolute < 0.70:
        return "strong"

    return "very strong"


def correlation_direction(
    correlation
):

    if correlation is None:
        return "unknown"

    if correlation > 0:
        return "positive"

    if correlation < 0:
        return "negative"

    return "zero"


# ============================================================
# EFFECT SIZE INTERPRETATION
# ============================================================

def interpret_cohens_d(
    value
):

    if value is None:
        return None

    absolute = abs(
        float(value)
    )

    if absolute < 0.20:
        size = "very small"

    elif absolute < 0.50:
        size = "small"

    elif absolute < 0.80:
        size = "moderate"

    else:
        size = "large"

    return (
        f"Cohen's d indicates a "
        f"{size} standardized group difference "
        f"(d = {float(value):.3f})."
    )


def interpret_eta_squared(
    value
):

    if value is None:
        return None

    value = float(value)

    if value < 0.01:
        size = "very small"

    elif value < 0.06:
        size = "small"

    elif value < 0.14:
        size = "moderate"

    else:
        size = "large"

    return (
        f"Eta-squared indicates a "
        f"{size} proportion of variance "
        f"associated with group membership "
        f"(η² = {value:.3f})."
    )


def interpret_epsilon_squared(
    value
):

    if value is None:
        return None

    value = float(value)

    if value < 0.01:
        size = "very small"

    elif value < 0.08:
        size = "small"

    elif value < 0.26:
        size = "moderate"

    else:
        size = "large"

    return (
        f"Epsilon-squared indicates a "
        f"{size} group-associated effect "
        f"(ε² = {value:.3f})."
    )


def interpret_cramers_v(
    value
):

    if value is None:
        return None

    value = float(value)

    if value < 0.10:
        size = "negligible"

    elif value < 0.30:
        size = "weak"

    elif value < 0.50:
        size = "moderate"

    else:
        size = "strong"

    return (
        f"Cramér's V indicates a "
        f"{size} association "
        f"(V = {value:.3f})."
    )


# ============================================================
# SIGNIFICANCE
# ============================================================

def significance_information(
    result
):

    raw_p = result.get(
        "p_value"
    )

    fdr_p = result.get(
        "p_value_fdr"
    )

    bonferroni_p = result.get(
        "p_value_bonferroni"
    )

    raw_significant = (
        result.get(
            "significant_raw"
        )
    )

    fdr_significant = (
        result.get(
            "significant_fdr"
        )
    )

    bonferroni_significant = (
        result.get(
            "significant_bonferroni"
        )
    )

    if raw_significant is None:

        raw_significant = (
            raw_p is not None
            and float(raw_p) < ALPHA
        )

    if fdr_significant is None:

        fdr_significant = (
            fdr_p is not None
            and float(fdr_p) < ALPHA
        )

    if bonferroni_significant is None:

        bonferroni_significant = (
            bonferroni_p is not None
            and float(bonferroni_p) < ALPHA
        )

    return {
        "raw_p_value":
            raw_p,

        "fdr_p_value":
            fdr_p,

        "bonferroni_p_value":
            bonferroni_p,

        "raw_significant":
            raw_significant,

        "fdr_significant":
            fdr_significant,

        "bonferroni_significant":
            bonferroni_significant
    }


# ============================================================
# MULTIPLE TESTING INTERPRETATION
# ============================================================

def multiple_testing_sentence(
    result
):

    info = (
        significance_information(
            result
        )
    )

    raw = info[
        "raw_significant"
    ]

    fdr = info[
        "fdr_significant"
    ]

    raw_p = info[
        "raw_p_value"
    ]

    fdr_p = info[
        "fdr_p_value"
    ]

    if (
        raw is True
        and fdr is True
    ):

        return (
            "The result is statistically "
            "significant before and after "
            "Benjamini–Hochberg FDR correction "
            f"(raw p = {format_p_value(raw_p)}, "
            f"FDR-adjusted p = "
            f"{format_p_value(fdr_p)})."
        )

    if (
        raw is True
        and fdr is False
    ):

        return (
            "The result is statistically "
            "significant at the 5% level using "
            "the unadjusted p-value, but it does "
            "not remain significant after "
            "Benjamini–Hochberg FDR correction "
            f"(raw p = {format_p_value(raw_p)}, "
            f"FDR-adjusted p = "
            f"{format_p_value(fdr_p)})."
        )

    if raw is False:

        return (
            "The result is not statistically "
            "significant at the 5% level using "
            f"the unadjusted p-value "
            f"(p = {format_p_value(raw_p)})."
        )

    return (
        "Statistical significance could not "
        "be determined from the available "
        "p-value information."
    )


# ============================================================
# ASSUMPTION WARNINGS
# ============================================================

def generate_assumption_warnings(
    result
):
    """
    Generate only test-relevant assumption warnings.

    Non-parametric tests such as Mann–Whitney U and
    Kruskal–Wallis do not require normally distributed
    groups in the same way ANOVA/t-tests do, so normality
    failures are not reported as warnings for those tests.
    """

    warnings = []

    assumptions = result.get(
        "assumptions"
    )

    if not isinstance(
        assumptions,
        dict
    ):
        return warnings

    test = str(
        result.get(
            "test",
            ""
        )
    ).lower()

    # --------------------------------------------------------
    # Tests for which normality is relevant
    # --------------------------------------------------------

    normality_relevant = (
        "anova" in test
        or "t-test" in test
        or "t test" in test
        or "student" in test
        or "welch" in test
    )

    if normality_relevant:

        normality = assumptions.get(
            "normality",
            []
        )

        for item in normality:

            label = item.get(
                "group_label",
                "Unknown group"
            )

            normal = item.get(
                "normal"
            )

            if normal is False:

                warnings.append(
                    f"Group '{label}' does not "
                    "appear normally distributed "
                    "according to the "
                    "Shapiro–Wilk test."
                )

            if item.get(
                "warning"
            ):

                warnings.append(
                    f"Group '{label}': "
                    f"{item['warning']}"
                )

    # --------------------------------------------------------
    # Equal-variance assumption is relevant to parametric
    # group comparisons. Do not flag it for Mann–Whitney
    # or Kruskal–Wallis.
    # --------------------------------------------------------

    variance_relevant = (
        normality_relevant
        or "group comparison" in test
    )

    if (
        variance_relevant
        and assumptions.get(
            "equal_variances"
        ) is False
    ):

        warnings.append(
            "The group variances do not "
            "appear equal according to "
            "Levene's test."
        )

    if (
        variance_relevant
        and assumptions.get(
            "levene_warning"
        )
    ):

        warnings.append(
            "Levene's test warning: "
            +
            str(
                assumptions[
                    "levene_warning"
                ]
            )
        )

    return warnings


# ============================================================
# PEARSON / SPEARMAN
# ============================================================

def interpret_correlation(
    result
):

    correlation = result.get(
        "correlation"
    )

    test = result.get(
        "test",
        ""
    )

    strength = (
        correlation_strength(
            correlation
        )
    )

    direction = (
        correlation_direction(
            correlation
        )
    )

    raw_p = result.get(
        "p_value"
    )

    if correlation is None:

        finding = (
            "A correlation estimate could "
            "not be calculated."
        )

    else:

        finding = (
            f"The {test} indicates a "
            f"{strength} {direction} association "
            f"between the variables "
            f"(r = {float(correlation):.3f})."
        )

    significance = (
        multiple_testing_sentence(
            result
        )
    )

    return (
        finding
        + " "
        + significance
    )


# ============================================================
# GROUP COMPARISON
# ============================================================

def interpret_group_comparison(
    result
):

    test = result.get(
        "test",
        "group comparison"
    )

    statistic = result.get(
        "statistic"
    )

    p_value = result.get(
        "p_value"
    )

    if statistic is None:

        finding = (
            "A group comparison could "
            "not be completed."
        )

    else:

        if "ANOVA" in test:

            statistic_text = (
                f"F = {float(statistic):.4f}"
            )

        elif "Kruskal" in test:

            statistic_text = (
                f"H = {float(statistic):.4f}"
            )

        else:

            statistic_text = (
                f"statistic = "
                f"{float(statistic):.4f}"
            )

        finding = (
            f"The {test} produced "
            f"{statistic_text}."
        )

    significance = (
        multiple_testing_sentence(
            result
        )
    )

    return (
        finding
        + " "
        + significance
    )


# ============================================================
# CHI-SQUARE
# ============================================================

def interpret_chi_square(
    result
):

    chi2 = result.get(
        "chi2"
    )

    if chi2 is None:

        finding = (
            "The chi-square test could "
            "not be completed."
        )

    else:

        finding = (
            "The chi-square test of "
            "independence produced "
            f"χ² = {float(chi2):.4f}."
        )

    significance = (
        multiple_testing_sentence(
            result
        )
    )

    return (
        finding
        + " "
        + significance
    )


# ============================================================
# EFFECT SIZE LIST
# ============================================================

def get_effect_size_interpretations(
    result
):

    effects = []

    cohens_d = result.get(
        "cohens_d"
    )

    eta_squared = result.get(
        "eta_squared"
    )

    epsilon_squared = result.get(
        "epsilon_squared"
    )

    cramers_v = result.get(
        "cramers_v"
    )

    interpretation = (
        interpret_cohens_d(
            cohens_d
        )
    )

    if interpretation:

        effects.append(
            interpretation
        )

    interpretation = (
        interpret_eta_squared(
            eta_squared
        )
    )

    if interpretation:

        effects.append(
            interpretation
        )

    interpretation = (
        interpret_epsilon_squared(
            epsilon_squared
        )
    )

    if interpretation:

        effects.append(
            interpretation
        )

    interpretation = (
        interpret_cramers_v(
            cramers_v
        )
    )

    if interpretation:

        effects.append(
            interpretation
        )

    return effects


# ============================================================
# MAIN DISPATCHER
# ============================================================

def interpret_result(
    result
):

    test = str(
        result.get(
            "test",
            ""
        )
    )

    variables = [
        result.get(
            "variable1"
        ),
        result.get(
            "variable2"
        )
    ]

    variables = [
        value
        for value in variables
        if value is not None
    ]

    warnings = (
        generate_assumption_warnings(
            result
        )
    )

    if result.get(
        "status"
    ) in [
        "analysis_error",
        "insufficient_data",
        "insufficient_mapped_data",
        "no_data",
        "constant_variable",
        "insufficient_categories",
        "insufficient_groups"
    ]:

        finding = (
            "This analysis could not be "
            "completed because "
            +
            str(
                result.get(
                    "status"
                )
            )
            + "."
        )

    elif (
        "Pearson" in test
        or "Spearman" in test
    ):

        finding = interpret_correlation(
            result
        )

    elif "Chi-square" in test:

        finding = interpret_chi_square(
            result
        )

    elif (
        "t-test" in test
        or "Mann" in test
        or "ANOVA" in test
        or "Kruskal" in test
    ):

        finding = interpret_group_comparison(
            result
        )

    else:

        finding = (
            "The statistical result was "
            "generated successfully."
        )

    significance = (
        significance_information(
            result
        )
    )

    return {
        "finding": finding,

        "test": test,

        "statistic":
            result.get(
                "statistic"
            ),

        "p_value":
            result.get(
                "p_value"
            ),

        "p_value_fdr":
            result.get(
                "p_value_fdr"
            ),

        "p_value_bonferroni":
            result.get(
                "p_value_bonferroni"
            ),

        "raw_significant":
            significance[
                "raw_significant"
            ],

        "fdr_significant":
            significance[
                "fdr_significant"
            ],

        "bonferroni_significant":
            significance[
                "bonferroni_significant"
            ],

        "sample_size":
            result.get(
                "sample_size"
            ),

        "group_labels":
            result.get(
                "group_labels"
            ),

        "cohens_d":
            result.get(
                "cohens_d"
            ),

        "eta_squared":
            result.get(
                "eta_squared"
            ),

        "epsilon_squared":
            result.get(
                "epsilon_squared"
            ),

        "cramers_v":
            result.get(
                "cramers_v"
            ),

        "effect_size_interpretations":
            get_effect_size_interpretations(
                result
            ),

        "warnings":
            warnings,

        "assumptions":
            result.get(
                "assumptions"
            )
    }


# ============================================================
# GENERATE ALL INTERPRETATIONS
# ============================================================

def generate_all_interpretations(
    results
):

    interpretations = []

    for index, result in enumerate(
        results,
        start=1
    ):

        variables = result.get(
            "variables"
        )

        if not variables:

            variables = [
                result.get(
                    "variable1"
                ),
                result.get(
                    "variable2"
                )
            ]

            variables = [
                value
                for value in variables
                if value is not None
            ]

        interpretations.append({

            "result_id":
                index,

            "variables":
                variables,

            "test":
                result.get(
                    "test"
                ),

            "interpretation":
                interpret_result(
                    result
                ),

            "statistical_result":
                result

        })

    return interpretations


# ============================================================
# SAVE INTERPRETATIONS
# ============================================================

def save_interpretations(
    interpretations,
    output_file=DEFAULT_OUTPUT_FILE
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

        json.dump(
            interpretations,
            file,
            indent=2,
            default=str
        )

    return output_file


# ============================================================
# MAIN
# ============================================================

def main(
    input_file=DEFAULT_INPUT_FILE,
    output_file=DEFAULT_OUTPUT_FILE
):

    print(
        "Starting interpretation engine..."
    )

    print(
        f"Statistical results file: "
        f"{input_file}"
    )

    print(
        f"Interpretation output file: "
        f"{output_file}"
    )

    results = load_results(
        input_file
    )

    print(
        f"Statistical results loaded: "
        f"{len(results)}"
    )

    interpretations = (
        generate_all_interpretations(
            results
        )
    )

    save_interpretations(
        interpretations,
        output_file
    )

    print(
        f"Interpretations saved to: "
        f"{output_file}"
    )

    print(
        f"Interpretations generated: "
        f"{len(interpretations)}"
    )

    return interpretations


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate statistical interpretations "
            "from statistical analysis results."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_FILE
        ),
        help=(
            "Path to statistical_results.json"
        )
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_FILE
        ),
        help=(
            "Path for interpretations.json"
        )
    )

    args = parser.parse_args()

    try:

        main(
            input_file=args.input,
            output_file=args.output
        )

    except Exception as error:

        print(
            "\nInterpretation engine failed."
        )

        print(
            f"Error: {error}"
        )

        raise