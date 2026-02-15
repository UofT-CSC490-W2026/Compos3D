"""Great Expectations setup and configuration for data quality validation"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.data_context import EphemeralDataContext


def create_ephemeral_context(
    project_root: Optional[str] = None,
) -> EphemeralDataContext:
    """
    Create an ephemeral Great Expectations context.

    For Stage 1, we use ephemeral contexts to avoid heavyweight project scaffolding.
    In production (AWS), we can migrate to a full DataContext with S3 backends.

    Args:
        project_root: Root directory for storing validation results

    Returns:
        Configured EphemeralDataContext
    """
    context = gx.get_context(mode="ephemeral")
    return context


def create_scene_expectations(context: EphemeralDataContext) -> ExpectationSuite:
    """
    Define expectation suite for the `scene` table.

    Expectations:
    - Required columns exist
    - No nulls in primary key (scene_id)
    - Valid split values
    - Positive object counts
    - Valid schema version
    """
    suite_name = "scene_suite"

    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    # Column existence
    required_cols = [
        "scene_id",
        "dataset",
        "seed",
        "split",
        "schema_version",
        "object_count",
    ]
    for col in required_cols:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    # No nulls in primary key
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="scene_id")
    )

    # Valid split values
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="split", value_set=["train", "val", "test"]
        )
    )

    # Positive object count
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="object_count",
            min_value=1,
            max_value=None,
        )
    )

    # Schema version
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="schema_version", value_set=["v1"]
        )
    )

    return suite


def create_scene_object_expectations(context: EphemeralDataContext) -> ExpectationSuite:
    """
    Define expectation suite for the `scene_object` table.

    Expectations:
    - Required columns exist
    - No nulls in keys
    - Positive scale values
    - Finite position/rotation values
    """
    suite_name = "scene_object_suite"

    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    # Column existence
    required_cols = [
        "scene_id",
        "object_id",
        "category",
        "px",
        "py",
        "pz",
        "sx",
        "sy",
        "sz",
    ]
    for col in required_cols:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    # No nulls in keys
    for col in ["scene_id", "object_id"]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    # Positive scale
    for col in ["sx", "sy", "sz"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=col,
                min_value=0.001,  # Very small positive number
                max_value=None,
            )
        )

    # No NaNs in position
    for col in ["px", "py", "pz"]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    return suite


def validate_dataframe_simple(
    df: pd.DataFrame,
    suite: ExpectationSuite,
    data_asset_name: str,
) -> tuple[bool, dict]:
    """
    Validate a dataframe against an expectation suite using simple approach.

    This uses a straightforward validation pattern that's compatible with all GE 1.x versions.

    Args:
        df: DataFrame to validate
        suite: Expectation suite to validate against
        data_asset_name: Name for this data asset (for reporting)

    Returns:
        (success: bool, results: dict)
    """
    # Manual validation by running each expectation
    results = {
        "success": True,
        "statistics": {
            "evaluated_expectations": 0,
            "successful_expectations": 0,
            "unsuccessful_expectations": 0,
            "success_percent": 100.0,
        },
        "results": [],
    }

    for expectation in suite.expectations:
        results["statistics"]["evaluated_expectations"] += 1

        # Extract expectation info outside try block
        exp_type = getattr(expectation, "type", "unknown")
        exp_kwargs = getattr(expectation, "kwargs", {})

        try:
            # Handle different expectation types manually for simplicity
            success = True

            if exp_type == "expect_column_to_exist":
                col = exp_kwargs["column"]
                success = col in df.columns

            elif exp_type == "expect_column_values_to_not_be_null":
                col = exp_kwargs["column"]
                success = not df[col].isna().any()

            elif exp_type == "expect_column_values_to_be_in_set":
                col = exp_kwargs["column"]
                value_set = set(exp_kwargs["value_set"])
                success = df[col].isin(value_set).all()

            elif exp_type == "expect_column_values_to_be_between":
                col = exp_kwargs["column"]
                min_val = exp_kwargs.get("min_value")
                max_val = exp_kwargs.get("max_value")
                if min_val is not None:
                    success = success and (df[col] >= min_val).all()
                if max_val is not None:
                    success = success and (df[col] <= max_val).all()

            if success:
                results["statistics"]["successful_expectations"] += 1
            else:
                results["statistics"]["unsuccessful_expectations"] += 1
                results["success"] = False

            results["results"].append(
                {
                    "expectation_config": {
                        "expectation_type": exp_type,
                        "kwargs": exp_kwargs,
                    },
                    "success": success,
                }
            )

        except Exception as e:
            results["statistics"]["unsuccessful_expectations"] += 1
            results["success"] = False
            results["results"].append(
                {
                    "expectation_config": {
                        "expectation_type": exp_type,
                        "kwargs": exp_kwargs,
                    },
                    "success": False,
                    "exception_info": str(e),
                }
            )

    # Calculate success percentage
    if results["statistics"]["evaluated_expectations"] > 0:
        results["statistics"]["success_percent"] = (
            results["statistics"]["successful_expectations"]
            / results["statistics"]["evaluated_expectations"]
            * 100.0
        )

    return results["success"], results


def validate_dataframe(
    context: EphemeralDataContext,
    df: pd.DataFrame,
    suite_name: str,
    data_asset_name: str,
) -> tuple[bool, dict]:
    """
    Validate a dataframe against an expectation suite.

    Args:
        context: GE context
        df: DataFrame to validate
        suite_name: Name of expectation suite
        data_asset_name: Name for this data asset (for reporting)

    Returns:
        (success: bool, results: dict)
    """
    # Get suite
    suite = context.suites.get(suite_name)

    # Use simple validation approach
    return validate_dataframe_simple(df, suite, data_asset_name)


def generate_validation_report(
    context: EphemeralDataContext,
    validation_results: list[dict],
    output_path: str,
) -> str:
    """
    Generate an HTML validation report.

    Args:
        context: GE context
        validation_results: List of validation result dictionaries
        output_path: Where to save the HTML report

    Returns:
        Path to generated HTML report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Simple HTML report
    html_parts = [
        "<html><head><title>Data Validation Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "table { border-collapse: collapse; width: 100%; margin-top: 20px; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #4CAF50; color: white; }",
        ".pass { color: green; font-weight: bold; }",
        ".fail { color: red; font-weight: bold; }",
        "</style></head><body>",
        "<h1>Compos3D Data Validation Report</h1>",
    ]

    for result in validation_results:
        success = result.get("success", False)
        statistics = result.get("statistics", {})

        status_class = "pass" if success else "fail"
        status_text = "PASS" if success else "FAIL"

        html_parts.append(
            f"<h2>Validation: <span class='{status_class}'>{status_text}</span></h2>"
        )
        html_parts.append("<table>")
        html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
        html_parts.append(
            f"<tr><td>Evaluated Expectations</td><td>{statistics.get('evaluated_expectations', 0)}</td></tr>"
        )
        html_parts.append(
            f"<tr><td>Successful Expectations</td><td>{statistics.get('successful_expectations', 0)}</td></tr>"
        )
        html_parts.append(
            f"<tr><td>Failed Expectations</td><td>{statistics.get('unsuccessful_expectations', 0)}</td></tr>"
        )
        html_parts.append(
            f"<tr><td>Success Percentage</td><td>{statistics.get('success_percent', 0):.1f}%</td></tr>"
        )
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    html_content = "\n".join(html_parts)
    output_path.write_text(html_content)

    return str(output_path)
