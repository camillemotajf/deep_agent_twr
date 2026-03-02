import pandas as pd
import json
import ast
from typing import List


# =====================================================
# Safe parsing for dict-like columns
# =====================================================
def parse_dict_col(val):
    """
    Safely parse a value that may represent a dictionary.
    Accepts dict, JSON string, Python literal string, or NaN.
    """
    if isinstance(val, dict):
        return val
    if pd.isna(val):
        return {}
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return {}


# =====================================================
# Expand dictionary column ensuring all keys exist
# Missing keys are filled with "absent"
# =====================================================
def expand_with_absent(df: pd.DataFrame, dict_column: str) -> pd.DataFrame:
    """
    Expands a dictionary column into multiple columns.
    Ensures that all possible keys appear in every row,
    filling missing values with 'absent'.
    """
    dicts = df[dict_column].apply(parse_dict_col)

    all_keys = set().union(*dicts)

    expanded = pd.DataFrame([
        {key: d.get(key, "absent") for key in all_keys}
        for d in dicts
    ])

    return expanded


# =====================================================
# Create ML metadata (class and error type)
# =====================================================
def create_ml_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates classification metadata for ML results:
    - class: bot / unsafe
    - error_type: correct / FP / FN
    """
    meta = df[["pred", "target", "is_error"]].copy()

    meta["class"] = meta["pred"].map({1: "bot", 0: "unsafe"})
    meta["error_type"] = "correct"

    meta.loc[
        (meta["target"] == 0) & (meta["pred"] == 1),
        "error_type"
    ] = "FP"

    meta.loc[
        (meta["target"] == 1) & (meta["pred"] == 0),
        "error_type"
    ] = "FN"

    return meta


# =====================================================
# Convert wide dataframe to long format
# =====================================================
def to_long_format(df: pd.DataFrame, meta_columns: List[str]) -> pd.DataFrame:
    """
    Converts expanded dataframe into long format:
    one row per (key, value).
    """
    feature_columns = [c for c in df.columns if c not in meta_columns]

    return df.melt(
        id_vars=meta_columns,
        value_vars=feature_columns,
        var_name="key",
        value_name="value"
    )


# =====================================================
# Full preparation pipeline (headers + URL params)
# =====================================================
def prepare_dataset(
    df: pd.DataFrame,
    source: str,
    is_ml: bool = False,
    class_column: str = None
) -> pd.DataFrame:
    """
    Prepares dataset for frequency analysis.
    Expands headers and URL params, attaches metadata.
    """

    headers_df = expand_with_absent(df, "headers")
    headers_df["feature_type"] = "header"

    params_df = expand_with_absent(df, "request")
    params_df["feature_type"] = "param"

    data = pd.concat([headers_df, params_df], ignore_index=True)

    if is_ml:
        meta = create_ml_metadata(df)
        meta = pd.concat([meta, meta], ignore_index=True)
        data = data.join(meta)
    else:
        data["class"] = df[class_column].values.repeat(2)
        data["error_type"] = "database"

    data["source"] = source
    return data


# =====================================================
# Compute absolute and percentage frequencies
# =====================================================
def compute_frequencies(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Computes absolute and relative frequencies
    for each (key, value).
    """
    freq = (
        df_long
        .groupby(
            ["source", "feature_type", "class", "error_type", "key", "value"],
            dropna=False
        )
        .size()
        .reset_index(name="count")
    )

    totals = (
        df_long
        .groupby(
            ["source", "feature_type", "class", "error_type", "key"],
            dropna=False
        )
        .size()
        .reset_index(name="total")
    )

    freq = freq.merge(
        totals,
        on=["source", "feature_type", "class", "error_type", "key"]
    )

    freq["percentage"] = (freq["count"] / freq["total"]) * 100
    return freq


# =====================================================
# High-level API function
# =====================================================
def analyze_frequencies(
    df_analysis: pd.DataFrame,
    df_database: pd.DataFrame
) -> pd.DataFrame:
    """
    High-level function to compare feature frequencies
    between ML analysis data and database context.
    """

    ml_data = prepare_dataset(
        df_analysis,
        source="ml",
        is_ml=True
    )

    db_data = prepare_dataset(
        df_database,
        source="database",
        is_ml=False,
        class_column="decision"
    )

    combined = pd.concat([ml_data, db_data], ignore_index=True)

    meta_cols = ["source", "feature_type", "class", "error_type"]
    combined_long = to_long_format(combined, meta_cols)

    return compute_frequencies(combined_long)