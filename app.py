"""Streamlit prediction and dataset-search interface."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Student Depression Predictor", page_icon="🧠", layout="wide")
ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"


@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS / "depression_model.joblib")


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


@st.cache_data
def load_dataset(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text)


st.title("Student Depression Predictor")
if not (ARTIFACTS / "depression_model.joblib").exists():
    st.error("No trained model found. Run train.py first; see README.md.")
    st.stop()

model = load_model()
metadata = load_json("metadata.json")
schema = load_json("feature_schema.json")

predict_tab, search_tab, about_tab = st.tabs(["Prediction", "Search dataset", "About"])

with predict_tab:
    st.write("Enter the student details below. The result is a model estimate, not a medical diagnosis.")
    values: dict[str, object] = {}
    left, right = st.columns(2)
    for index, feature in enumerate(metadata["features"]):
        container = left if index % 2 == 0 else right
        details = schema[feature]
        with container:
            if details["kind"] == "numeric":
                values[feature] = st.number_input(
                    feature,
                    min_value=float(details["min"]),
                    max_value=float(details["max"]),
                    value=float(details["median"]),
                )
            else:
                options = details["options"]
                values[feature] = st.selectbox(feature, options, key=f"input_{feature}")
    if st.button("Estimate depression risk", type="primary"):
        input_frame = pd.DataFrame([values], columns=metadata["features"])
        probability = float(model.predict_proba(input_frame)[0, 1])
        prediction = int(model.predict(input_frame)[0])
        if prediction == 1:
            st.warning(f"Estimated probability of the dataset's `Depression = 1` label: **{probability:.1%}**")
        else:
            st.success(f"Estimated probability of the dataset's `Depression = 1` label: **{probability:.1%}**")
        st.caption("This screening-style output cannot diagnose depression. If someone may be at risk of self-harm, contact local emergency services or a qualified mental-health professional now.")

with search_tab:
    source_path = os.getenv("DATASET_PATH", metadata.get("source_file", ""))
    st.caption("Set the optional DATASET_PATH environment variable to point the deployed app to a CSV.")
    try:
        dataset = load_dataset(source_path)
        filter_column = st.selectbox("Column to filter", dataset.columns.tolist())
        raw_values = dataset[filter_column].dropna().astype(str).unique().tolist()
        chosen_values = st.multiselect("Keep values", sorted(raw_values), default=[])
        query_text = st.text_input("Optional text search across all columns")
        filtered = dataset.copy()
        if chosen_values:
            filtered = filtered[filtered[filter_column].astype(str).isin(chosen_values)]
        if query_text:
            found = filtered.astype(str).apply(lambda column: column.str.contains(query_text, case=False, na=False))
            filtered = filtered[found.any(axis=1)]
        st.write(f"Showing {len(filtered):,} of {len(dataset):,} rows")
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button("Download filtered rows as CSV", filtered.to_csv(index=False), "filtered_students.csv", "text/csv")
    except (FileNotFoundError, PermissionError, pd.errors.ParserError) as error:
        st.info(f"Dataset search is unavailable: {error}")

with about_tab:
    metrics = load_json("metrics.json")
    st.json(metrics)
    st.write("The model is logistic regression with median/mode imputation and one-hot encoding for categorical columns. It excludes the dataset `id` column.")

