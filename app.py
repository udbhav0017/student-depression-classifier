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
MODEL_PATH = ARTIFACTS / "depression_model.joblib"
DEPLOYED_MODEL_PATH = ROOT / "depression_model.joblib"
DATASET_NAME = "Student_Depression_Dataset.csv"


@st.cache_resource
def load_model():
    path = MODEL_PATH if MODEL_PATH.exists() else DEPLOYED_MODEL_PATH
    return joblib.load(path)


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


@st.cache_data
def load_dataset(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text)


def find_dataset_path(metadata: dict) -> Path | None:
    """Return the first existing dataset CSV (env var, repo folders, then metadata path)."""
    candidates: list[Path] = []
    env_path = os.getenv("DATASET_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates += [ROOT / "data" / DATASET_NAME, ARTIFACTS / DATASET_NAME, ROOT / DATASET_NAME]
    if (ROOT / "data").exists():
        candidates += sorted((ROOT / "data").glob("*.csv"))
    source_file = metadata.get("source_file", "")
    if source_file:
        candidates.append(Path(source_file))
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


st.title("Student Depression Predictor")
if not MODEL_PATH.exists() and not DEPLOYED_MODEL_PATH.exists():
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
                values[feature] = st.selectbox(feature, details["options"], key=f"input_{feature}")
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
    dataset: pd.DataFrame | None = None
    dataset_path = find_dataset_path(metadata)
    if dataset_path is not None:
        try:
            dataset = load_dataset(str(dataset_path))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            st.warning(f"Could not read {dataset_path.name}: {error}")
    else:
        st.info(
            f"`{DATASET_NAME}` was not found in the deployed repository. Add it as "
            f"`data/{DATASET_NAME}` in GitHub, or upload it below to search it now."
        )
        uploaded = st.file_uploader("Upload the dataset CSV", type="csv")
        if uploaded is not None:
            try:
                dataset = pd.read_csv(uploaded)
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
                st.error(f"Could not read the uploaded file: {error}")

    if dataset is not None:
        filter_column = st.selectbox("Column to filter", dataset.columns.tolist())
        raw_values = dataset[filter_column].dropna().astype(str).unique().tolist()
        chosen_values = st.multiselect("Keep values", sorted(raw_values), default=[])
        query_text = st.text_input("Optional text search across all columns")
        filtered = dataset.copy()
        if chosen_values:
            filtered = filtered[filtered[filter_column].astype(str).isin(chosen_values)]
        if query_text:
            found = filtered.astype(str).apply(lambda c: c.str.contains(query_text, case=False, na=False, regex=False))
            filtered = filtered[found.any(axis=1)]
        st.write(f"Showing {len(filtered):,} of {len(dataset):,} rows")
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button("Download filtered rows as CSV", filtered.to_csv(index=False), "filtered_students.csv", "text/csv")

with about_tab:
    st.json(load_json("metrics.json"))
    st.write("The model is logistic regression with median/mode imputation and one-hot encoding for categorical columns. It excludes the dataset `id` column.")
