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
    if (ROOT /
