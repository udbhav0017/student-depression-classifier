# Student Depression Classifier

This supervised learning project predicts the dataset's binary `Depression` column (`0` or `1`). It is a classification project because the target is not continuous; it intentionally does **not** present a regression model.

## What it uses

- Target: `Depression`
- Features: all columns except `Depression` and row identifier `id`
- Train/test split: stratified 80/20, random seed 42
- Pipeline: median imputation for numeric data; most-frequent imputation and one-hot encoding for categorical data; balanced logistic regression
- Evaluation: accuracy, precision, recall, and ROC-AUC on the held-out test set

## Windows CMD: setup and run

Open Command Prompt, then run these commands. Replace the data path only if you move the provided dataset.

```cmd
py -m venv C:\student-ml-env
C:\student-ml-env\Scripts\activate
cd /d "C:\Users\Udbhav618\Documents\Codex\2026-10-18\referenced-chatgpt-conversation-this-is-an\outputs\student_depression_classifier"
py -m pip install --upgrade pip
pip install -r requirements.txt
python train.py --data "C:\Users\Udbhav618\Downloads\Student Depression Dataset sumit" --output-dir artifacts
set "DATASET_PATH=C:\Users\Udbhav618\Downloads\Student Depression Dataset sumit"
streamlit run app.py
```

The app opens at `http://localhost:8501`. Training creates `artifacts/depression_model.joblib`, preprocessing and model together; do not separate them.

## Deploy free on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `train.py`, `requirements.txt`, `README.md`, and the generated `artifacts/` directory. Do **not** upload sensitive source records unless you have permission.
2. On your own computer run the training command above before uploading so the model artifacts exist.
3. Go to [share.streamlit.io](https://share.streamlit.io/), sign in with GitHub, choose the repository and `app.py`, then click **Deploy**.
4. Dataset filtering requires a CSV accessible to the deployed app. For a privacy-preserving public demo, omit it: the prediction tab will still work. For a non-sensitive public dataset, add it to the repository and set `DATASET_PATH` in Streamlit's app settings to its relative/accessible path.

## Limits and responsible use

The training data and labels may contain bias, noise, missing values, and relationships that do not generalize. A probability is not a diagnosis and should never be the sole basis for clinical, disciplinary, or admission decisions. The data includes an especially sensitive self-harm-related feature; obtain consent, protect access, and involve qualified professionals if using this in any real setting. Re-evaluate performance on an independent, representative dataset before operational use.

