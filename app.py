# ML Assignment 2 - Streamlit app
# Parth Kaushik (2026AJ05016), M.Tech AIML
# Loads the 5 models trained in model/train_models.ipynb and evaluates them on the
# uploaded test csv. Models are only loaded here, no training happens in this file.

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
METRIC_INFO = {
    "Accuracy": "Fraction of sessions predicted correctly",
    "AUC": "Area under ROC curve",
    "Precision": "Out of predicted buyers, how many actually bought",
    "Recall": "Out of actual buyers, how many were caught",
    "F1": "Harmonic mean of precision and recall",
    "MCC": "Matthews correlation coefficient, ranges from -1 to 1",
}

st.set_page_config(page_title="ML Assignment 2", page_icon="🛒", layout="wide")


@st.cache_resource
def load_models():
    """Loads the metadata, the 5 saved models and the metrics noted during training."""
    with open(MODEL_DIR / "model_metadata.json") as f:
        meta = json.load(f)
    models = {}
    for name, file_name in meta["model_files"].items():
        models[name] = joblib.load(MODEL_DIR / file_name)
    train_time_metrics = pd.read_csv(MODEL_DIR / "metrics_summary.csv")
    return meta, models, train_time_metrics


@st.cache_data
def read_csv(file_bytes):
    """Reads the uploaded csv bytes into a dataframe."""
    return pd.read_csv(io.BytesIO(file_bytes))


def convert_target(col):
    """Converts the target column to 0/1, works for True/False, 1/0 or yes/no."""
    if col.dtype == bool:
        return col.astype(int)
    mapping = {"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0}
    converted = col.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        wrong = col[converted.isna()].unique()[:5]
        raise ValueError(f"Target column has values I can't read: {list(wrong)}")
    return converted.astype(int)


@st.cache_data(show_spinner="Running the model on test data...")
def evaluate(model_name, X, y):
    """Runs the model on X and returns predictions, probabilities and all 6 metrics."""
    model = load_models()[1][model_name]
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    res = {
        "pred": pred,
        "prob": prob,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    # AUC can't be calculated if the file has only one class
    res["AUC"] = roc_auc_score(y, prob) if y.nunique() > 1 else np.nan
    return res


def confusion_matrix_plot(y, pred, labels):
    """Draws the confusion matrix with counts and row-wise percentages."""
    cm = confusion_matrix(y, pred, labels=[0, 1])
    row_total = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_total, out=np.zeros(cm.shape), where=row_total > 0)
    text = [[f"{cm[i, j]}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=np.array(text), fmt="", cmap="Greens", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"fontsize": 11})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def roc_plot(y, prob, auc, name):
    """Plots the ROC curve of the selected model along with the random baseline."""
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(fpr, tpr, color="#1F6F5C", lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def comparison_plot(df):
    """Bar chart comparing all 5 models on every metric."""
    long_df = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=long_df, x="Metric", y="Score", hue="Model", palette="Set2", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


@st.fragment
def predictions_table(out):
    """Shows the prediction table, the toggle only reruns this part and not the whole page."""
    show_wrong = st.toggle("Show only wrong predictions")
    st.dataframe(out[~out["Correct"]] if show_wrong else out, width="stretch", height=420)


meta, models, train_time_metrics = load_models()
features = meta["feature_columns"]
target = meta["target"]
labels = [meta["class_labels"]["0"], meta["class_labels"]["1"]]

st.title("Online Shoppers Purchase Prediction")
st.write("ML Assignment 2 (M.Tech AIML). Predicts if an online shopping session ends in a "
         "purchase, using the UCI Online Shoppers Purchasing Intention dataset. "
         "Upload the test csv, select a model from the sidebar and check the results.")

with st.sidebar:
    st.header("Test data")
    uploaded_file = st.file_uploader(
        "Upload test CSV", type=["csv"],
        help=f"File should have all {len(features)} feature columns and the '{target}' column.")
    use_sample = st.checkbox("Click to use sample test data (only for dev testing)")

    st.header("Model")
    selected_model = st.selectbox("Select model", list(models.keys()), index=4)
    with st.expander("Best hyperparameters (from grid search)", expanded=True):
        st.json(meta["best_params"][selected_model])

    st.caption("Models were trained in the notebook on 80% of the data using 5-fold "
               "GridSearchCV. They are only loaded here, not retrained.")

if uploaded_file is not None:
    data = read_csv(uploaded_file.getvalue())
    file_label = uploaded_file.name
elif use_sample:
    data = read_csv(SAMPLE_CSV.read_bytes())
    file_label = "sample test_data.csv"
else:
    st.info("Please upload the test CSV (test_data.csv from the GitHub repo) from the sidebar, "
            "or tick the sample data checkbox below the upload button to try it out.")
    st.stop()

missing_cols = [c for c in features + [target] if c not in data.columns]
if missing_cols:
    st.error("These columns are missing in the uploaded file: " + ", ".join(missing_cols))
    st.stop()

try:
    y_true = convert_target(data[target])
except ValueError as e:
    st.error(str(e))
    st.stop()
X = data[features]

st.caption(f"Using **{file_label}**: {len(data)} rows, {y_true.mean():.1%} of them are purchases")

tab1, tab2, tab3, tab4 = st.tabs(["Model evaluation", "Compare all models",
                                  "Predictions", "About dataset"])

with tab1:
    res = evaluate(selected_model, X, y_true)
    st.subheader(f"Evaluation metrics - {selected_model}")
    metric_cols = st.columns(6)
    for col, m in zip(metric_cols, METRIC_NAMES):
        val = res[m]
        col.metric(m, "n/a" if np.isnan(val) else f"{val:.4f}", help=METRIC_INFO[m])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion matrix**")
        st.pyplot(confusion_matrix_plot(y_true, res["pred"], labels))
    with c2:
        st.markdown("**ROC curve**")
        if np.isnan(res["AUC"]):
            st.warning("ROC curve needs both classes in the uploaded data.")
        else:
            st.pyplot(roc_plot(y_true, res["prob"], res["AUC"], selected_model))

    st.markdown("**Classification report**")
    report = classification_report(y_true, res["pred"], labels=[0, 1], target_names=labels,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

with tab2:
    st.subheader("All 5 models on the selected test data")
    rows = []
    for name in models:
        r = evaluate(name, X, y_true)
        rows.append({"Model": name, **{m: r[m] for m in METRIC_NAMES}})
    compare_df = pd.DataFrame(rows)

    st.dataframe(compare_df.set_index("Model").style.format("{:.4f}")
                 .highlight_max(axis=0, color="#CDEBDD"), width="stretch")
    st.pyplot(comparison_plot(compare_df))

    best_mcc = compare_df.loc[compare_df["MCC"].idxmax(), "Model"]
    best_f1 = compare_df.loc[compare_df["F1"].idxmax(), "Model"]
    st.success(f"Best MCC: {best_mcc} | Best F1: {best_f1} (green cell = best value in that column)")

    with st.expander("Metrics noted at training time (same as README table)"):
        st.dataframe(train_time_metrics.set_index("ML Model Name"), width="stretch")

with tab3:
    st.subheader(f"Predictions - {selected_model}")
    if selected_model == "Support Vector Machine":
        st.caption("Note: for SVM the predicted class comes from the decision function, while "
                   "P(Purchase) comes from Platt scaling, so for a few rows close to the boundary "
                   "they may not agree with a 0.5 cut-off.")
    out = data.copy()
    out["Predicted"] = np.where(res["pred"] == 1, labels[1], labels[0])
    out["P(Purchase)"] = res["prob"].round(4)
    out["Correct"] = res["pred"] == y_true.to_numpy()

    predictions_table(out)

with tab4:
    st.markdown("""
**Source:** UCI Machine Learning Repository, Online Shoppers Purchasing Intention Dataset
(Sakar et al., 2019). Each row is one session on an online store, collected over one year.
`Revenue` is the target and tells whether the session ended in a purchase.

| Group | Columns |
|---|---|
| Pages visited and time spent | Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration |
| Google Analytics values | BounceRates, ExitRates, PageValues |
| Date related | SpecialDay, Month, Weekend |
| Visitor / technical | VisitorType, OperatingSystems, Browser, Region, TrafficType |
| Target | Revenue (True = purchase) |

**Preprocessing (saved inside each model pipeline):** log1p + StandardScaler for the skewed
count, duration and PageValues columns, StandardScaler for the rate columns, and one-hot
encoding for Month, VisitorType, Weekend, OperatingSystems, Browser, Region and TrafficType.
""")
    st.markdown("**First 20 rows of the loaded data**")
    st.dataframe(data.head(20), width="stretch")
