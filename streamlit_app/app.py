"""
Network Traffic Anomaly Detector
=================================
A Streamlit app built on top of the pipeline from notebooks/11_network_anomaly_demo.ipynb.

Trains a classifier on the embedded-system network security dataset (label: 0 = normal,
1 = anomaly), then lets a user explore the model's performance and try it on new traffic
records -- either typed in by hand or uploaded as a CSV.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
)

RANDOM_STATE = 42
DATA_PATH = "network_traffic_data.csv"
TARGET_COLUMN = "label"

st.set_page_config(page_title="Network Anomaly Detector", page_icon="🛰️", layout="wide")


# ----------------------------------------------------------------------------
# Data + model (cached so the app doesn't retrain on every click)
# ----------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_pipeline(df: pd.DataFrame):
    """Same pipeline as notebook 11: split -> scale -> baseline -> grid search -> pick best."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    tree_search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [2, 3, 4, 5, 7, 10, None], "min_samples_leaf": [1, 5, 10]},
        cv=cv, scoring="f1_weighted",
    )
    tree_search.fit(X_train_scaled, y_train)

    logreg_search = GridSearchCV(
        LogisticRegression(max_iter=5000),
        {"C": [0.01, 0.1, 1, 10, 100]},
        cv=cv, scoring="f1_weighted",
    )
    logreg_search.fit(X_train_scaled, y_train)

    candidates = {
        "Decision Tree": tree_search.best_estimator_,
        "Logistic Regression": logreg_search.best_estimator_,
    }
    final_name = max(candidates, key=lambda n: candidates[n].score(X_test_scaled, y_test))
    final_model = candidates[final_name]

    y_pred = final_model.predict(X_test_scaled)
    y_proba = final_model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
    }

    return {
        "model": final_model,
        "model_name": final_name,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
        "class_balance": y.value_counts(normalize=True).to_dict(),
    }


df = load_data()
pipeline = train_pipeline(df)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.title("Network Anomaly Detector")
st.sidebar.markdown(
    "Classifies network traffic records as **normal** or **anomalous** using a "
    "model trained on the embedded-system network security dataset."
)
st.sidebar.markdown(f"**Model in use:** {pipeline['model_name']}")
st.sidebar.markdown(f"**Training rows:** {len(df):,}")
st.sidebar.markdown(
    f"**Class balance:** {pipeline['class_balance'].get(0, 0):.0%} normal / "
    f"{pipeline['class_balance'].get(1, 0):.0%} anomaly"
)
st.sidebar.caption(
    "Built from `notebooks/11_network_anomaly_demo.ipynb` in the ml-cst9 course repo."
)

page = st.sidebar.radio("Go to", ["Overview", "Try a Prediction", "Model Performance"])


# ----------------------------------------------------------------------------
# Page: Overview
# ----------------------------------------------------------------------------

if page == "Overview":
    st.title("🛰️ Network Traffic Anomaly Detector")
    st.markdown(
        "This app trains a classifier on network traffic features (packet size, "
        "inter-arrival time, port numbers, protocol flags, etc.) to flag traffic as "
        "**normal** or **anomalous**."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Features", f"{len(pipeline['feature_columns'])}")
    col3.metric("Anomaly rate", f"{pipeline['class_balance'].get(1, 0):.1%}")

    st.subheader("Sample of the training data")
    st.dataframe(df.head(20), width="stretch")

    st.subheader("A note on this dataset")
    st.info(
        "On this particular copy of the dataset, the 17 features carry very little "
        "recoverable signal about the label (confirmed in the notebook via feature-target "
        "correlation and the ROC curve) -- the model mostly falls back to predicting the "
        "majority class. This app is a template for deploying a classifier pipeline; swap "
        "in a dataset with a stronger signal for genuinely useful anomaly detection. See "
        "**Model Performance** for the honest numbers."
    )

# ----------------------------------------------------------------------------
# Page: Try a Prediction
# ----------------------------------------------------------------------------

elif page == "Try a Prediction":
    st.title("Try a Prediction")
    st.markdown("Enter a traffic record's features, or upload a CSV of records, to classify.")

    tab1, tab2 = st.tabs(["Manual entry", "Upload CSV"])

    with tab1:
        st.markdown("Fill in the feature values below (defaults are the dataset's median).")
        feature_defaults = df[pipeline["feature_columns"]].median(numeric_only=True)

        input_values = {}
        cols = st.columns(3)
        for i, col_name in enumerate(pipeline["feature_columns"]):
            with cols[i % 3]:
                col_dtype = df[col_name].dtype
                if col_dtype == bool:
                    input_values[col_name] = st.checkbox(col_name, value=bool(df[col_name].mode()[0]))
                elif col_name in ("src_port", "dst_port"):
                    input_values[col_name] = st.number_input(
                        col_name, min_value=0, max_value=65535,
                        value=int(feature_defaults.get(col_name, 0)),
                    )
                else:
                    input_values[col_name] = st.number_input(
                        col_name, value=float(feature_defaults.get(col_name, 0.0)), format="%.4f",
                    )

        if st.button("Classify this record", type="primary"):
            record = pd.DataFrame([input_values])[pipeline["feature_columns"]]
            record_scaled = pipeline["scaler"].transform(record)
            pred = pipeline["model"].predict(record_scaled)[0]
            proba = pipeline["model"].predict_proba(record_scaled)[0, 1]

            label = "🚨 Anomaly" if pred == 1 else "✅ Normal"
            st.subheader(f"Prediction: {label}")
            st.metric("Predicted anomaly probability", f"{proba:.1%}")
            st.progress(min(max(proba, 0.0), 1.0))

    with tab2:
        st.markdown(
            f"Upload a CSV with these {len(pipeline['feature_columns'])} columns: "
            f"`{', '.join(pipeline['feature_columns'])}`"
        )
        uploaded = st.file_uploader("Upload traffic records (CSV)", type="csv")
        if uploaded is not None:
            new_df = pd.read_csv(uploaded)
            missing_cols = set(pipeline["feature_columns"]) - set(new_df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {sorted(missing_cols)}")
            else:
                X_new = new_df[pipeline["feature_columns"]]
                X_new_scaled = pipeline["scaler"].transform(X_new)
                new_df["predicted_label"] = pipeline["model"].predict(X_new_scaled)
                new_df["anomaly_probability"] = pipeline["model"].predict_proba(X_new_scaled)[:, 1]
                new_df["prediction"] = new_df["predicted_label"].map({0: "normal", 1: "anomaly"})

                n_anomalies = int((new_df["predicted_label"] == 1).sum())
                st.success(f"Classified {len(new_df)} rows -- {n_anomalies} flagged as anomalous.")
                st.dataframe(
                    new_df.drop(columns=["predicted_label"]).style.map(
                        lambda v: "background-color: #ffcccc" if v == "anomaly" else "",
                        subset=["prediction"],
                    ),
                    width="stretch",
                )
                st.download_button(
                    "Download results as CSV",
                    new_df.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv",
                )

# ----------------------------------------------------------------------------
# Page: Model Performance
# ----------------------------------------------------------------------------

elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown(
        f"Evaluated on a held-out test set (never seen during training) using the "
        f"**{pipeline['model_name']}** model."
    )

    m = pipeline["metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{m['accuracy']:.3f}")
    col2.metric("Precision", f"{m['precision']:.3f}")
    col3.metric("Recall", f"{m['recall']:.3f}")
    col4.metric("F1 Score", f"{m['f1']:.3f}")
    col5.metric("AUC", f"{m['auc']:.3f}")

    majority_rate = max(pipeline["class_balance"].values())
    if abs(m["accuracy"] - majority_rate) < 0.01 and m["f1"] < 0.1:
        st.warning(
            "Accuracy matches the majority-class baseline almost exactly, and F1 is near "
            "zero -- this model isn't beating the laziest possible guess ('always predict "
            "normal'). See the confusion matrix below for the full picture."
        )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(pipeline["y_test"], pipeline["y_pred"])
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "anomaly"])
        disp.plot(cmap="Blues", ax=ax_cm, colorbar=False)
        st.pyplot(fig_cm)

    with col_b:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(pipeline["y_test"], pipeline["y_proba"])
        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4))
        ax_roc.plot(fpr, tpr, label=f"model (AUC = {m['auc']:.3f})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random guessing")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc)

    st.subheader("Sample test-set predictions")
    sample_idx = pipeline["X_test"].index[:10]
    sample_display = pipeline["X_test"].loc[sample_idx].copy()
    sample_display["true_label"] = pipeline["y_test"].loc[sample_idx].map({0: "normal", 1: "anomaly"})
    sample_display["predicted_label"] = pd.Series(pipeline["y_pred"], index=pipeline["X_test"].index).loc[sample_idx].map({0: "normal", 1: "anomaly"})
    st.dataframe(sample_display, width="stretch")
