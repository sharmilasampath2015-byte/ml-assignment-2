import streamlit as st

st.set_page_config(page_title="ML Classification Models", layout="wide")

st.title("🧬 Breast Cancer Classification - ML Models")

st.markdown("""
## Project Overview

This application compares 6 different machine learning classification models trained on the 
**Breast Cancer Wisconsin (Diagnostic)** dataset.

### Models Implemented

1. **Logistic Regression** — Linear baseline model
2. **Decision Tree Classifier** — Rule-based classification
3. **K-Nearest Neighbors (KNN)** — Instance-based learning
4. **Naive Bayes Classifier** — Probabilistic model
5. **Random Forest** — Ensemble of decision trees
6. **XGBoost** — Gradient boosting ensemble

### Evaluation Metrics

For each model, the following 6 metrics are calculated:
- **Accuracy** — Overall correctness
- **AUC Score** — Area under ROC curve
- **Precision** — True positive rate among predicted positives
- **Recall** — True positive rate among actual positives
- **F1 Score** — Harmonic mean of precision and recall
- **MCC** — Matthews Correlation Coefficient

### Key Results

| Metric | Best Model | Score |
|--------|-----------|-------|
| Accuracy | XGBoost | 0.9737 |
| AUC | Logistic Regression | 0.9960 |
| F1 Score | XGBoost | 0.9630 |
| Precision | XGBoost, Random Forest | 1.0000 |

### Recommendation

**XGBoost** is recommended for deployment with:
- Highest accuracy (97.37%)
- Perfect precision (1.0)
- Excellent AUC (0.994)
- Best F1-score (0.963)

---

### How to Use

1. Navigate to the **Streamlit App** using the main deployment link
2. Upload your test dataset (CSV format)
3. Select a model from the sidebar
4. View detailed metrics and make predictions

### Project Structure

```
ml-assignment-2/
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── evaluation_metrics.csv
└── notebooks/
    └── BC_Cancer_Classification.ipynb
```

---

**For detailed analysis, metrics, and observations, see the README.md file or access the full Streamlit application.**
""")

st.info("👉 Click 'Metrics Comparison' in the main app to view detailed model performance comparison.")
