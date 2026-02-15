# Breast Cancer Wisconsin (Diagnostic) — ML Classification Project

**a. Problem statement**

The task is to build and compare six supervised classification models to predict whether a breast tumor is malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to train, evaluate, save, and deploy models while reporting a standardized set of evaluation metrics and observations.

**b. Dataset description**

- Source: UCI / Kaggle (Breast Cancer Wisconsin (Diagnostic))
- File used: `data/data.csv`
- Instances: 569
- Features: 30 numeric features (measurements for mean, standard error and "worst" for 10 characteristics) + `id` + `diagnosis` (target)
- Target: `diagnosis` — binary (M = malignant, B = benign)
- Missing values: none expected in original CSV; any missing values during preprocessing are imputed using the median strategy.

Quick summary:
- Total instances: 569
- Total predictive features used: 30 (all numeric)
- Class distribution: Benign (B) ≈ 62.7%, Malignant (M) ≈ 37.3%


**c. Models used & evaluation metrics**

Implemented models (all trained on the same dataset):
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Naive Bayes (Gaussian)
- Random Forest (Ensemble)
- XGBoost (Ensemble)

Evaluation metrics calculated for each model (report these numeric values in the table below):
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


**Comparison Table**

| ML Model Name       | Accuracy | AUC    | Precision | Recall   | F1      | MCC    |
|---------------------|----------|--------|-----------|----------|---------|--------|
| Logistic Regression | 0.964912 | 0.996032 | 0.975000 | 0.928571 | 0.951220 | 0.924518 |
| Decision Tree       | 0.929825 | 0.924603 | 0.904762 | 0.904762 | 0.904762 | 0.849206 |
| kNN                 | 0.956140 | 0.982308 | 0.974359 | 0.904762 | 0.938272 | 0.905824 |
| Naive Bayes         | 0.921053 | 0.989087 | 0.923077 | 0.857143 | 0.888889 | 0.829162 |
| Random Forest       | 0.964912 | 0.993882 | 1.000000 | 0.904762 | 0.950000 | 0.925320 |
| XGBoost             | 0.973684 | 0.994048 | 1.000000 | 0.928571 | 0.962963 | 0.944155 |


**Model observations** (brief, one or two lines per model) — include these after you run experiments and fill with observed behaviour: 

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Strong baseline model with 96.49% accuracy and excellent AUC (0.996). Good balance between precision (0.975) and recall (0.929), making it reliable for both false positives and false negatives. Linear decision boundary works well for this dataset. |
| Decision Tree       | Lowest performance (92.98% accuracy) with lowest MCC (0.849), indicating overfitting tendencies. Lower recall (0.905) and precision (0.905) suggest it may struggle with minority class patterns. Simple decision rules are insufficient for this complex biomedical problem. |
| kNN                 | Competitive performance (95.61% accuracy) with high precision (0.974) and AUC (0.982). Effective after feature scaling. Sensitive to local neighborhood density, but performs well when neighbors are well-distributed in scaled feature space. |
| Naive Bayes         | Good AUC (0.989) but lower accuracy (92.11%) due to the conditional independence assumption not holding for highly correlated features. Decent recall (0.857) suggests it captures malignant cases reasonably well despite feature correlations. |
| Random Forest       | Excellent performer (96.49% accuracy) with perfect precision (1.0) and high AUC (0.994). Ensemble approach effectively captures non-linear patterns; robust to feature interactions. High MCC (0.925) indicates very good overall classification quality. |
| XGBoost             | Best overall model (97.37% accuracy, 0.994 AUC, perfect 1.0 precision, 0.963 F1-score, 0.944 MCC). Gradient boosting captures complex patterns most effectively. Superior performance on recall-precision tradeoff makes it most suitable for clinical deployment. |


**How to run (local / Virtual Lab)**

- Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

- Run the Jupyter notebook to train models and produce evaluation outputs:

```bash
jupyter notebook notebooks/BC_Cancer_Classification.ipynb
# or open the notebook in your Virtual Lab environment and run all cells
```

- The notebook saves:
  - Trained model pickles to `models/` (or `model/` depending on final path; ensure folder exists)
  - `models/evaluation_metrics.csv` with the comparison table
  - Plots in `models/` as PNG files


**Streamlit app / Deployment (required)**

- App entry: `streamlit_app.py` (or `app.py`)
- To run locally:

```bash
streamlit run streamlit_app.py
```

- Deploy on Streamlit Community Cloud: push repository to GitHub, then create a new app on https://streamlit.io/cloud and point it to your `streamlit_app.py`.


