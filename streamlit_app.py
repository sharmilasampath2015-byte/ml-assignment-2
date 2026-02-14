import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="BC Cancer Classification - ML Models",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Breast Cancer Classification - ML Models Comparison")
st.markdown("---")

# Load models and scaler
@st.cache_resource
def load_models_and_data():
    models = {}
    model_names = ['logistic_regression', 'decision_tree', 'knn', 'naive_bayes', 'random_forest', 'xgboost']
    
    for name in model_names:
        try:
            model_path = f'models/{name}.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load {name} model: {e}")
    
    try:
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = None
    except Exception as e:
        st.warning(f"Could not load scaler: {e}")
        scaler = None
    
    # Load metrics CSV
    try:
        metrics_path = 'models/evaluation_metrics.csv'
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path, index_col=0)
        else:
            metrics_df = None
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
        metrics_df = None
    
    return models, scaler, metrics_df

models, scaler, metrics_df = load_models_and_data()

# Sidebar
with st.sidebar:
    st.header("📊 Configuration")
    
    st.subheader("Dataset Upload")
    uploaded_file = st.file_uploader("Choose a CSV file (test data)", type=['csv'])
    
    if uploaded_file is not None:
        st.success(f"File loaded: {uploaded_file.name}")
    
    st.subheader("🎯 Model Selection")
    model_names_display = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'knn': 'KNN',
        'naive_bayes': 'Naive Bayes',
        'random_forest': 'Random Forest (Ensemble)',
        'xgboost': 'XGBoost (Ensemble)'
    }
    
    selected_model_display = st.selectbox(
        "Select a classification model:",
        options=list(model_names_display.values())
    )
    
    selected_model = [k for k, v in model_names_display.items() if v == selected_model_display][0]

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics Comparison", "Model Details", "Dataset Info", "About"])

with tab1:
    st.subheader("Evaluation Metrics Comparison - All 6 Models")
    
    if metrics_df is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Models", len(metrics_df))
            st.metric("🏆 Best Accuracy", f"{metrics_df['Accuracy'].max():.4f}")
        
        with col2:
            best_model_acc = metrics_df['Accuracy'].idxmax()
            st.metric("⭐ Best Model (Accuracy)", best_model_acc)
            st.metric("Top AUC Score", f"{metrics_df['AUC'].max():.4f}")
        
        with col3:
            st.metric("Avg Accuracy", f"{metrics_df['Accuracy'].mean():.4f}")
            st.metric("Avg F1 Score", f"{metrics_df['F1'].mean():.4f}")
        
        st.markdown("---")
        st.write("**Full Metrics Table:**")
        st.dataframe(metrics_df.style.format("{:.6f}"), use_container_width=True)
        
        # Visualization: Metrics comparison
        st.markdown("---")
        st.write("**Metrics Comparison Visualization:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(metrics_df.index))
            width = 0.13
            
            metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, metric in enumerate(metrics):
                ax.bar(x + i*width, metrics_df[metric], width, label=metric, color=colors[i])
            
            ax.set_xlabel('Models', fontweight='bold')
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title('All 6 Evaluation Metrics Comparison', fontweight='bold', fontsize=12)
            ax.set_xticks(x + width * 2.5)
            ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
            ax.legend(loc='lower right', fontsize=9)
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_df['Accuracy'].sort_values(ascending=False).plot(
                kind='barh', ax=ax, color='steelblue'
            )
            ax.set_xlabel('Accuracy Score', fontweight='bold')
            ax.set_title('Accuracy Ranking', fontweight='bold', fontsize=12)
            ax.set_xlim([0.85, 1.0])
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("⚠️ Metrics CSV not found. Please ensure models have been trained and saved.")

with tab2:
    st.subheader(f"📋 {selected_model_display} - Model Details")
    
    if metrics_df is not None and selected_model in metrics_df.index:
        col1, col2, col3 = st.columns(3)
        
        metrics_for_model = metrics_df.loc[selected_model]
        
        with col1:
            st.metric("Accuracy", f"{metrics_for_model['Accuracy']:.4f}")
            st.metric("Precision", f"{metrics_for_model['Precision']:.4f}")
        
        with col2:
            st.metric("AUC Score", f"{metrics_for_model['AUC']:.4f}")
            st.metric("Recall", f"{metrics_for_model['Recall']:.4f}")
        
        with col3:
            st.metric("F1 Score", f"{metrics_for_model['F1']:.4f}")
            st.metric("MCC", f"{metrics_for_model['MCC']:.4f}")
        
        st.markdown("---")
        
        # Model observations
        observations = {
            'logistic_regression': "Strong baseline model with excellent AUC (0.996) and good balance between precision and recall. Linear decision boundary works well for this dataset.",
            'decision_tree': "Lowest performance with some overfitting tendencies. Simple decision rules insufficient for complex biomedical patterns.",
            'knn': "Competitive performance with high precision after feature scaling. Effective when neighbors are well-distributed in scaled feature space.",
            'naive_bayes': "Good AUC despite lower accuracy due to conditional independence assumption. Reasonable recall for malignant case detection.",
            'random_forest': "Excellent performer with perfect precision and high AUC. Robust ensemble approach captures non-linear patterns effectively.",
            'xgboost': "Best overall model with highest accuracy and F1-score. Superior performance on recall-precision tradeoff makes it most suitable for clinical deployment."
        }
        
        st.write("**Model Performance Observation:**")
        st.info(observations.get(selected_model, "Model information unavailable."))
    
    # Upload test data and make predictions
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("🔮 Make Predictions on Test Data")
        
        try:
            test_df = pd.read_csv(uploaded_file)
            st.write(f"Test data shape: {test_df.shape}")
            st.dataframe(test_df.head())
            
            if selected_model in models:
                model = models[selected_model]
                
                # Prepare data (drop non-numeric columns if any)
                test_features = test_df.select_dtypes(include=[np.number])
                
                if selected_model in ['logistic_regression', 'knn', 'naive_bayes']:
                    if scaler is not None:
                        test_features_scaled = scaler.transform(test_features)
                        predictions = model.predict(test_features_scaled)
                        predictions_proba = model.predict_proba(test_features_scaled)
                    else:
                        st.warning("Scaler not available for scaling test data.")
                else:
                    predictions = model.predict(test_features)
                    predictions_proba = model.predict_proba(test_features)
                
                st.write("**Predictions:**")
                pred_df = pd.DataFrame({
                    'Prediction (0=Benign, 1=Malignant)': predictions,
                    'Probability Benign': predictions_proba[:, 0],
                    'Probability Malignant': predictions_proba[:, 1]
                })
                st.dataframe(pred_df, use_container_width=True)
            else:
                st.warning(f"Model {selected_model} not loaded.")
        
        except Exception as e:
            st.error(f"Error processing test data: {e}")

with tab3:
    st.subheader("Dataset Information")
    
    st.write("""
    **Breast Cancer Wisconsin (Diagnostic) Dataset**
    
    - **Source:** UCI Machine Learning Repository
    - **Total Instances:** 569
    - **Total Features:** 30 numeric features
    - **Target Variable:** diagnosis (M = Malignant, B = Benign)
    - **Class Distribution:** Benign ≈ 62.7%, Malignant ≈ 37.3%
    - **Missing Values:** Imputed using median strategy
    
    **Features Description:**
    
    The dataset contains measurements for 10 characteristics (Radius, Texture, Perimeter, Area, 
    Smoothness, Compactness, Concavity, Concave points, Symmetry, Fractal dimension).
    
    For each characteristic, three statistics are computed:
    - Mean
    - Standard Error
    - Worst (largest value)
    
    This results in 30 total features used for classification.
    """)
    
    if uploaded_file is not None:
        st.write("---")
        st.write("**Uploaded Test Data Preview:**")
        test_df = pd.read_csv(uploaded_file)
        st.dataframe(test_df)
        st.write(f"Shape: {test_df.shape}")

with tab4:
    st.subheader("About This Project")
    
    st.write("""
    **ML Assignment 2 - Classification Models Comparison**
    
    This Streamlit application demonstrates the implementation and comparison of 6 machine learning 
    classification models on the Breast Cancer Wisconsin (Diagnostic) dataset.
    
    **Models Implemented:**
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbors (KNN)
    4. Naive Bayes (Gaussian)
    5. Random Forest (Ensemble)
    6. XGBoost (Ensemble)
    
    **Evaluation Metrics:**
    - Accuracy
    - AUC Score
    - Precision
    - Recall
    - F1 Score
    - Matthews Correlation Coefficient (MCC)
    
    **Key Findings:**
    
    Based on the comparison, **XGBoost** emerged as the best-performing model with:
    - Highest Accuracy: 97.37%
    - Perfect Precision: 1.0
    - Highest F1-Score: 0.963
    - Excellent AUC: 0.994
    
    This makes it the most suitable for clinical deployment in breast cancer diagnosis support.
    
    **Repository:** [GitHub Link - Add your GitHub URL]
    
    **Author:** Assignment Submission
    **Date:** February 2026
    """)
    
    st.markdown("---")
    st.write("📚 **README.md Content** — See the main repository for complete documentation.")

# Footer
st.markdown("---")
st.markdown("""
<center>
    <small>Breast Cancer Classification - ML Models Comparison | Built with Streamlit | Assignment 2</small>
</center>
""", unsafe_allow_html=True)
