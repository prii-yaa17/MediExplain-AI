MediExplain AI

Explainable AI-Based Medical Diagnosis System Using Multi-Model Machine Learning

Overview

MediExplain AI is a machine learning-based medical diagnosis support system designed to predict heart disease risk using patient health parameters. The system integrates multiple machine learning algorithms to generate predictions, evaluate model performance, and provide interpretable results using Explainable Artificial Intelligence (XAI).

It uses SHAP (SHapley Additive exPlanations) to identify the key features influencing model predictions, helping users understand how and why a particular prediction is generated.

The application provides an interactive dashboard built with Streamlit for entering patient data, viewing prediction results, comparing models, and exploring feature contributions.

Key Features

- Multi-Model Prediction: Implements multiple machine learning algorithms for heart disease risk prediction.
- Model Comparison: Evaluates different models using standard performance metrics.
- Explainable AI: Uses SHAP to explain individual predictions and identify important features.
- Risk Prediction: Generates heart disease risk predictions based on patient health parameters.
- Interactive Dashboard: Provides a user-friendly Streamlit interface.
- Data Visualization: Displays model performance and feature importance through visualizations.
- What-If Analysis: Explores how changes in patient parameters influence model predictions.

Machine Learning Algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Explainable AI

The project integrates SHAP (SHapley Additive exPlanations) to improve model interpretability.

SHAP helps:

- Identify important features influencing predictions.
- Understand individual model predictions.
- Visualize positive and negative feature contributions.
- Improve transparency in machine learning-based decision support.

Technology Stack

Category| Technologies
Programming Language| Python
Machine Learning| Scikit-learn
Explainable AI| SHAP
Data Processing| Pandas, NumPy
Visualization| Matplotlib
Dashboard| Streamlit
Dataset| UCI Cleveland Heart Disease Dataset

Project Workflow

1. Data Collection: Load the heart disease dataset.
2. Data Preprocessing: Clean and prepare the dataset for model training.
3. Model Training: Train multiple machine learning classification algorithms.
4. Model Evaluation: Compare models using accuracy, precision, recall, F1-score, and confusion matrix.
5. Explainability: Apply SHAP to understand feature contributions.
6. Prediction: Generate predictions using patient health parameters.
7. Visualization: Display predictions and explainability results through the Streamlit dashboard.

Project Objectives

- Develop a machine learning-based medical diagnosis support system.
- Compare multiple classification algorithms for predictive performance.
- Improve prediction transparency using Explainable AI.
- Provide understandable insights into factors influencing predictions.
- Build an interactive dashboard for exploring prediction results.

Disclaimer

MediExplain AI is developed for academic and educational purposes. It is intended as a decision-support prototype and should not be used as a replacement for professional medical advice, diagnosis, or treatment.

Author

Priyadarshini K C
