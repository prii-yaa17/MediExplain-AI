import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Explainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        # Note: Depending on the model, we use TreeExplainer or LinearExplainer
        # For simplicity and robustness across LogReg, RF, XGB, we can use an independent explainer 
        # or determine it by type. TreeExplainer is best for RF/XGB.
        
        try:
            self.explainer = shap.TreeExplainer(model)
            self.is_tree = True
        except Exception:
            # Fallback for Logistic Regression
            self.is_tree = False
            self.explainer = shap.LinearExplainer(model, shap.maskers.Independent(np.zeros((1, len(feature_names)))))

    def get_local_explanation(self, instance_df):
        """Returns the SHAP values for a single patient instance."""
        if self.is_tree:
            shap_values = self.explainer.shap_values(instance_df)
            # XGBoost returns (n_samples, n_features) for binary classification
            # Random Forest returns a list of arrays [class_0, class_1] or a 3D array
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Path for positive class
        else:
            shap_values = self.explainer.shap_values(instance_df)
            
        # Extract correct 1D array for features
        sv_array = np.array(shap_values)
        if len(sv_array.shape) == 3:
            # Shape is (n_samples, n_features, n_classes). We want class 1.
            vals = sv_array[0, :, 1]
        elif len(sv_array.shape) == 2:
            vals = sv_array[0, :]
        else:
            vals = sv_array.flatten()

        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': vals
        })
        feature_importance['Abs_Importance'] = feature_importance['Importance'].abs()
        feature_importance = feature_importance.sort_values(by='Abs_Importance', ascending=False)
        return feature_importance

    def generate_natural_language_explanation(self, importance_df, target_prediction):
        """Generates a text summary based on the highest impact features."""
        # Top 3 features pushing risk up
        positive_contributors = importance_df[importance_df['Importance'] > 0].head(2)
        # Top 2 features pushing risk down
        negative_contributors = importance_df[importance_df['Importance'] < 0].head(2)

        risk_str = "High" if target_prediction == 1 else "Low"

        # Feature mapping for better readability
        feature_map = {
            'age': 'Age', 'sex': 'Gender', 'cp': 'Chest Pain Type', 'trestbps': 'Resting Blood Pressure',
            'chol': 'Cholesterol', 'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting ECG',
            'thalach': 'Maximum Heart Rate', 'exang': 'Exercise Induced Angina',
            'oldpeak': 'ST Depression', 'slope': 'Slope of Peak Exercise ST Segment',
            'ca': 'Number of Major Vessels', 'thal': 'Thalassemia'
        }

        explanation = f"The model predicted a {risk_str} risk of heart disease.\n\n"
        
        if not positive_contributors.empty:
            pos_features = [feature_map.get(f, f) for f in positive_contributors['Feature']]
            explanation += f"Factors significantly increasing the risk include: {', '.join(pos_features)}.\n"
            
        if not negative_contributors.empty:
            neg_features = [feature_map.get(f, f) for f in negative_contributors['Feature']]
            explanation += f"Factors helping to lower the risk include: {', '.join(neg_features)}."

        return explanation

    def plot_local_bar(self, importance_df):
        """Creates a customized matplotlib block for local feature importances."""
        # Setup colors
        colors = ['red' if val > 0 else 'blue' for val in importance_df.head(10)['Importance']]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df.head(10)['Feature'][::-1], importance_df.head(10)['Importance'][::-1], color=colors[::-1])
        ax.set_xlabel("SHAP Value (Impact on Output)")
        ax.set_title("Top 10 Influential Factors for this Patient")
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        plt.tight_layout()
        return fig
