import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from utils.data_handler import DataHandler

class ModelTrainer:
    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        results = {}
        best_f1 = 0
        best_model_name = ""
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            
            # Save results
            results[name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC AUC": roc_auc,
                "fpr": fpr,
                "tpr": tpr
            }
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{name.replace(' ', '_').lower()}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name

        results["best_model"] = best_model_name
        
        # Save results dictionary
        with open(os.path.join(self.models_dir, "model_metrics.pkl"), 'wb') as f:
            pickle.dump(results, f)
            
        return results

if __name__ == "__main__":
    dh = DataHandler()
    df = dh.load_and_clean_data()
    X_train, X_test, y_train, y_test, scaler, cols = dh.preprocess_data(df)
    
    # Save the scaler for inference time
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
        
    # Save column names for inference
    with open(os.path.join("models", "features.pkl"), 'wb') as f:
        pickle.dump(cols, f)
        
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    print("Training complete. Best Model:", results["best_model"])
