import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataHandler:
    def __init__(self, data_url="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"):
        self.data_url = data_url
        self.data_path = os.path.join("data", "heart.csv")
        self.columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol',
            'fbs', 'restecg', 'thalach', 'exang', 
            'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]

    def download_data(self):
        """Downloads the UCI Heart Disease dataset if it doesn't exist."""
        if not os.path.exists(self.data_path):
            print("Downloading data...")
            urllib.request.urlretrieve(self.data_url, self.data_path)
            print("Download complete.")
        else:
            print("Data already exists.")

    def load_and_clean_data(self):
        """Loads data and handles missing values marked as '?'."""
        self.download_data()
        df = pd.read_csv(self.data_path, names=self.columns)
        
        # Replace '?' with NaN and drop those rows
        df = df.replace('?', pd.NA).dropna()
        
        # Convert columns to numeric types
        for col in ['ca', 'thal']:
            df[col] = pd.to_numeric(df[col])
            
        # Target in Cleveland dataset is 0 (no presence) to 4. We want binary classification (0 or 1)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        return df

    def preprocess_data(self, df):
        """Splits data and scales features."""
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve feature names (useful for SHAP)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

if __name__ == "__main__":
    dh = DataHandler()
    df = dh.load_and_clean_data()
    X_train, X_test, y_train, y_test, scaler, cols = dh.preprocess_data(df)
    print(f"Data shape: {df.shape}")
    print("Preprocessing complete.")
