import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import os
from data_preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


DAGSHUB_USERNAME = "azaams"  
REPO_NAME = "Spotify-Churn-ML"     
DATASET_PATH = 'spotify_churn_preprocessing.csv'
TARGET_COLUMN = 'churned'

def main():
    # Cek File Data
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: File {DATASET_PATH} not found!")
        return
    # Inisialisasi DagsHub dan MLflow
    try:
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow")
        mlflow.set_experiment("Spotify_Advanced_Tuning")
    except Exception as e:
        print(f"Gagal koneksi: {e}")
        return

    # Load Data
    print("Loading Data...")
    
    df = pd.read_csv(DATASET_PATH)

    # Preprocess Data
    X, y = preprocess_data(df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = [
        {'n_estimators': 50, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 150, 'max_depth': 15}
    ]

    print(f"Mulai proses training {len(param_grid)}")

    for i, params in enumerate(param_grid):
        run_name = f"Run_Tuning_{i+1}"
        
        with mlflow.start_run(run_name=run_name):
            print(f"Training {run_name} -> {params}")

            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mlflow.log_param("n_estimators", params['n_estimators'])
            mlflow.log_param("max_depth", params['max_depth'])

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            
            signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, "model", signature=signature)

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix ({run_name})')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            cm_path = f"confusion_matrix_{i}.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path) 

            plt.figure(figsize=(8, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            features = X.columns
            
            sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
            plt.title(f"Feature Importance ({run_name})")
            
            fi_path = f"feature_importance_{i}.png"
            plt.savefig(fi_path)
            plt.close()
            mlflow.log_artifact(fi_path)

            if os.path.exists(cm_path): os.remove(cm_path)
            if os.path.exists(fi_path): os.remove(fi_path)

    print(f"Link: https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}")

if __name__ == "__main__":
    main()