import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_preprocessing import preprocess_data

DATASET_PATH = 'spotify_churn_preprocessing.csv'
TARGET_COLUMN = 'churned'

EXPERIMENT_NAME = 'Spotify Churn Prediction'
RUN_NAME = 'RandomForest_Basic'

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.autolog()

    with mlflow.start_run(run_name=RUN_NAME) as run:
        print("Loading data...")

        if not os.path.exists(DATASET_PATH):
            print(f"ERROR: File {DATASET_PATH} tidak ditemukan!")
            return

        # Load Data
        df = pd.read_csv(DATASET_PATH)

        # Preprocess Data
        X, y = preprocess_data(df, TARGET_COLUMN)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define hyperparameters
        n_estimators = 100
        max_depth = 10
        
        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset_source", "Spotify Churn Dataset")

        # Train Model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate Model
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Save model
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model, "model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        image_path = 'confusion_matrix.png'
        plt.savefig(image_path)
        plt.close()

        # Log confusion matrix image
        mlflow.log_artifact(image_path)

        # Clean up
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    main()
