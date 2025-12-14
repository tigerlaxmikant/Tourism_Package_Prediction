# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, mean_squared_error, mean_absolute_error
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# Paths to training/test splits on Hugging Face Hub
Xtrain_path = "hf://datasets/laxmikantdeshpande/tourism-package-prediction/Xtrain.csv"
Xtest_path  = "hf://datasets/laxmikantdeshpande/tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/laxmikantdeshpande/tourism-package-prediction/ytrain.csv"
ytest_path  = "hf://datasets/laxmikantdeshpande/tourism-package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()  # ensure Series
ytest  = pd.read_csv(ytest_path).squeeze()

# Define numeric and categorical features
numeric_features = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar',
    'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'CityTier', 'Occupation', 'Gender', 'ProductPitched', 'Designation'
]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost Classifier
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)

    # Probabilities
    y_proba_train = best_model.predict_proba(Xtrain)[:, 1]
    y_proba_test  = best_model.predict_proba(Xtest)[:, 1]

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(ytrain, y_proba_train))
    test_rmse  = np.sqrt(mean_squared_error(ytest, y_proba_test))

    train_mae = mean_absolute_error(ytrain, y_proba_train)
    test_mae  = mean_absolute_error(ytest, y_proba_test)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score'],
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae
    })

    # Save the model locally
    model_path = "best_tourism_project_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "laxmikantdeshpande/tourism_project_model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
