import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/vardhman916/thyroid_predict.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "vardhman916"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "ajaladllasdladadLAD"

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Inside your training script
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.85)
