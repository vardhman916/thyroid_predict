## this is my thyroid project

import dagshub
dagshub.init(repo_owner='vardhman916', repo_name='thyroid_predict', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)