import os
import sys
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsClassifier
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object,evaluate_models
from sklearn.preprocessing import LabelEncoder
import dagshub
from sklearn.metrics import mean_absolute_error, r2_score
#import warnings
#warnings.filterwarnings("ignore")

@dataclass
class modelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=modelTrainerConfig()

    def eval_metrics(self, actual, predicted):
       # rmse = mean_squared_error(actual, predicted, squared=False)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        return  mae, r2

    def initiate_model_trainer(self,train_array,test_array):
        try:

            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            encoder = LabelEncoder()
            y_train_numeric = encoder.fit_transform(y_train)
            y_test_numeric = encoder.fit_transform(y_test)

            models = {
                'Logistic_Regression' : LogisticRegression(max_iter = 2000),
                'Decision_Classifier' : DecisionTreeClassifier(random_state=42),
                'AdaBoost_Classifier' : AdaBoostClassifier(random_state=42),
                'Random_Classifier':RandomForestClassifier(random_state=42),
                'KNeighborsClassifier':KNeighborsClassifier()
                }
            
            params = {
                'Logistic_Regression':{
                    'penalty':['l1', 'l2'], 
                    'solver': ['saga']

                },
                'Decision_Classifier':{
                    'criterion': ['gini', 'entropy', 'log_loss']    
                },
                'AdaBoost_Classifier':{
                    'n_estimators':[8,16,32,64,128,256],
                    'learning_rate':[.1,.01,.05,.001]
                },
                'Random_Classifier':{
                    'criterion':['gini', 'entropy'],
                    'max_features':['sqrt', 'log2', None]
                },
                'KNeighborsClassifier':{
                    'weights':['uniform', 'distance']
                }
            
            }

            model_report:dict = evaluate_models(X_train,y_train_numeric,X_test,y_test_numeric,models,params)

            best_score_model = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_score_model)
            ]
            print("This is the best model:")
            print(best_model_name)
            best_model = models[best_model_name]

            model_names = list(params.keys())
            actual_model = "" 
            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

   
            best_params = params[actual_model]
            # mlflow.set_registry_uri("https://dagshub.com/vardhman916/thyroid_predict.mlflow")
            # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme



            # Initialize DAGsHub for MLflow tracking
            dagshub.init(repo_owner='vardhman916', repo_name='thyroid_predict', mlflow=True)

            # Start MLflow run
            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)

                (mae, r2) = self.eval_metrics(y_test_numeric, predicted_qualities)

                # Log parameters and metrics
                mlflow.log_params(best_params)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)



            if best_score_model < 0.6:
                raise CustomException("No best model found",sys)
            
            
            logging.info(f"best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            encoder = LabelEncoder()
            y_test_numeric = encoder.fit_transform(y_test)

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test_numeric,predicted)

            return r2


            
        except Exception as e:
            raise CustomException(e,sys)
        


        
    
