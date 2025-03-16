import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
passw = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb = pymysql.connect(
            host = host,
            user = user,
            password = passw,
            db = db
        )
        logging.info("Connection establised",mydb)
        df = pd.read_sql_query('Select * from thyroid_diff',mydb)
        print(df.head())

        return df

    except Exception as e:
        raise CustomException(e,sys)
    

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    



def evaluate_models(X_train,y_train,X_test,y_test,models,params):
        try:
            report = {}
            encoder = LabelEncoder()
            y_train_numeric = encoder.fit_transform(y_train)
            y_test_numeric = encoder.transform(y_test)

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]

                gs = GridSearchCV(model,para,cv = 5)
                gs.fit(X_train,y_train_numeric)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train_numeric)

                #model.fit(X_train,y_train) #train model
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                

                train_model_score = r2_score(y_train_numeric,y_train_pred)
                test_model_score = r2_score(y_test_numeric,y_test_pred)

                report[list(models.keys())[i]] = test_model_score
                
            return report 


        except Exception as e:
            raise CustomException(e,sys)
