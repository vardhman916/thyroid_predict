from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.model_trainer import modelTrainerConfig,ModelTrainer
from src.mlproject.components.data_transformation import DataTransformationConfig,DataTransformation
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.mlproject.pipeline.predict_pipelien import CustomData,PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoints():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Age = request.form.get('Age'),
            Gender = request.form.get('Gender'),
            Smoking = request.form.get('Smoking'),
            Hx_Smoking = request.form.get('Hx_Smoking'),
            Hx_Radiothreapy = request.form.get('Hx_Radiothreapy'),
            Thyroid_Function = request.form.get('Thyroid_Function'),
            Physical_Examination = request.form.get('Physical_Examination'),
            Adenopathy = request.form.get('Adenopathy'),
            Pathology = request.form.get('Pathology'),
            Focality = request.form.get('Focality'),
            Risk = request.form.get('Risk'),
            T = request.form.get('T'),
            N = request.form.get('N'),
            M = request.form.get('M'),
            Stage = request.form.get('Stage'),
            Response = request.form.get('Response')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        results = "Yes" if results == 1 else "No"
        return render_template('home.html',results = results)

if __name__=='__main__':
    logging.info("The execution has started")
    try:
        #data_ingestio_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path =  data_ingestion.initiate_data_ingestion()
        #data_transformation_config  = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        #print(train_arr.shape)
        #model traianing
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        raise CustomException(e,sys)
    
app.run(host = "0.0.0.0",port = 5000,debug = True)