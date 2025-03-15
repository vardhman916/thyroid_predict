import sys
import pandas as pd
from src.mlproject.exception import CustomException
from src.mlproject.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,feature):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scale = preprocessor.transform(feature)
            preds = model.predict(data_scale)
            return preds
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,Age,Gender,Smoking,Hx_Smoking,Hx_Radiothreapy,Thyroid_Function,Physical_Examination,Adenopathy,Pathology,Focality,Risk,T,N,M,Stage,Response):
        self.Age = Age
        self.Gender = Gender
        self.Smoking = Smoking
        self.Hx_Smoking = Hx_Smoking
        self.Hx_Radiothreapy = Hx_Radiothreapy
        self.Thyroid_Function = Thyroid_Function
        self.Physical_Examination = Physical_Examination
        self.Adenopathy = Adenopathy
        self.Pathology = Pathology
        self.Focality = Focality
        self.Risk = Risk
        self.T = T
        self.N = N
        self.M = M
        self.Stage = Stage
        self.Response = Response

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age":[self.Age],
                "Gender":[self.Gender],
                "Smoking":[self.Smoking],
                "Hx_Smoking":[self.Hx_Smoking],
                "Hx_Radiothreapy":[self.Hx_Radiothreapy],
                "Thyroid_Function":[self.Thyroid_Function],
                "Physical_Examination":[self.Physical_Examination],
                "Adenopathy":[self.Adenopathy],
                "Pathology":[self.Pathology],
                "Focality":[self.Focality], 
                "Risk":[self.Risk],
                "T":[self.T],
                "N":[self.N],
                "M":[self.M],
                "Stage":[self.Stage],
                "Response":[self.Response]

            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
