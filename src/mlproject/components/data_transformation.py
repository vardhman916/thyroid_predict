import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from  sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.mlproject.utils import save_object
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''this class is responsible for data transformation'''
        try:
            numerical_columns = ['Age']
            categorical_columns = ['Gender','Smoking','Hx_Smoking',
                                   'Hx_Radiothreapy',
                                   'Thyroid_Function','Physical_Examination',
                                   'Adenopathy','Pathology','Focality',
                                   'Risk','T','N','M','Stage',
                                   'Response'
                                   ]

            num_pipeline = Pipeline(steps = [
                ("imputer",SimpleImputer(strategy = 'median')),
                ("scaler",StandardScaler())
            ])

            cat_pipeline = Pipeline(steps = [
                 ("imputer",SimpleImputer(strategy='most_frequent')),
                 ("one_hot_encoder",OneHotEncoder()),
                 ("scaler",StandardScaler(with_mean = False))
            ])

            logging.info(f"Categorical Column:{categorical_columns}")
            logging.info(f"Numerical Column:{numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]

            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("reading the test and train file")
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name = "Recurred"
            input_feature_train_df = train_df.drop(columns = [target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]
           

            input_feature_test_df = test_df.drop(columns = [target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]

            

            logging.info("Applying Preprocessing on training and testing")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            input_feature_train_arr = input_feature_train_arr.toarray()
            input_feature_test_arr = input_feature_test_arr.toarray()
            target_feature_train_df = target_feature_train_df.to_numpy()
            target_feature_test_df = target_feature_test_df.to_numpy()


            # print("Shape of input_feature_train_arr:", type(input_feature_train_arr))
            # print("Shape of target_feature_train_df:", type(target_feature_train_df))
            # print("Shape of input_feature_test_arr:", type(input_feature_test_arr))
            # print("Shape of target_feature_test_df:", type(target_feature_test_df))

            # print("Shape of input_feature_train_arr:", input_feature_train_arr.ndim)
            # print("Shape of target_feature_train_df:", target_feature_train_df.ndim)
            # print("Shape of input_feature_test_arr:", input_feature_test_arr.ndim)
            # print("Shape of target_feature_test_df:", target_feature_test_df.ndim)

            target_feature_train_df = target_feature_train_df.reshape(-1, 1)
            target_feature_test_df = target_feature_test_df.reshape(-1, 1)
 
            # print(f"input_feature_train_arr shape: {input_feature_train_arr.shape}")
            # print(f"target_feature_train_df shape after reshaping: {np.array(target_feature_train_df).reshape(-1, 1).shape}")

            train_arr = np.c_[
                input_feature_train_arr,target_feature_train_df
            ]
            test_arr = np.c_[
                input_feature_test_arr,target_feature_test_df
            ]
            
            logging.info(f"Saved preprocessing")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
           
            
        except Exception as e:
            raise CustomException(e,sys)

        
