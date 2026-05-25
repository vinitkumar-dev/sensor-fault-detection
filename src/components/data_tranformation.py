import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    artifact_dir = os.path.join(ARTIFACTS_DIR)
    transformed_train_file_path = os.path.join(artifact_dir,'train.npy') # numpy arr k roop m store
    transformed_test_file_path= os.path.join(artifact_dir,'test.npy')
    transformed_object_file_path = os.path.join(artifact_dir,'preprocessor.pkl')


class DataTransformation:
    def __init__(self,feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()
        
    
    @staticmethod
    def get_data(feature_store_file_path:str)->pd.DataFrame:

        try:
            data = pd.read_csv(feature_store_file_path)
            data.rename({'Good/Bad':TARGET_COLUMN},inplace=True)

            return data
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_data_transform_object(self):

        try:
            imputer_step = ('imputer',SimpleImputer(strategy='consatant',fill_value=0))

            scaler_step = ('scaler',StandardScaler())

            preprocessor = Pipeline(steps=[imputer_step,scaler_step])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self):
        
        logging.info('Entered initiated data transformation method')
        try:
            dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)

            dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].replace({-1:0})

            X = dataframe.drop(TARGET_COLUMN,axis=1)
            y = dataframe[TARGET_COLUMN]

            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

            preprocessor = self.get_data_transform_object()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            preprocessor_path = self.data_transformation_config.transformed_object_file_path

            os.makedirs(preprocessor_path,exist_ok=True)

            self.utils.save_object(file_path=preprocessor_path,obj=preprocessor)
            
            train_arr = np.c[X_train_scaled,y_train]
            test_arr = np.c[X_test_scaled,y_test]

            return (train_arr,test_arr,preprocessor_path)
        


        except Exception as e:
            raise CustomException(e,sys)