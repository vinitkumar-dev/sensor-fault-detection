import sys
import os
from typing import Generator,List,Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from dataclasses import dataclass

@dataclass
class ModelTrainConfig:
    artifact_folder = os.path.join(ARTIFACTS_DIR)
    trained_model_path = os.path.join(artifact_folder,'train.pickel')
    model_config_file_path = os.path.join('config','model.yaml')



class ModelTrainer:
    def __init__(self):
        
        self.model_trainer_config = ModelTrainConfig()
        self.utils = MainUtils()
        self.models = {
            "LogisticRegression":LogisticRegression(),
            "SVC" :SVC(),
            "GradientBoostingClassifier":GradientBoostingClassifier(),
            "RandomForestClassifier" :RandomForestClassifier(),
            "XGBClassifier":XGBClassifier()
        }

    def evaluate_models(self,X_train,X_test,y_train,y_test,models):

        try:
    
            report = {}

            for i in range(len(models)):
                model = list(models.values())[i]
                model.fit(X_train,y_train)
                y_test_pred = model.predict(X_test)
                test_model_score = accuracy_score(y_test,y_test_pred)
                report[list(models.keys())[i]] = test_model_score

                
            return report

        except Exception as e:
             raise CustomException(e,sys)
        
    def get_best_model(self,X_train:np.array,X_test:np.array,y_train:np.array,y_test:np.array):

            try:
                 model_report = self.evaluate_models(X_train,X_test,y_train,y_test,models=self.models)

                 best_model_score = max(model_report.values())

                 best_model_name = max(model_report, key=model_report.get)

                 best_model_object = self.models[best_model_name]

                 return best_model_object,best_model_name,best_model_score
            
            except Exception as e:
                raise CustomException(e,sys)
        
    def finetune_best_model(self,best_model_object:object,best_model_name,X_train,y_train)->object:
         
        try:
              model_param_grid = self.utils.read_yaml_file(self,self.model_trainer_config.model_config_file_path)['models'][best_model_name]

              grid_search = GridSearchCV(estimator=best_model_object,param_grid=model_param_grid,cv=5,n_jobs=-1,verbose=1)

              grid_search.fit(X_train,y_train)
              best_params= grid_search.best_params_

              print('best params are', best_params)
              finetuned_model = best_model_object.set_params(**best_params)

              return finetuned_model
              

        except Exception as e:
                raise CustomException(e,sys)
        
    
    def initiate_model_trainer(self,train_array,test_array):
         try:
              logging.info(f'Splitting training and testing input and target features')

              X_train,y_train,X_test,y_test = (
                   train_array[:,:-1],
                   train_array[:,-1],
                   test_array[:,:-1],
                   test_array[:,-1],
              )
              logging.info('Extracting model config file path')

              obj,name,score = self.get_best_model(X_train,X_test,y_train,y_test)

              finetuned_model = self.finetune_best_model(best_model_name=name,best_model_object=obj,X_train=X_train,y_train=y_train)

              y_pred = finetuned_model.predict(X_test)

              best_model_score = accuracy_score(y_test,y_pred)

              if best_model_score < 0.6:
                   raise Exception('No best model is found with an accuracy greater than the threshold 0.6')
            

              logging.info(f'Best found model on both training and testing dataset')

              os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path),exist_ok=True)

              self.utils.save_object(
                   file_path = self.model_trainer_config.trained_model_path,
                   obj = finetuned_model
              )

              return self.model_trainer_config.trained_model_path

              


         except Exception as e:
              raise CustomException(e,sys)