from src.exception import  CustomException
from src.constant import *
from src.logger import logging
from src.utils.main_utils import MainUtils

import os,sys
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split,GridSearchCV


from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    artifact_folder:str = ARTIFACTS_DIR
    trained_model_path = os.path.join(artifact_folder,'model.pkl')
    expected_accuracy = 0.55
    model_config_file_path = os.path.join('config','model.yaml')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
            "LogisticRegression":LogisticRegression(),
            "SVC": SVC(),
            "DecisionTreeClassifier":DecisionTreeClassifier(),
            "XGBClassifier":XGBClassifier(),
            "RandomForestClassifier":RandomForestClassifier(),
            "GaussianNB":GaussianNB()
        }

    def evaluate_models(self,X_train,y_train,X_test,y_test,models):

        try:
            report = {}

            for name, model in models.items():
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                test_score = accuracy_score(y_test, y_pred)

                report[name] = test_score

            return report
        
        except Exception as e:
            raise CustomException(e,sys)
    

    def get_best_models(self,X_train,y_train,X_test,y_test):
        try:
            model_report:dict = self.evaluate_models(
                X_train,y_train,X_test,y_test,self.models
            )

            print(model_report)

            best_model_name = max(model_report,key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model_object = self.models[best_model_name]

            return best_model_object,best_model_name,best_model_score



        except Exception as e:
            raise CustomException(e,sys)
        
    
    def finetune_best_model(self,best_model_object,best_model_name,X_train,y_train):
        try:
            params = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)['models'][best_model_name]

            grid_search = GridSearchCV(estimator=best_model_object,param_grid=params,cv=5,verbose=1)

            grid_search.fit(X_train,y_train)

            best_params = grid_search.best_params_

            print("best params are:", best_params)


            finetuned_model = best_model_object.set_params(**best_params)

            return finetuned_model


        except Exception as e:
            raise CustomException(e,sys)


    def initiate_model_trainer(self,train_array,test_array):

        try:
            logging.info(f"Splitting training and testing input and target feature")


            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info(f"Extracting model config file path")

            logging.info(f"Extracting model config file path")

            best_model_object,best_model_name,best_model_score =  self.get_best_models(X_train,y_train,X_test,y_test)

            best_model = self.finetune_best_model(best_model_object=best_model_object,best_model_name=best_model_name,X_train=X_train,y_train=y_train)


            y_pred = best_model.predict(X_test)
            best_model_score = accuracy_score(y_test, y_pred)

            print(f"best model name {best_model_name} and score: {best_model_score}")

            if best_model_score < 0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.6")
           
            logging.info(f"Best found model on both training and testing dataset")

            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_path}")
            
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path),exist_ok=True)

            self.utils.save_object(file_path=self.model_trainer_config.trained_model_path,obj=best_model)


            return best_model_score
        
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise e
        



        

            