import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class DataTransformationConfig:
    artifact_dir: str = ARTIFACTS_DIR

    transformed_train_file_path: str = os.path.join(
        artifact_dir,
        "train.npy"
    )

    transformed_test_file_path: str = os.path.join(
        artifact_dir,
        "test.npy"
    )

    transformed_object_file_path: str = os.path.join(
        artifact_dir,
        "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self, feature_store_file_path: str):

        try:
            self.feature_store_file_path = feature_store_file_path

            self.data_transformation_config = DataTransformationConfig()

            self.utils = MainUtils()

            logging.info("DataTransformation class initialized")

        except Exception as e:
            raise CustomException(e, sys)

    
    def get_data(self,feature_store_file_path: str) -> pd.DataFrame:

        try:
            logging.info("Reading dataset from feature store")

            dataframe = pd.read_csv(feature_store_file_path)

            logging.info("Dataset loaded successfully")

            dataframe.rename(
                columns={"Good/Bad": TARGET_COLUMN},
                inplace=True
            )

            logging.info("Target column renamed successfully")

            return dataframe

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transform_object(self) -> Pipeline:

        try:
            logging.info("Creating preprocessing pipeline")

            imputer_step = (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0
                )
            )

            scaler_step = (
                "scaler",
                StandardScaler()
            )

            preprocessor = Pipeline(
                steps=[
                    imputer_step,
                    scaler_step
                ]
            )

            logging.info("Preprocessing pipeline created successfully")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):

        try:
            logging.info(
                "Entered initiate_data_transformation method"
            )

            dataframe = self.get_data(
                feature_store_file_path=self.feature_store_file_path
            )

            logging.info("Replacing target values")

            dataframe[TARGET_COLUMN] = dataframe[
                TARGET_COLUMN
            ].replace({-1: 0})

            logging.info("Splitting input and target features")

            X = dataframe.drop([TARGET_COLUMN], axis=1)

            y = dataframe[TARGET_COLUMN]

            logging.info("Performing train test split")

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42
            )

            logging.info("Train test split completed")

            preprocessor = self.get_data_transform_object()

            logging.info("Applying preprocessing on training data")

            X_train_scaled = preprocessor.fit_transform(X_train)

            logging.info("Applying preprocessing on testing data")

            X_test_scaled = preprocessor.transform(X_test)

            train_arr = np.c_[
                X_train_scaled,
                y_train.to_numpy()
            ]

            test_arr = np.c_[
                X_test_scaled,
                y_test.to_numpy()
            ]

            logging.info("Creating artifacts directory")

            os.makedirs(
                self.data_transformation_config.artifact_dir,
                exist_ok=True
            )

            logging.info("Saving preprocessor object")

            self.utils.save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor
            )

            logging.info("Saving transformed train array")

            np.save(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            logging.info("Saving transformed test array")

            np.save(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            logging.info(
                "Data transformation completed successfully"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.transformed_object_file_path
            )

        except Exception as e:
            logging.error("Error occurred in data transformation")
            raise CustomException(e, sys)