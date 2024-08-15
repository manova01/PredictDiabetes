import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a ColumnTransformer object that preprocesses the numerical columns
        by imputing missing values with the median and scaling the features.
        """
        try:
            numerical_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 
                                 'SkinThickness', 'Insulin', 'BMI', 
                                 'DiabetesPedigreeFunction', 'Age']
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
                ]
            )
            
            logging.info('Standard scaler pipeline created.')
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies the data transformation to the training and testing datasets, and saves the preprocessing object.
        """
        try:
            # Read training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Read train data: {train_df.shape}")
            logging.info(f"Read test data: {test_df.shape}")

            logging.info("Obtaining preprocessing object")

            # Obtain the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Outcome"
            numerical_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 
                                 'SkinThickness', 'Insulin', 'BMI', 
                                 'DiabetesPedigreeFunction', 'Age']

            # Separate input features and target feature
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Apply the preprocessing transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features with target feature
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")

            logging.info("Saving preprocessing object.")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)