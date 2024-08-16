import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self, model_path='artifacts/model.pkl', preprocessor_path='artifacts/preprocessor.pkl'):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = self._load_model()
        self.preprocessor = self._load_preprocessor()

    def _load_model(self):
        """Load the machine learning model."""
        try:
            return load_object(file_path=self.model_path)
        except Exception as e:
            raise CustomException(f"Error loading model: {e}", sys)

    def _load_preprocessor(self):
        """Load the preprocessor (e.g., scaler)."""
        try:
            return load_object(file_path=self.preprocessor_path)
        except Exception as e:
            raise CustomException(f"Error loading preprocessor: {e}", sys)

    def predict(self, features):
        """Scale the input features and make a prediction."""
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(f"Error during prediction: {e}", sys)


class CustomData:
    def __init__(self, BMI, Pregnancies, Glucose, DiabetesPedigreeFunction, BloodPressure, Insulin, SkinThickness, Age):
        self.data = {
            "BMI": [BMI],
            "BloodPressure": [BloodPressure],
            "Pregnancies": [Pregnancies],
            "Insulin": [Insulin],
            "Glucose": [Glucose],
            "SkinThickness": [SkinThickness],
            "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
            "Age": [Age],
        }

    def get_data_as_data_frame(self):
        """Convert input data into a DataFrame."""
        try:
            return pd.DataFrame(self.data)
        except Exception as e:
            raise CustomException(f"Error creating DataFrame: {e}", sys)
