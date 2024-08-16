import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
import warnings
# Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import warnings

from src.utils import save_object

@dataclass
class ModelTrainerconfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    accuracy_threshold: float = 0.6  # Configurable threshold for model selection

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()
        
    def evaluate_model(self, X_train, y_train, X_test, y_test, models):
        model_report = {}
        
        for model_name, model_details in models.items():
            logging.info(f"Evaluating model: {model_name}")
            model = model_details["model"]
            params = model_details["params"]

            grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            model_report[model_name] = accuracy
            logging.info(f"Model: {model_name}, Accuracy: {accuracy}")
        
        return model_report
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Define the models and their hyperparameters
            models = {
                "Logistic Regression": {
                    "model": LogisticRegression(),
                    "params": {
                        "C": [0.1, 1, 10, 100],
                        "solver": ["liblinear", "lbfgs"]
                    }
                },
                "Decision Tree": {
                    "model": DecisionTreeClassifier(),
                    "params": {
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10]
                    }
                },
                "Random Forest": {
                    "model": RandomForestClassifier(),
                    "params": {
                        "n_estimators": [10, 50, 100],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5]
                    }
                },
                "Support Vector Machine": {
                    "model": SVC(kernel='rbf'),
                    "params": {
                        "C": [0.1, 1, 10],
                        "gamma": [0.001, 0.01, 0.1, 1]
                    }
                },
                "K Neighbour": {
                    "model": KNeighborsClassifier(),
                    "params": {
                        "n_neighbors": [3, 5, 7, 9],
                        "weights": ["uniform", "distance"]
                    }
                },
                "Gradient Boosting": {
                    "model": GradientBoostingClassifier(),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7]
                    }
                },
                "Naive Bayes": {
                    "model": GaussianNB(),
                    "params": {}
                },
                "Neural Networks": {
                    "model": MLPClassifier(hidden_layer_sizes=(100,)),
                    "params": {
                        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                        "activation": ["relu", "tanh"],
                        "learning_rate": ["constant", "adaptive"]
                    }
                }
            }
            
            model_report = self.evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            # To get the best model score from the dict
            best_model_score = max(sorted(model_report.values()))
            
            # To get the model name from the dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]["model"]
            
            if best_model_score < self.model_trainer_config.accuracy_threshold:
                raise CustomException('No best model found')
            logging.info(f"Best found model: {best_model_name} with accuracy: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            return best_model_score, best_model
        
        except Exception as e:
            raise CustomException(e, sys)
