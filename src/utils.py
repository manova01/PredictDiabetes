import os
import sys
import dill
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param, scoring='accuracy'):
    try:
        report = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})

            gs = GridSearchCV(estimator=model, param_grid=params, cv=3, scoring=scoring, verbose=1, n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Optionally save the best model
            save_object(f'artifacts/{model_name}_best_model.pkl', best_model)

            report[model_name] = test_model_score
            print(f"{model_name} Test Accuracy: {test_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)