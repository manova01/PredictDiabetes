import sys
import pandas as pd
from flask import Flask, request, render_template
from src.utils import load_object
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            BMI=float(request.form.get('BMI')),
            Pregnancies=int(request.form.get('Pregnancies')),
            Glucose=int(request.form.get('Glucose')),
            BloodPressure=int(request.form.get('BloodPressure')),
            SkinThickness=int(request.form.get('SkinThickness')),
            Insulin=int(request.form.get('Insulin')),
            DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction')),
            Age=int(request.form.get('Age'))
        )
        
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
