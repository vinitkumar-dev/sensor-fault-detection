from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline
from src.logger import logging
from src.exception import CustomException

import os,sys

from flask import Flask, render_template,jsonify,request,send_file

app = Flask(__name__)

@app.route('/')
def home():
   accuracy = "98.7%"
   return render_template(
        'index.html',
        accuracy=accuracy
    )

@app.route('/train')
def train_route():
    try:
        training_pipeline = TrainingPipeline()

        accuracy = training_pipeline.run_pipeline()

        return render_template(
            'train.html',
            accuracy=accuracy
        )

    except Exception as e:
        raise CustomException(e,sys)

@app.route('/predict',methods = ['POST','GET'])
def upload():
    try:
        
        if request.method == 'POST':
            prediction_pipeline = PredictionPipeline(request)

            prediction_file_detail = prediction_pipeline.run_pipeline()

            logging.info('prediction completed.Downloading prediction files')

            return send_file(
                prediction_file_detail.prediction_file_path,download_name=prediction_file_detail.prediction_file_name,
                as_attachment=True
            )
        
        else:
            return render_template('upload_file.html')

    except Exception as e:
        raise CustomException(e,sys)
    

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
