from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('ontheside.html')

@app.route('/predict', methods=['GET', 'POST'])
def serve_predictions():
    carat_size = request.form['carat']
    length = request.form['length']
    diamond_clarity = request.form['clarity']
    diamond_color = request.form['color']
    array = np.array([carat_size, length, diamond_clarity, diamond_color])
    array = array.astype('float')
    array = array.reshape(1, -1)
    prediction = model.predict(array)
    
    return render_template('ontheside.html', data=int(prediction))

if __name__ == '__main__':
    app.run(debug=True)