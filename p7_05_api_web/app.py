#to run :
#en mode local :dans le dossier du fichier app.py faire python app.py puis dans le navigateur aller à http://127.0.0.1:5000/

# Import all packages and libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request
import pickle
import math
import base64
from zipfile import ZipFile
from lightgbm import LGBMClassifier


app= Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])

def predict():
    '''
    For rendering results on HTML GUI
    '''

    z = ZipFile("df_test_imputed.zip")
    dataframe = pd.read_csv(z.open('df_test_imputed.csv'), encoding ='utf-8')
    all_id_client = list(dataframe['SK_ID_CURR'].unique())

    model = pickle.load(open('best_final_prediction.pickle', 'rb'))
    seuil = 0.6

    ID = request.form['id_client']
    ID = int(ID)
    if ID not in all_id_client:
        prediction="Ce client n'est pas reppertorie"
    else :
        X = dataframe[dataframe['SK_ID_CURR'] == ID]
        X = X.drop(['SK_ID_CURR'], axis=1)

        #data = df[df.index == comment]
        probability_default_payment = model.predict_proba(X)[:, 1]
        if probability_default_payment >= seuil:
            prediction = "Prêt NON Accordé"
        else:
            prediction = "Prêt Accordé"

    return render_template('index.html', prediction_text=prediction)

# Define endpoint for flask
app.add_url_rule('/predict', 'predict', predict)

# Run app.
# Note : comment this line if you want to deploy on heroku
#app.run()
#app.run(debug=True)
