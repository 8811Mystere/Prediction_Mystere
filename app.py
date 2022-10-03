from distutils.log import debug
from flask import Flask, request,jsonify, render_template, redirect, url_for
import sklearn 
# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import pickle
app=Flask(__name__)


@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    model=pickle.load(open('predMemoireSVMTous.pkl','rb'))
    int_features=[float(i) for i in request.form.values()]
    dernier_features=[np.array(int_features)]
    # features_model=np.delete(dernier_features)
    dernier_features=np.array([dernier_features]).reshape(1,22)
    predire=model.predict(dernier_features)
    #orientation=round(predire[0])
    #orientation
    return render_template('index.html',prediction_text_='Votre orientation est: {}'.format(predire))

if __name__ == "__main__":
    app.run(debug=True)
    