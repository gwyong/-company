import pickle, joblib
import pandas as pd
import numpy as np
import timeit

from flask import Flask, request
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression as  PLS
from sklearn.kernel_ridge import KernelRidge as KR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

@app.route('/riskscore', methods=['GET', 'POST'])
def post():
    params = request.get_json()
    fileRoot = 'fileRoot'
    ## load encoders
    folderRoot = 'folderRoot'
    loaded_OHE = joblib.load(fileRoot + folderRoot + 'OHE.pkl')

    ## loaded models
    folderRoot = 'folderRooot'
    loaded_model    = pickle.load(open(fileRoot + folderRoot + 'model.pkl', 'rb'))
    loaded_nlpModel = pickle.load(open(fileRoot + folderRoot + 'nlp_en_model.pkl',"rb"))

    ## input the user's query.
    data    = params['data']
    nlpData = params['nlpData']

    ## NLP Embedding
    embeddings = loaded_nlpModel.encode([nlpData]) # size (1, 768)
    
    ## OneHotEncoding
    convertedData = loaded_OHE.transform(np.array([data]).reshape(-1,1))

    input = np.concatenate((convertedData, embeddings), axis=1)
    estimatedRiskScore = loaded_model.predict(input)[0]

    return({"score": estimatedRiskScore})

if __name__ == '__main__':
    app.run()