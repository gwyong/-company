## import libraries and packages
try:
  from sentence_transformers import SentenceTransformer
except:
  from pip._internal import main as pip
  pip(['install', '--user', 'sentence_transformers'])
  import sentence_transformers

try:
  from tqdm import tqdm
except:
  from pip._internal import main as pip
  pip(['install', '--user', 'tqdm'])
  from tqdm import tqdm

try:
  import sklearn
except:
  from pip._internal import main as pip
  pip(['install', '--user', 'sklearn'])
  import sklearn

import pickle, json, argparse, joblib, time
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.neural_network import *
from sklearn.cross_decomposition import *
from sklearn.kernel_ridge import *
from sklearn.metrics import mean_squared_error

## load scalers and models
loaded_LE_regions        = joblib.load("./scalers/LE_regions.pkl")
loaded_OHE_regions       = joblib.load("./scalers/OHE_regions.pkl")
loaded_LE_riskCategory   = joblib.load("./scalers/LE_riskCategory.pkl")
loaded_OHE_riskCategory  = joblib.load("./scalers/OHE_riskCategory.pkl")
loaded_LE_stakeholder    = joblib.load("./scalers/LE_stakeholder.pkl")
loaded_OHE_stakeholder   = joblib.load("./scalers/OHE_stakeholder.pkl")
loaded_LE_riskType       = joblib.load("./scalers/LE_riskType.pkl")
loaded_OHE_riskType      = joblib.load("./scalers/OHE_riskType.pkl")

loaded_model    = pickle.load(open("./models/model.pkl", 'rb')) # pre-trained riskScorePrediction model
loaded_nlpModel = pickle.load(open("./models/nlpModel.pkl", 'rb')) # pre-trained NLP model

## input the user's query.
parser = argparse.ArgumentParser(description='process string type of risk data.')

parser.add_argument("--region", required=True, type=str, help="Note the Region information")
parser.add_argument("--riskCategory", required=True, type=str, help="Note the Risk Category information")
parser.add_argument("--riskDescription", required=True, type=str, help="Note the Risk Description information")
parser.add_argument("--riskDriver", required=True, type=str, help="Note the Risk Driver information")
parser.add_argument("--riskImpact", required=True, type=str, help="Note the Risk Impact information")
parser.add_argument("--stakeholder", required=True, type=str, help="Note the Stakeholder information")
parser.add_argument("--riskType", required=True, type=str, help="Note the Risk Type information")

args = parser.parse_args()
data_region          = args.region
data_riskCategory    = args.riskCategory
data_riskDescription = args.riskDescription
data_riskDriver      = args.riskDriver
data_riskImpact      = args.riskImpact
data_stakeholder     = args.stakeholder
data_riskType        = args.riskType

## NLP Process
embeddings_riskDescription = loaded_nlpModel.encode([data_riskDescription]) # size (1, 768)
embeddings_riskDriver      = loaded_nlpModel.encode([data_riskDriver])
embeddings_riskImpact      = loaded_nlpModel.encode([data_riskImpact])

## OneHotEncoding
converted_data_region       = loaded_OHE_regions.transform(loaded_LE_regions.transform([data_region]).reshape(-1,1))
converted_data_riskCategory = loaded_OHE_riskCategory.transform(loaded_LE_riskCategory.transform([data_riskCategory]).reshape(-1,1))
converted_data_stakeholder  = loaded_OHE_stakeholder.transform(loaded_LE_stakeholder.transform([data_stakeholder]).reshape(-1,1))
converted_data_riskType     = loaded_OHE_riskType.transform(loaded_LE_riskType.transform([data_riskType]).reshape(-1,1))

np_input = np.concatenate((converted_data_region,
                           converted_data_riskCategory,
                           converted_data_stakeholder,
                           converted_data_riskType,
                           embeddings_riskDescription,
                           embeddings_riskDriver,
                           embeddings_riskImpact
                          ), axis=1)

estimatedRiskScore     = loaded_model.predict(np_input)[0]
estimateRiskScoreRange = [estimatedRiskScore-1, estimatedRiskScore+1]

print("Estimated Risk Score: " estimatedRiskScore)
print("Estimated Risk Score Range: " estimatedRiskScoreRange)
