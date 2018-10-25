import numpy as np
import csv
from helpers import load_csv_data, predict_labels, create_csv_submission, find_arg_min
from models import LeastSquare_MSE_Model
from features import clean_data

OUTPUT_DIR = 'results/'
COMPUTE_PREDICTION = True

hs = { 
    'degree': np.arange(20), 
    'lambda': np.logspace(-9, -3, 20)
}

y, x, _ = load_csv_data('data/train.csv', sub_sample=False)

myModel = LeastSquare_MSE_Model(x, y)
res = myModel.evaluate(hs, filename=OUTPUT_DIR+'RidgeRegression_MSE_Degree_Model')
h = find_arg_min(res, 'mse')
print(f"Best parameters found : {h}")

if COMPUTE_PREDICTION:
    _, xT, idsT = load_csv_data('data/test.csv', sub_sample=False)
    ws = myModel.get_weights(h)
    ypred = predict_labels(ws, clean_data(xT))
    create_csv_submission(idsT, ypred, "output.csv")
