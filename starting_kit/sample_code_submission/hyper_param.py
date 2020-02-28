from sys import path
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from data_manager import DataManager
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

import scipy as sp
from data_io import write
from libscores import get_metric

from model1 import model1
from model2 import model2
from model3 import model3
from model6 import model6
from model7 import model7
import time

model_dir = 'sample_code_submission/'                        # Change the model to a better one once you have one!
#model_dir = '../FILES/pretty_good_sample_code_submission/'
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
data_name='xporters'
trained_model_name = model_dir + data_name

#impossible de comprendre le fonctionement des methodes
def grid(model,data_name= 'xporters',data_dir='./input_data/'):
    temps_a=time.time()
    D = DataManager(data_name, data_dir , replace_missing=True)
    M=model()
    X_train = D.data['X_train']
    Y_train = D.data['Y_train']
    #if not(M.is_trained) : M.fit(X_train, Y_train)
    param = M.mod.get_params()
    print(param)
    
def random(model,data_name= 'xporters',data_dir='./input_data/'):
    temps_a=time.time()
    D = DataManager(data_name, data_dir , replace_missing=True)
    M=model()
    X_train = D.data['X_train']
    Y_train = D.data['Y_train']
    #if not(M.is_trained) : M.fit(X_train, Y_train)
    param = M.mod.get_params()
    print(param)