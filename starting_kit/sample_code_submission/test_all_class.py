'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
'''


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

def test_score(modele,data_name= 'xporters',data_dir='./input_data/'):
    temps_a=time.time()
    D = DataManager(data_name, data_dir , replace_missing=True)
    M=modele()
    X_train = D.data['X_train']
    Y_train = D.data['Y_train']
    if not(M.is_trained) : M.fit(X_train, Y_train)                     
    Y_hat_train = M.predict(D.data['X_train']) # Optional, not really needed to test on taining examples
    Y_hat_valid = M.predict(D.data['X_valid'])
    Y_hat_test = M.predict(D.data['X_test'])
    M.save(trained_model_name)                 
    result_name = result_dir + data_name
    write(result_name + '_train.predict', Y_hat_train)
    write(result_name + '_valid.predict', Y_hat_valid)
    write(result_name + '_test.predict', Y_hat_test)
    metric_name, scoring_function = get_metric()
    scores = cross_val_score(M, X_train, Y_train, cv=5, scoring=make_scorer(scoring_function))
    print('\nCV score (95 perc. CI): %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    temps_b=time.time()-temps_a
    print("voici le temps d'éxécution du modele: ",modele," ",temps_b)
    return scores.mean(),scores.std() * 2


def test_comparaison(data='xporters', data_dir='./input_data/'):
    model_list=[model1,model2,model3,model6,model7]
    p=1
    for i in model_list:
        print('model ',p )
        score, incertitude =test_score(i,data_name='xporters',data_dir='./input_data/')
        print("cross validation ",score ,"incertitude ",incertitude )
        p+=1
    