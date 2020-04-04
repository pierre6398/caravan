from sys import path
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from data_manager import DataManager
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import scipy as sp
from data_io import write
from libscores import get_metric
import time

model_dir = 'sample_code_submission/'                        # Change the model to a better one once you have one!
#model_dir = '../FILES/pretty_good_sample_code_submission/'
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
data_name='xporters'
trained_model_name = model_dir + data_name

default_params = {
    'alpha': 0.9, 
    'ccp_alpha': 0.0,
    'criterion': 'friedman_mse',
    'init': None,
    'learning_rate': 0.1,
    'loss': 'ls',
    'max_depth': 3,
    'max_features': None,
    'max_leaf_nodes': None, 
    'min_impurity_decrease': 0.0, 
    'min_impurity_split': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 100,
    'n_iter_no_change': None,
    'presort': 'deprecated',
    'random_state': None,
    'subsample': 1.0,
    'tol': 0.0001,
    'validation_fraction': 0.1,
    'verbose': 0,
    'warm_start': False
}

#without preprocess
param_grid_1 = {
        'learning_rate' : np.linspace(0.05, 0.15, 5),
        'max_depth' : np.arange(5) + 8,
        'n_estimators' : np.arange(5) * 10 + 80,
        'min_samples_leaf' : np.arange(5) + 1
    }
after_grid_1 = {'learning_rate': 0.125, 'max_depth': 10, 'min_samples_leaf': 3, 'n_estimators': 120}

usefull_param = []

#impossible de comprendre le fonctionement des methodes
def grid(data_name= 'xporters',data_dir='./input_data/'):
    temps_a = time.time()
    D = DataManager(data_name, data_dir , replace_missing=True)
    M = GradientBoostingRegressor()
    X_train = D.data['X_train']
    Y_train = D.data['Y_train']
    
    param_grid = {
        'learning_rate' : np.linspace(0.05, 0.15, 5),
        'max_depth' : np.arange(5) + 8,
        'n_estimators' : np.arange(5) * 10 + 80,
        'min_samples_leaf' : np.arange(5) + 1
    }
    
    GSCV = GridSearchCV(M, param_grid, verbose=2, n_jobs=8)
    GSCV.fit(X_train,Y_train)
    
    temps_b = time.time()
    
    print(temps_b - temps_a)
    print(GSCV.best_params_)
    
def random(data_name= 'xporters',data_dir='./input_data/'):
    temps_a=time.time()
    D = DataManager(data_name, data_dir , replace_missing=True)
    M=model()
    X_train = D.data['X_train']
    Y_train = D.data['Y_train']
    #if not(M.is_trained) : M.fit(X_train, Y_train)
    param = M.mod.get_params()
    res = RandomizedSearchCV(M.mod, {"max_depth":[1,2,3,4,5,6,7,8,9],"random_state":[0], "n_estimators":[100]})
    res.fit(X_train,Y_train)
    print(res.best_params_)
    
    