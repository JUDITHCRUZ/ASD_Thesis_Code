'''
This model set up considers every matrix-atlas-feature selection method(fsm)-model combination for the given parameters
the evaluation is performed using the data from every site in 10 cross validation fold split of the data.
the results are saved to the directory location.

'''



#!/usr/bin/env python
# coding: utf-8


# ### Atlas name:
# - "fmri_msdl",
# -  "fmri_harvard_oxford_cort_prob_2mm",
# - "fmri_basc064", 
# - "fmri_basc122",
# - "fmri_basc197",
# - "fmri_craddock_scorr_mean",
# - "fmri_power_2011" 
# 
# ### Matrix type: 
# 
# - "correlation",
# - "partial correlation",
# - "tangent", 
# - "covariance",
# - "precision"

# ### Import the raw data


from problem import get_train_data,get_test_data
data_train, labels_train = get_train_data()
data_test, labels_test = get_test_data()


# ### Extract time-series data

# In[ ]:




import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

# Extract fmri data by atlas and matrix
from nilearn.connectome import ConnectivityMeasure

def _load_fmri(fmri_filenames):
    """Load time-series extracted from the fMRI using a specific atlas."""
    return np.array([pd.read_csv(subject_filename,
                                 header=None).values
                     for subject_filename in fmri_filenames])


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # make a transformer which will load the time series and compute the
        # connectome matrix
        self.transformer_fmri = make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind=matrix, vectorize=True))
    
    def fit(self, X_df, y):
        fmri_filenames = X_df[atlas]
        self.transformer_fmri.fit(fmri_filenames, y)
        return self

    def transform(self, X_df):
        fmri_filenames = X_df[atlas]
        X_connectome = self.transformer_fmri.transform(fmri_filenames)
        X_connectome = pd.DataFrame(X_connectome, index=X_df.index)
        X_connectome.columns = ['connectome_{}'.format(i)
                                for i in range(X_connectome.columns.size)]
        # get the anatomical information
#         X_anatomy = X_df[mri_features_rfecv]
        
        
        return X_connectome
#         return pd.concat([X_connectome, X_anatomy], axis=1)


# ### Cross validation settings

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
def get_cv(X,y):
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    return cv.split(X,y)


# ###  Classifier and Feature Selection Pipeline

# In[ ]:


from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif,f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeClassifier

from sklearn.pipeline import Pipeline, FeatureUnion

config_file = 'config.txt'

class Classifier(BaseEstimator):
    def __init__(self):   
        self.atlas, self.fsm, self.matrix, self.model = self.get_parameters()
        self.percentile_value={
            "fmri_msdl":64,
            "fmri_harvard_oxford_cort_prob_2mm":25,
            "fmri_basc064":98, 
            "fmri_basc122":80,
            "fmri_basc197":74,
            "fmri_craddock_scorr_mean":66,
            "fmri_power_2011":91
        }
        clf2 = self.get_clf(self.model)                   
        fsm2 = self.get_fsm(self.fsm)
        self.clf = make_pipeline(
            StandardScaler(),
            fsm2,
            clf2
        )
    
    def get_parameters(self):
        with open(config_file) as f:
            line = f.read()
            return line.split('\t')

    def get_fsm(self, fsm):
        if fsm == 'elasticnet':
            my_fsm = SelectFromModel(ElasticNetCV(cv=10, max_iter=5000,n_jobs=-1, random_state=1988) , threshold="mean")
        elif fsm == 'lasso':
            my_fsm=SelectFromModel(LassoCV(cv=10,  max_iter=10000, n_jobs=-1, random_state=1988)) 
        elif fsm =='percentile':
            my_fsm=SelectPercentile(mutual_info_classif, percentile=self.percentile_value[self.atlas])        
        else:
            my_fsm=SelectFromModel(RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1,10], cv=10,normalize=True))
        return my_fsm
    
    def get_clf(self, clf_name):
        if clf_name == "LogReg":
            return LogisticRegression(
                C=20, 
                penalty="elasticnet", 
                l1_ratio=0.01,
                solver="saga",
                max_iter=10000,
                random_state=1988
            )
        elif clf_name == "SVC":
             return SVC(
                 C=450, 
                 kernel='rbf', 
                 gamma='auto', 
                 probability=True
             )
        
             
       

    def fit(self, X, y):
        self.clf.fit(X, y)
        print(f'fit {type(self)}')
        return self
        
    def predict(self, X):
        print(f'predict {type(self.clf)}')
        return self.clf.predict(X)

    def predict_proba(self, X):
        print(f'predict_proba {type(self.clf)}')
        return self.clf.predict_proba(X)


# ### Model Evaluation (ROC-AUC, Accuracy, f1, f1_weighted)

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from problem import get_cv

def evaluation(X, y):
    pipe = make_pipeline(FeatureExtractor(), Classifier())
    cv = get_cv(X, y)
    results = cross_validate(pipe, X, y, scoring=['roc_auc', 'accuracy', "f1", "f1_weighted"], cv=cv,
                             verbose=30, return_train_score=True,
                             n_jobs=-1)
    
    return results


# ### Combinations of Atlases, Feature Selection Methods (FSM), Matrices and Models used

# In[ ]:


import itertools as itt


matrix_list=["correlation", "partial correlation", "tangent", "covariance", "precision"]

fsm_list=["elasticnet", "ridge", "lasso","percentile"]

model_list=["Ridge","SVC", "LogReg"]
atlas_list=["fmri_msdl", "fmri_harvard_oxford_cort_prob_2mm",
       "fmri_basc064", "fmri_basc122", "fmri_basc197", "fmri_craddock_scorr_mean", "fmri_power_2011"]

def save_parameters(
    atlas, 
    fsm,
    matrix,
    model
):
    with open(config_file, 'w') as f:
        f.write('\t'.join([atlas, fsm, matrix, model]))
    
results_per_fsm={}
for atlas, fsm, matrix, model in itt.product(atlas_list, fsm_list, matrix_list, model_list):
    print(atlas)
    save_parameters(atlas, fsm, matrix, model)
    results = pd.DataFrame(
        evaluation(
            data_train[[col for col in data_train.columns if col.startswith("fmri")]], 
            labels_train
        )
    )
    results_per_fsm[atlas,matrix,fsm,model]=results
    break 
    


# ### Directory to Save Results by Atlas and Matrix
# The name of every file refers to the model and fsm

# In[ ]:


import os

for i in results_per_fsm.keys():     
    atlas_name=i[0]
    matrix=i[1]
    model=i[3]
    
    my_dir=os.getcwd()   
    file_name=os.path.join(my_dir, atlas_name, matrix)
   
    print(file_name)
    display (results_per_fsm[i])
    if not os.path.exists(file_name):        
        os.makedirs(file_name)
    (results_per_fsm[i]).to_csv(os.path.join(file_name, f'{model}_{i[2]}.csv'))
    

