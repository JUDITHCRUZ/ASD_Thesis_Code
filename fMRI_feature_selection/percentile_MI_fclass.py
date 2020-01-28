'''
This script was used for every atlas using only the tangent matrix to explore the percentile
value that best classify the ASD condition, the results were saved.
The values given to extract the data are the atlas and the matrix as well as the maximum number of regions per atlas
***The Mutual Information Parameter as well as the f_class were modified as needed.


'''


# ############################################################################# 
#Original Dataset given by the challenge
# #############################################################################   
from problem import get_train_data,get_test_data

data_train, labels_train = get_train_data()
data_test, labels_test = get_test_data()

# #############################################################################   
# fMRI feature extractor by Atlas and Matrix


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from nilearn.connectome import ConnectivityMeasure

def _load_fmri(fmri_filenames):
    return np.array([pd.read_csv(subject_filename, header=None).values for subject_filename in fmri_filenames])

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        
        self.choice = {
#   Craddock-Tangent                
#           (0, 2):list(range(31125))
# #############################################################################            
#  power-Tangent          
          (1,2):list(range(34980))
#             (0,2):list(range(3))
# #############################################################################            
        
        }
        
        self.atlas = [
            'fmri_craddock_scorr_mean', 
            'fmri_power_2011', 
             
           
        ]
        self.kind = [
            'correlation', 
            'partial correlation', 
            'tangent', 
            'covariance', 
            'precision',
        ]
           
      
        
        self.list_transformer_fmri = [make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind=self.kind[elem[1]], vectorize=True)
        ) for elem in self.choice.keys()]        
    
    def fit(self, X_df, y):
        for i,elem in enumerate(self.choice.keys()):            
            self.list_transformer_fmri[i].fit(X_df[self.atlas[elem[0]]], y)
        return self

    def transform(self, X_df):
        list_X_connectome = []
        for i,elem in enumerate(self.choice.keys()):
            X_connectome = self.list_transformer_fmri[i].transform(X_df[self.atlas[elem[0]]])
            X_connectome = pd.DataFrame(X_connectome, index=X_df.index)
            X_connectome = X_connectome.iloc[:, self.choice[elem]]
            list_X_connectome.append(X_connectome) 
        
        X_connectome = pd.concat(list_X_connectome, axis=1)
        X_connectome.columns = ['connectome_{}'.format(i) for i in range(X_connectome.columns.size)]
        return X_connectome       
        
  # print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif,f_classif
import time



def feature_selection(data,clf:str, labels):

    start = time.process_time()


    # #############################################################################
    # Data is collected from the fMRI code

    # #############################################################################
    # Create a feature-selection transform, a scaler and an instance of LogRegression that we
    # combine together to have an full-blown estimator
    clf = Pipeline([('scaler', StandardScaler()), ('univariate_sel', SelectPercentile(mutual_info_classif)),                
                    ('logReg', LogisticRegression(C=20,penalty="elasticnet",l1_ratio=0.01, solver="saga",
                                                  multi_class='ovr', n_jobs=-1, max_iter=1000,random_state=1988))])



    # #############################################################################

    data_features_acc={}
    data_features_std={}

    percentiles=np.arange(0, 101).tolist()

    for percentile in percentiles:
        clf.set_params(anova__percentile=percentile)
        this_scores = cross_val_score(clf, data,labels,cv=3)
        data_features_acc[percentile]=this_scores.mean()
        data_features_std[percentile]=this_scores.std()


    print(time.process_time() - start)

 
    return clf, data_features_acc,data_features_std
clf_power_MI,data_features_acc_PW_MI,data_features_std_PW_MI=feature_selection(transformed_features_power_train,"power",labels_train)

    