'''
Due to time constrains (the second time) this analysis did not make it. However, I changed the code to add the 
feature selection as part of the pipeline to reduce the overfitting due to data leakage.
These results were NOT Reported in the thesis report.
'''

#!/usr/bin/env python
# coding: utf-8

# ### Import Raw Data

# In[40]:


from problem import get_train_data,get_test_data

data_train, labels_train = get_train_data()
data_test, labels_test = get_test_data()


# ### Select Atlas and Matrix
# Based on performance Craddock and Tangent were selected

# In[41]:


matrix="tangent"
atlas_name="fmri_craddock_scorr_mean"


# In[42]:


# Voting Ensemble for Classification

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
        fmri_filenames = X_df[atlas_name]
        self.transformer_fmri.fit(fmri_filenames, y)
        return self

    def transform(self, X_df):
        fmri_filenames = X_df[atlas_name]
        X_connectome = self.transformer_fmri.transform(fmri_filenames)
        X_connectome = pd.DataFrame(X_connectome, index=X_df.index)
        X_connectome.columns = ['connectome_{}'.format(i)
                                for i in range(X_connectome.columns.size)]
        # get the anatomical information
        X_anatomy = X_df['participants_site']
        
        
#         return X_connectome
        return pd.concat([X_connectome, X_anatomy], axis=1)


# ### Allocate the information in a variable

# In[43]:


sol_feature=FeatureExtractor()
training_set_basc_sol=sol_feature.fit(data_train, labels_train)
test_set_basc_sol=sol_feature.fit(data_test, labels_test)


# In[44]:


sol_transform_train=training_set_basc_sol.transform(data_train)
sol_transform_test=test_set_basc_sol.transform(data_test)


# In[46]:


# Voting Ensemble for Classification
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif,f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import StandardScaler

## Define the Group information 
sites=sol_transform_train["participants_site"]
cv = LeaveOneGroupOut()

estimators = []

ada_boost = make_pipeline(StandardScaler(), SelectPercentile(mutual_info_classif, percentile=74),
                          AdaBoostClassifier(LogisticRegression(C=250,penalty="elasticnet",l1_ratio=0.01, 
                                                                solver="saga",max_iter=1000,n_jobs=-1)))
estimators.append(('adaboost', ada_boost ))

# create the ensemble model
ensemble = VotingClassifier(estimators,n_jobs=-1)
results = pd.DataFrame(model_selection.cross_val_score(ensemble, sol_transform_train,labels_train, cv=3,groups=sites,verbose=30))
print(results.mean(),results.std())


# In[1]:


