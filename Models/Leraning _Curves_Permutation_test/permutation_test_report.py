'''
The second time the permutation test was performed, the data used was the BASC197 and the matrix type was tangent
the feature selection method was percentile (66) extracted using mutual_info_classif.
The code produces the visualization of the data distribution and the p value.

'''



#!/usr/bin/env python
# coding: utf-8

# In[6]:


from problem import get_train_data,get_test_data
data_train, labels_train = get_train_data()
data_test, labels_test = get_test_data()


# In[8]:


### 
# Select the data set: atlas name and the matrix type
###

# "covariance" and "precision",
atlas_name="fmri_basc197"
matrix="tangent"
model="logreg"
feature_selection="percentile"

# Atlas values dataframe: used to extract the time-series values.

data_f_MRI= data_train[[col for col in data_train.columns if col.startswith(atlas_name)]]


# In[23]:


# check the atlas name is correct
data_f_MRI


# In[10]:


# Extract the data from the atlases and the matrix type and return data frame based on the fmri data

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
#         X_anatomy = X_df[mri_features_rfecv]
        
        
        return X_connectome
#         return pd.concat([X_connectome, X_anatomy], axis=1)


# Retrive the data

# In[11]:


craddock_tangent=FeatureExtractor()
train_craddock_tangent=craddock_tangent.fit(data_train, labels_train)
test_craddock_tangent=craddock_tangent.fit(data_test, labels_test)


# In[12]:


train_craddock_tangent_tranformed=train_craddock_tangent.transform(data_train)
test_craddock_tangent_transformed=test_craddock_tangent.transform(data_test)


# In[13]:


train_craddock_tangent_tranformed.shape


# In[18]:



# Author:  Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif,f_classif


# #############################################################################
# Loading  the fmri dataset

X = train_craddock_tangent_tranformed
y = labels_train
n_classes = np.unique(y).size

# Create not correlated noisy data
random = np.random.RandomState(seed=0)
E = random.normal(size=(len( X), 2200))

# Add noisy data to the informative features for make the task harder
X = np.c_[X, E]


log_reg= make_pipeline(StandardScaler(), 
                       SelectPercentile(mutual_info_classif, percentile=66),
                       LogisticRegression(C=20,penalty="elasticnet",l1_ratio=0.01, solver="saga",
                       max_iter=10000,random_state=1988))
       
# estimator = LogisticRegression(C=0.01,penalty="elasticnet",l1_ratio=0.001, solver="saga", max_iter=10000,random_state=1988)
cv = StratifiedKFold(3)

score, permutation_scores, pvalue = permutation_test_score(
    log_reg,  X , labels_train, scoring="roc_auc", cv=cv, n_permutations=50, n_jobs=-1,verbose=30)

print("Classification score %s (pvalue : %s)" % (score, pvalue))

# #############################################################################


# In[21]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
# sns.set(style = 'white', context='notebook', rc={"lines.linewidth": 2.5})
sns.set(palette="colorblind")


# In[22]:


# View histogram of permutation scores
fig=plt.figure(figsize=(8,6))
plt.hist(permutation_scores, 20, label='Permutation scores',
         edgecolor='black')

 
font = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 10,   }
    


ylim = plt.ylim()
# BUG: vlines(..., linestyle='--') fails on older versions of matplotlib
# plt.vlines(score, ylim[0], ylim[1], linestyle='--',
#          color='g', linewidth=3, label='Classification Score'
#          ' (pvalue %s)' % pvalue)
# plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',
#          color='k', linewidth=3, label='Luck')
plt.plot(2 * [score], ylim, '--r', linewidth=3,
         label='Classification Score'
         ' (pvalue %s)' % pvalue)
plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Random')

plt.ylim(ylim)
plt.legend()
plt.xlabel('ROC_AUC')

plt.savefig(r'C:\Users\Ju\Documents\Learning_curves\logre_permutation100_roc_auc.png', bbox_inches="tight")
plt.show()

