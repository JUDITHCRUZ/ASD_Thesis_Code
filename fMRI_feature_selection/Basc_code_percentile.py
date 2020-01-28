'''
The BASC code is an example of how the parameters where changed using the specific atlas, matrix and 
brain regions based on the predefined number of ROIs.
Additionally, the list used to collect the score (means,std) were replaced by dictionaries because they are
faster.

'''

#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# MRI Data pre-processing 

# In[1]:


# Original Dataset given by the challenge
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
#  BASC122-Tangent                
#           (1, 2):list(range(7503))
# #############################################################################            
#  BASC064-Tangent          
#           (0,2):list(range(2080))
# #############################################################################            
#  BASC197-Tangent    
          (2, 2):list(range(19503)) 
            
# #############################################################################            
        }
        
        self.atlas = [
            'fmri_basc064', 
            'fmri_basc122', 
            'fmri_basc197' 
           
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


# In[29]:


# Check the Values
features=FeatureExtractor()
training_features=features.fit(data_train, labels_train)
test_features=features.fit(data_test, labels_test)
transformed_features_train=training_features.transform(data_train)
transformed_features_test=test_features.transform(data_test)


# In[31]:


# check shape of the fMRI Data
transformed_features_train.shape


# Feature Extraction using Percentile and Mutual Information (MI) and f_classif 

# In[ ]:


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif,f_classif


# #############################################################################
# Data is collected from the fMRI code


# #############################################################################
# Create a feature-selection transform, a scaler and an instance of LogRegression that we
# combine together to have an full-blown estimator
clf = Pipeline([('scaler', StandardScaler()), ('anova', SelectPercentile(mutual_info_classif)),                
                ('logReg', LogisticRegression(C=20,penalty="elasticnet",l1_ratio=0.01, solver="saga",
                                              multi_class='ovr', n_jobs=-1, max_iter=1000,random_state=1988))])



# #############################################################################
# Plot the cross-validation score as a function of percentile of features
score_means = list()
score_stds = list()
percentiles = np.arange(1, 101).tolist()

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    this_scores = cross_val_score(clf, transformed_features_train,labels_train,cv=10)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())


# In[27]:


import pandas as pd 
df3 = pd.DataFrame(list(zip(percentiles, score_means, score_stds)), 
               columns =['Percentile', 'Mean_Accuracy', 'Std_Accuracy']) 
df3

# #Export values 
# df_features = pd.DataFrame(list(zip(percentiles, score_means, score_stds)), 
#                columns =['Percentile', 'Mean_Accuracy', 'Std_Accuracy']) 
df3.to_csv(r'C:\Users\Ju\Documents\classifier_results\Changes_thesis\Results_after_feature_selection\CV10_mutual_information_Perecentile_tangent\Basc197.csv')


# In[14]:


# apply the classifier and check AUC and Accuracy
clf.fit(transformed_features_train, labels_train)
clf.predict(transformed_features_test)


# In[15]:


from sklearn.metrics import accuracy_score
accuracy_score(labels_test,clf.predict(transformed_features_test))


# In[17]:


from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, clf.predict(transformed_features_test))


# In[ ]:




