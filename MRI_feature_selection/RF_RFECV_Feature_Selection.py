'''
Script to try Random Forest Classifier using Recursive Features Elimination in Cross Validation
'''



#Importing the data from IMPAC for all subjects

from problem import get_train_data
data_train, labels_train = get_train_data()
from problem import get_test_data

data_test, labels_test = get_test_data()

# Original Dataset given by the challenge
# #############################################################################   
from problem import get_train_data,get_test_data

data_train, labels_train = get_train_data()
data_test, labels_test = get_test_data()

# #############################################################################   
# Train Dataset changes 

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import numpy as np

# Copy df
data_train_with_labels=data_train

# Decimal Reduction 
decimals = 2
decimals_age = 0

data_train_with_labels['participants_age'] = data_train_with_labels['participants_age'].apply(lambda x: round(x,decimals_age))
data_train_with_labels['repetition_time'] = data_train_with_labels['repetition_time'].apply(lambda x: round(x, decimals))


# Encode gender as two different classes, Male=1, Female=-1
data_train['participants_sex'].replace({ 'F': -1,'M': 1}, inplace= True)
data_train['Group'] =labels_train

# Represent age as a percentage
data_train["ratio_age"]=data_train.participants_age/100

# Data stratification in Infants  (0-12 years), Teens(12-20), Adults(21++)

data_train['age_stratification'] = 'Contraction' 
# data_train[(data_train.participants_age>=0)&(data_train.participants_age<=12)]
data_train.loc[(data_train.participants_age>=0)&(data_train.participants_age<=12), 'age_stratification'] = 1
data_train.loc[(data_train.participants_age>12)&(data_train.participants_age<=20), 'age_stratification'] = 2
data_train.loc[(data_train.participants_age>20)&(data_train.participants_age<=100), 'age_stratification'] = 3

# Remove  set
data_train=data_train.loc[(data_train.fmri_select==1)]
data_train=data_train.loc[(data_train.anatomy_select!=0)]
# Exploring site data, this did not presented interesting results
# data_train=data_train.loc[(data_train.anatomy_select!=3)]
# data_train=data_train.loc[(data_train.participants_site!=6)]
# data_train=data_train.loc[(data_train.participants_site!=8)]

# Re-index and change the labels based on the filters
columns=data_train.columns.values
data_train = data_train.reindex(columns=columns)
labels_train=list(data_train.Group.values)

del data_train['Group']
del data_train['fmri_select']
##########################################################################################################################

#Same changes in test set

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import numpy as np

data_test_with_labels=data_test

decimals = 2
decimals_age = 0


data_test_with_labels['participants_age'] = data_test_with_labels['participants_age'].apply(lambda x: round(x,decimals_age))
data_test_with_labels['repetition_time'] = data_test_with_labels['repetition_time'].apply(lambda x: round(x, decimals))

data_test['participants_sex'].replace({ 'F': -1,'M': 1}, inplace= True)

data_test['Group'] = labels_test
data_test["ratio_age"]=data_test.participants_age/100

data_test['age_stratification'] = 'Contraction' 
# data_test[(data_test.participants_age>=0)&(data_test.participants_age<=12)]
data_test.loc[(data_test.participants_age>=0)&(data_test.participants_age<=12), 'age_stratification'] = 1
data_test.loc[(data_test.participants_age>12)&(data_test.participants_age<=20), 'age_stratification'] = 2
data_test.loc[(data_test.participants_age>20)&(data_test.participants_age<=100), 'age_stratification'] = 3

data_test=data_test.loc[(data_test.fmri_select==1)]
data_test=data_test.loc[(data_test.anatomy_select!=0)]
# data_test=data_test.loc[(data_test.anatomy_select!=3)]
# data_test=data_test.loc[(data_test.participants_site!=6)]
# data_test=data_test.loc[(data_test.participants_site!=8)]

columns=data_test.columns
data_test = data_test.reindex(columns=columns)
labels_test=list(data_test.Group.values)
del data_test['Group']
del data_test['fmri_select']

######################################################################################################################

import pandas as pd
from pandas import Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


#fitting on train 
rf = RandomForestClassifierWithCoef(n_estimators=1000, min_samples_leaf=5, n_jobs=-1)
rfecv = RFECV(estimator=rf, step=1, cv=3, scoring='accuracy', verbose=30)
selector=rfecv.fit(data_train._get_numeric_data(), labels_train)


# collect only the important features
df_important_features_train = data_train._get_numeric_data()[data_train._get_numeric_data().columns[rfecv.get_support()]]
df_important_features_test = data_test._get_numeric_data()[data_test._get_numeric_data().columns[rfecv.get_support()]]
# selector.get_support()

# accuracy in test
from sklearn.metrics import  accuracy_score
accuracy_score(labels_test,rfecv.predict(data_test._get_numeric_data())

