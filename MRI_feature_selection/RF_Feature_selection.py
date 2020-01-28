### Random forest for feature selection 

### Data pre-processing for gender and reduce the repetition time for 2 decimals
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from problem import get_train_data,get_test_data
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline       
from sklearn.model_selection import cross_val_score 



# Import the data from IMPAC
data_train,labels_train  = get_train_data()
data_test, labels_test = get_test_data()

# #############################################################################   

# Use OneHotEncoder to label gender
le = preprocessing.LabelEncoder()
data_train['participants_sex'] = le.fit_transform(data_train.participants_sex.values)

# Reduce decimals number in repetition time
decimals = 2    
data_train['repetition_time'] = data_train['repetition_time'].apply(lambda x: round(x, decimals))

### do the reduction from labels
data_test['repetition_time'] = data_test['repetition_time'].apply(lambda x: round(x, decimals))
data_test['participants_sex'] = le.fit_transform(data_test.participants_sex.values)

# #############################################################################   

# Standardization 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X=scaler.fit_transform(data_train._get_numeric_data())
scaled_test=scaler.fit_transform(data_test._get_numeric_data())


# feature selection and model evaluation before feature selection in cv
# #############################################################################   

# Create a new random forest classifier for the most important features
clf_mri = RandomForestClassifier(n_estimators=10000, random_state=1988,oob_score=True, n_jobs=-1)
# Train the new classifier on the new dataset containing the most important features
clf_mri.fit(scaled_X, labels_train)

# Create a selector object that will use the random forest classifier to identify 
# features that have an importance of more than median
sfm_mri = SelectFromModel(clf_mri, threshold="mean")
# Train the selector
sfm_mri.fit(scaled_X,labels_train )

# Collect the feature with importance
anatomy=[]
for feature_list_index in sfm_mri.get_support(indices=True):
    anatomy.append(data_train.columns[feature_list_index])
    
# Transform the data to create a new dataset containing only the most important features
# to both the training X and test X data.
X_important_train = sfm_mri.transform(scaled_X)
X_important_test = sfm_mri.transform(scaled_test)


# Create a new random forest classifier for the most important features
clf_mri_features = RandomForestClassifier(n_estimators=10000, random_state=1988,oob_score=True, n_jobs=-1)
# Train the new classifier on the new dataset containing the most important features
clf_mri_features.fit(X_important_train, labels_train)


from sklearn.metrics import accuracy_score
# Apply the full featured classifier to the Test Data
y_important_pred = clf_mri_features.predict(X_important_test)

# View the Accuracy of the Limited Feature Model
print(accuracy_score(labels_test, y_important_pred))




 
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline        
from sklearn.model_selection import cross_val_score
import numpy as np



# #############################################################################   
# Feature selection in CV
feat_sel = RandomForestClassifier(n_estimators=1000, random_state=1988,oob_score=True, n_jobs=-1)
sfm_mri = SelectFromModel(feat_sel, threshold="mean")

clf = RandomForestClassifier(n_estimators=10000, random_state=1988,oob_score=True, n_jobs=-1)
pipe = Pipeline([('scaler', StandardScaler()),('select_from_model',sfm_mri), ('RF',clf)])

# #############################################################################   
# Results in train 
scores_train = cross_val_score(pipe, data_train._get_numeric_data(), labels_train, cv =10, scoring = 'roc_auc')
print(np.mean(scores_train))



