import numpy as np
from skfeature.utility.mutual_information import su_calculation


from sklearn.preprocessing import LabelEncoder 
# Encode values
data_train['participants_sex'] = LabelEncoder().fit_transform(data_train['participants_sex'])
data_test['participants_sex'] = LabelEncoder().fit_transform(data_test['participants_sex'])

## Remove the fmri data
# data_train.dtypes==object
# fmri_basc064, fmri_basc122, fmri_basc197, fmri_craddock_scorr_mean, fmri_harvard_oxford_cort_prob_2mm            
# fmri_motions, fmri_msdl, fmri_power_2011           
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_data_train= data_train.select_dtypes(include=numerics)
numeric_data_test = data_test.select_dtypes(include=numerics)

def merit_calculation(X, y):
    """
    This function calculates the merit of X given class labels y, where
    merits = (k * rcf)/sqrt(k+k*(k-1)*rff)
    rcf = (1/k)*sum(su(fi,y)) for all fi in X
    rff = (1/(k*(k-1)))*sum(su(fi,fj)) for all fi and fj in X
    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    Output
    ----------
    merits: {float}
        merit of a feature subset X
    """

    n_samples, n_features = X.shape
    rff = 0
    rcf = 0
    for i in range(n_features):
        fi = X[:, i]
        rcf += su_calculation(fi, y)
        for j in range(n_features):
            if j > i:
                fj = X[:, j]
                rff += su_calculation(fi, fj)
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits


def cfs(X, y):
    """
    This function uses a correlation based heuristic to evaluate the worth of features which is called CFS
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    Output
    ------
    F: {numpy array}
        index of selected features
    Reference
    ---------
    Zhao, Zheng et al. "Advancing Feature Selection Research - ASU Feature Selection Repository" 2010.
    """

    n_samples, n_features = X.shape
    F = []
    # M stores the merit values
    M = []
    while True:
        merit = -100000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                # calculate the merit of current selected features
                t = merit_calculation(X[:, F], y)
                if t > merit:
                    merit = t
                    idx = i
                F.pop()
        F.append(idx)
        M.append(merit)
        if len(M) > 5:
            if M[len(M)-1] <= M[len(M)-2]:
                if M[len(M)-2] <= M[len(M)-3]:
                    if M[len(M)-3] <= M[len(M)-4]:
                        if M[len(M)-4] <= M[len(M)-5]:
                            break
    return np.array(F)
	
cfs_list_features=cfs(numeric_data_train.to_numpy(copy=True), labels_train)	
	
# Sub-set of features CFS in a list
Features=list(numeric_data_train.columns.to_numpy()[cfs_list_features])

# #############################################################################################################

# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import rfecv_cfs
from sklearn import  linear_model
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score



# Create a linear regression
clf = linear_model.LogisticRegression(solver="liblinear",max_iter=7000)
# clf  = SVC(kernel="linear")
# Create recursive feature eliminator that scores features by mean squared errors
rfecv_cfs = rfecv_cfs(estimator=clf, step=1, scoring="accuracy", cv=10)

# Fit recursive feature eliminator

rfecv_cfs.fit(data_train[Features], labels_train)

# Recursive feature elimination
rfecv_cfs.transform(data_train[Features])

# Number of best features
rfecv_cfs.n_features_


 
# Select variables and calulate test accuracy
cclf_not_corr = data_train[Features].columns[rfecv_cfs.support_]
acc = accuracy_score(labels_test, rfecv_cfs.estimator_.predict(data_test[cclf_not_corr]))
print('Number of features selected: {}'.format(rfecv_cfs.n_features_))
print('Test Accuracy {}'.format(acc))