# Reduce the correlation between features

import numpy as np
def reduce_redundance(df_train, df_test):
    # Create correlation matrix
    corr_matrix = df_train.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.50)]
    
    # Drop correlated features #     
    df_train_not_correlated= df_train[df_train.columns.drop(to_drop)]

    test_not_correlated = df_test.drop(df_test[to_drop], axis=1)
#     
    return  df_train_not_correlated, test_not_correlated

# ############################################################################################	
	
# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn import  linear_model
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

# Create a logistic regression 
clf = linear_model.LogisticRegression(solver="liblinear",max_iter=7000)

# Create recursive feature eliminator 
rfecv = RFECV(estimator=clf, step=1, scoring="accuracy", cv=10)

# Fit recursive feature eliminator
rfecv.fit(df_train_not_correlated, labels_train)

# Recursive feature elimination
rfecv.transform(df_train_not_correlated)

# Number of best features
rfecv.n_features_

 
# Select variables and calculate test accuracy
cclf_not_corr = data_train[ae1].columns[rfecv.support_]
acc = accuracy_score(labels_test, rfecv.estimator_.predict(data_test[cclf_not_corr]))
print('Number of features selected: {}'.format(rfecv.n_features_))
print('Test Accuracy {}'.format(acc))



################################################################
### Remove the correlations higher than 0.5, using Pearson

### Building correlation Matrix

corr =data_train.loc[:,data_train.dtypes == 'float64'].corr()

not_correlated_features = deepcopy(corr)
#print(cor1)
not_correlated_features =not_correlated_features[abs(cor1)<0.5000]
#display(not_correlated_features)
not_correlated_features.shape


# visualize the results
sns.set(style="white")

# Generate a mask for the upper triangle
mask = np.zeros_like(cor1, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(not_correlated_features, mask=mask, cmap=cmap, vmax=1, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})