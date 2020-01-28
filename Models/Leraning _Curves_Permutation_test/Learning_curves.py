'''
This general template was used to compare the performance of the model as the sample number was increased.
It also provided an idea of how the trained model behaved compared to the tested one.

'''



#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Adapted from https://scikit-learn.org/stable/modules/learning_curve.html


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
sns.set(palette="colorblind")


# In[2]:


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    
   

    
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 10,
            }
    
    fig=plt.figure(figsize=(5,3))

    plt.title(title, fontdict= font)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples", fontdict=font)
    plt.ylabel("Score",  fontdict=font)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid(b=True, which='both', color='w', linewidth=1.0)  
#     

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="blue")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="orange")
    plt.plot(train_sizes, train_scores_mean, 'o--', color="blue",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, '--o', color="orange",
             label="Cross-validation score")
    
    plt.legend(facecolor='white',loc="best")

    fig.savefig("learning_Curve.pdf", dpi=300,bbox_inches='tight')
    plt.close(fig)

    return plt

# #############################################################################
# Data-sets
from problem import get_train_data,get_test_data

data_train, labels_train = get_train_data()
X, y = data_train._get_numeric_data(), labels_train

# #############################################################################

# MODELS to Check the code works

title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=34, test_size=0.2, random_state=42)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, ylim=(0.5, 1.01), cv=cv, n_jobs=4)


title = r"Learning Curves (LogRegression, liblinear, $\gamma=0.1$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=34, test_size=0.2, random_state=42)
estimator = LogisticRegression(C=1.0,solver='liblinear', random_state=42)
plot_learning_curve(estimator, title, X, y, (0.1, 1.01), cv=cv, n_jobs=4)

plt.show()


# In[ ]:




