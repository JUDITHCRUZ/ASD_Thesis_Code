'''
The third best solution uses only the combination of BASC Atlases to extract the data and provide these regions to 
the data extractor Class.
The results are highly optimistic in both cases train and test. The model evaluation results are 0.96 ROC-AUC and 0.90 for accuracy_score.
The test values are up to 0.95 accuracy and ROC-AUC.

'''


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from problem import get_train_data

data_train, labels_train = get_train_data()
from problem import get_test_data

data_test, labels_test = get_test_data()


# In[ ]:


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
          (1, 2): [3445, 4918, 3257, 6339, 5443, 1902, 1190, 7051, 1962, 1554, 4326, 638, 6989, 6387,
                   2352, 253, 2959, 4839, 6833, 6880, 858, 4822, 6665, 6191, 6674, 2599, 191, 4626, 5034, 6097, 
                   4125, 953, 2991, 3366, 6653, 5327, 2455, 1232, 4404, 3601, 6772, 3563, 811, 4001, 4871, 1743, 
                   266, 4103, 6677, 6929, 5540, 4042, 7453, 3562, 6907, 4864, 3395, 655, 6205, 5950, 2187, 3669, 
                   3576, 7115, 2958, 6921, 5408, 4758, 1831, 3162, 5614, 547, 3352, 5911, 4694, 7014, 3989, 1524,
                   613, 2039, 4421, 1674, 1867, 4378, 6234, 5270, 7435, 1541, 6920, 1311, 1742, 3058, 6003, 5956, 
                   5835, 236, 4320, 5433, 657, 6780, 5551, 5076, 2450, 4316, 454, 1131, 5045, 5618, 1725, 555, 4459,
                   1816, 2221, 3357, 2545, 1336, 2135, 4352, 1913, 4429, 3375, 1226, 5755, 7439, 444, 5099, 7049, 5764,                   
                   6754, 5662, 3360, 5430, 3889, 6791, 935, 627, 3600, 3452, 5365, 4237, 5434, 4812, 452, 6545, 2516, 
                   1957, 3215, 1542, 5984, 2290, 5220, 2878, 2514, 5397, 4939, 6226, 6304, 6860, 7108, 2077, 6260, 3309,
                   2080, 7385, 2487, 5043, 4449, 3401, 337, 3529, 6960, 5184, 4786, 5630, 1405, 4295, 2303, 5260, 5287, 
                   7384, 5222, 4309, 1531, 6553, 5155], 
            
          (0, 2): [2019, 2070, 1266, 1387, 309, 944, 1421, 460, 1938, 1715, 1139, 1400, 1335, 370, 2009, 400, 1908, 237,
                   1472, 1279, 417, 1437, 52, 1328, 838, 1469, 516, 2073, 473, 1357, 1779, 327, 806, 979, 1981, 748, 358,
                   1579, 587, 323, 879, 1806, 1074, 1410, 1822, 1616, 1170, 1082, 542, 1182, 295, 365, 1802, 2008, 1558, 
                   74, 95, 2037, 874, 393, 1663, 1841, 1125, 1498, 1760, 471, 1010, 200, 688, 526, 2054, 82, 420, 1854, 
                   671, 1402, 1351, 896, 1988, 388, 1381, 1202, 840, 1646, 251, 203, 1640, 926, 1717, 1096, 15, 541, 651, 
                   970, 1089, 1722, 1599, 1285, 1877, 84, 211, 185, 1614, 1161, 771, 45, 227, 1420, 1444, 1681, 283, 1565,
                   299, 1846, 476, 344, 788, 701, 1117, 891, 807, 726, 1618, 172, 320, 616, 1298, 1783, 361, 1032, 754, 
                   737, 207, 829, 617, 1605, 1355, 2062, 1463, 722, 192, 638, 233, 2025, 1954, 707, 1944, 1337, 1283, 834, 
                   1478, 1887, 128, 791, 1260, 633, 2058, 1678, 1062, 900, 1882, 269, 687, 1045, 1934, 826, 1775, 1649, 
                   844, 161, 1689, 540, 1609, 1394, 1699, 1984, 1889, 1808, 225, 1427, 5, 725, 28, 753, 1383], 
            
          (2, 2): [3836, 9156, 15156, 17142, 4395, 8342, 5053, 1206, 11612, 15518, 306, 17977, 2242, 12923, 12722, 5547, 
                   12138, 6475, 2335, 2972, 17121, 16137, 11514, 16250, 16367, 12628, 18841, 5335, 9167, 3103, 4498, 5089,
                   13554, 12755, 2511, 7204, 18088, 7616, 14505, 9168, 13193, 7499, 4677, 1612, 3753, 18118, 13359, 6334, 
                   11226, 6237, 2836, 16854, 1542, 7575, 14283, 5201, 14490, 5706, 5486, 18998, 17449, 10238, 18994, 7356,
                   8332, 6241, 751, 11583, 13962, 17374, 11434, 18202, 415, 8707, 11790, 2581, 25, 2466, 3864, 17593, 10872,
                   13835, 16139, 3629, 5844, 3304, 12014, 2240, 13379, 5629, 7632, 19254, 8567, 4998, 15343, 4059, 8650, 
                   8035, 14023, 8394, 645, 15215, 18863, 12663, 16914, 16448, 2321, 15264, 1548, 1544, 7692, 17879, 5205,
                   15998, 11975, 3839, 9381, 9457, 18391, 7168, 12406, 3763, 16687, 5812, 3077, 18485, 12469, 11033, 13495,
                   8963, 14853, 4645, 12800, 17710, 1896, 11997, 15049, 630, 17458, 296, 16934, 6177, 13680, 16088, 8303, 
                   17410, 2367, 14697, 14701, 7682, 12818, 469, 17270, 18384, 18484, 5584, 16588, 7247, 1225, 876, 12556, 
                   14946, 5022, 9867],
            (5,2):[478]
        }
        
        self.atlas = [
            'fmri_basc064', 
            'fmri_basc122', 
            'fmri_basc197', 
            'fmri_craddock_scorr_mean', 
            'fmri_harvard_oxford_cort_prob_2mm', 
            'fmri_msdl',
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


# In[ ]:


features= FeatureExtractor()
training_features=features.fit(data_train, labels_train)
test_features=features.fit(data_test, labels_test)


# In[ ]:


transform_training_features=training_features.transform(data_train)
transform_test_features=test_features.transform(data_test)


#
# In[ ]:


transform_training_features.columns


# In[ ]:


from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, random_state=1994))

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self
        
    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


# In[ ]:


clf_erd=Classifier()
train_clf=clf_erd.fit(transform_training_features, labels_train)
predict_test=train_clf.predict(transform_test_features)
p_prob=train_clf.predict_proba(transform_test_features)


# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test, p_prob[:,1])


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(labels_test,predict_test,)



import pandas
from sklearn.metrics import classification_report
report_BASC=(classification_report(labels_test, predict_test, output_dict=True))
df = pandas.DataFrame(report_BASC).transpose()


# In[ ]:


df.to_csv(r'basc_classifier_results.csv')




