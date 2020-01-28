''' 
Individual Revision of performance of LassoCV, ElasticNetCV and RidgeClassifierCV
The fMRI data is extracted, standardized and provided to every feature selection method already mentioned.
The feature subset is also collected but the information is labeled as connectome and a consecutive integer in every case, so it 
was not so informative to report these names in the report.

'''

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


#Import raw data
from problem import get_train_data,get_test_data
data_train, labels_train = get_train_data()
data_test, labels_test = get_test_data()

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
            ConnectivityMeasure(kind='partial correlation', vectorize=True))
    
    def fit(self, X_df, y):
        fmri_filenames = X_df['fmri_basc197']
        self.transformer_fmri.fit(fmri_filenames, y)
        return self

    def transform(self, X_df):
        fmri_filenames = X_df['fmri_basc197']
        X_connectome = self.transformer_fmri.transform(fmri_filenames)
        X_connectome = pd.DataFrame(X_connectome, index=X_df.index)
        X_connectome.columns = ['connectome_{}'.format(i)
                                for i in range(X_connectome.columns.size)]
        # get the anatomical information
#         X_anatomy = X_df[main_list3]
        
        
        # concatenate both matrices
        return X_connectome
		
		
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

# Scaling data
def scalefmri(connectome_train,connectome_test ):
    
    robustscaler_by_atlas = RobustScaler()
    standardscaler_by_atlas = StandardScaler()
    robustscaler_train = robustscaler_by_atlas.fit_transform(connectome_train)
    robustscaler_test = robustscaler_by_atlas.fit_transform(connectome_test)

    #standarscaler
    standarscaler_train = standardscaler_by_atlas.fit_transform(connectome_train)
    standarscaler_test= standardscaler_by_atlas.fit_transform(connectome_test)

    return standarscaler_train,standarscaler_test,robustscaler_train,robustscaler_test


from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

def lassoCV_feature_sel(fmri_data, labels,scaled_fmri_data):

    # We use the base estimator LassoCV
    clf_atlas_name = LassoCV(cv=10,  max_iter=10000,n_jobs=-1, random_state=1988 )
    
    # Set a minimum threshold value and use the atlas
    sfm_atlas_name = SelectFromModel(clf_atlas_name, threshold="0.25*mean")
    sfm_atlas_name.fit(scaled_fmri_data , labels)
    n_features_atlas_name = sfm_atlas_name.transform(scaled_fmri_data).shape[1]

    # Extracting the index of important features
    feature_idx_atlas = sfm_atlas_name.get_support()

    # Using the index to print the names of the important variables
    featrues_with_support_atlas = fmri_data.columns[feature_idx_atlas]
    # Get the list of column names based on the indices
    my_cols_lasso = [fmri_data.columns.get_loc(c) for c in featrues_with_support_atlas if c in fmri_data]
    features_lasso = fmri_data.columns[my_cols_lasso]
    df_lasso_features = fmri_data[features_lasso]
    

    return df_lasso_features	
	
	
	
	from sklearn.linear_model import ElasticNetCV
def elasticnetCV_feature_not_scaled(fmri_data, labels):
    
    clf_elnet = ElasticNetCV(cv=10, max_iter=50000,n_jobs=-1, random_state=1988)    
    clf_elnet.fit(scaled_fmri_data, labels) 
    
    sfm = SelectFromModel(clf_elnet, threshold="mean")
    sfm.fit(fmri_data, labels)
    n_features = sfm.transform(fmri_data).shape[1]

    # Extracting the index of important features
    feature_idx = sfm.get_support()

    # Using the index to print the names of the important variables
    featrues_with_support_atlas = fmri_data.columns[feature_idx]
    
    my_cols_elnet = [fmri_data.columns.get_loc(c) for c in featrues_with_support_atlas if c in fmri_data]
    features_elnet = fmri_data.columns[my_cols_elnet]
    df_elnet_features = fmri_data[features_elnet]
    return df_elnet_features
	
from sklearn.linear_model import RidgeClassifierCV
def ridgeCV_feature_sel(fmri_data, labels,scaled_fmri_data):
    
    clf_ridge = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1,10],cv=10,normalize=True)    
    clf_ridge.fit(scaled_fmri_data, labels) 
    
    sfm = SelectFromModel(clf_ridge, threshold=-np.inf, max_features=50)
    sfm.fit(scaled_fmri_data, labels)
    n_features = sfm.transform(scaled_fmri_data).shape[1]
    feature_idx = sfm.get_support()
    feature_name = fmri_data.columns[feature_idx]
    df_features=fmri_data[feature_name]

     
    return  df_features