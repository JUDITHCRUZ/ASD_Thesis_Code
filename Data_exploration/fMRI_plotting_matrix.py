'''
fMRI data dimensionality reduction visualization using nilearn tools to visualize matrices and plot connectome relations and strength.

'''

#!/usr/bin/env python
# coding: utf-8

# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os 
import nibabel as nib
from nilearn import plotting
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from scipy import stats
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import LabelEncoder 
import seaborn as sns 
import scipy
import decimal
from matplotlib.backends.backend_pdf import PdfPages
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style = 'white', context='notebook', rc={"lines.linewidth": 2.5})
sns.set(palette="colorblind")


# In[2]:


# install needed dependencies 
import sys
get_ipython().system('{sys.executable} -m pip install xlrd')


# In[3]:


#Importing the data from IMPAC for all subjects

from problem import get_train_data
data_train, labels_train = get_train_data()
from problem import get_test_data

data_test, labels_test = get_test_data()


# In[4]:


def data_fuctions(df,labels):
'''
This function adds the group labels to the data frame in order to visualize the matrices by group and plot_connectomes
based on groups.


'''

    df ['participants_sex'] = LabelEncoder().fit_transform(data_train['participants_sex'])
   
    df['condition'] = labels
    df['Group'] = labels
    df['Group'].replace({ 0: 'TC',1: 'ASD'}, inplace= True)
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df.select_dtypes(include=numerics)    
    df["participants_age"]=df["participants_age"].round(0).astype(int)
    
    data_train_anatomy_area = df[[col for col in data_train.columns if  col.endswith('site') or col.endswith('area') or col.startswith('Group')]]
    data_train_anatomy_tickness =df[[col for col in data_train.columns if col.endswith('thickness') or col.startswith('Group')]]
  
   
    return df,data_train_anatomy_area,data_train_anatomy_tickness

# check values and reset index based on the subset of ASD or TC, Just needs to be change the group data and the condition

# data_train_with_labels all data for training
data_train_with_labels,data_train_anatomy_area, data_train_anatomy_tickness=data_fuctions(data_train,labels_train)

# data_train_asd only asd subjects
data_train_asd=data_train_with_labels.loc[(data_train_with_labels.Group=="ASD")]
columns=data_train_asd.columns
data_train_asd = data_train_asd.reindex(columns=columns)
labels_train_group=list(data_train_asd.Group.values)


# In[5]:


# print(len(labels_train_group))
data_train_asd.shape


# In[34]:


def _load_fmri(fmri_filenames):
    """Load time-series extracted from the fMRI using a specific atlas."""
    return np.array([pd.read_csv(subject_filename,
                                 header=None).values
                     for subject_filename in fmri_filenames])



class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):    

        
  
        # make a transformer which will load the time series and compute the
        # connectome matrix (tanget type for this example)
        
        self.transformer_fmri =make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind= "tangent", vectorize=False)) 
        
        
    def fit(self, X_df, y, atlas):
        # get only the time series for the the given atlas (msdl in the given example)
        fmri_filenames = X_df[atlas]
        self.transformer_fmri.fit(fmri_filenames, y)
        return self

    def transform(self, X_df, atlas):
        fmri_filenames = X_df[atlas]
        return self.transformer_fmri.transform(fmri_filenames)


# In[35]:


# Information from atlas based on the code above
feature_extractor_tangent= FeatureExtractor()

def data_extraction_matrix(feature_extractor_matrix,data_train_asd,labels_train_group):
'''
This function produces atlas-matrix dictionaries by groups
'''
    decimal.getcontext().prec = 10
    atlas=['fmri_msdl','fmri_harvard_oxford_cort_prob_2mm','fmri_basc064',
           'fmri_basc122', 'fmri_basc197','fmri_power_2011', 'fmri_craddock_scorr_mean']

    atlas_matrix={}
    for i in atlas:
        print (i)
        feature_extractor_train_matrix=feature_extractor_matrix.fit(data_train_asd, labels_train, i)
        transform_training_matrix=feature_extractor_train_matrix.transform(data_train_asd, i)
        atlas_matrix[i]=transform_training_matrix
#         print(transform_training_tangent.shape)
    return atlas_matrix

tangent =data_extraction_matrix(feature_extractor_tangent,data_train_asd,labels_train_group)


# Function to create a regressor to visualize the data 

import decimal
def regress_values(matrix, atlas:str):
  
    regressor_atlas_matrix={}
    castor=np.array([list(v) for v in correlation.values()])    
    for key in matrix:
        mean_regressor=np.mean(matrix[atlas], axis=0)
        std_regressor= np.std(matrix[atlas], axis=0)
        zscore_regressor=scipy.stats.zscore(matrix[atlas], axis=0)
        regressor_atlas_matrix[key]=mean_regressor
        return  mean_regressor
abc0=regress_values(tangent,"fmri_msdl")



# For visualization purposes the mean was selected as regressor
mean_tangent_matrix= np.mean(transform_training_tangent, axis=0)
std_tangent_matrix= np.std(transform_training_tangent, axis=0)
zscore_tangent_matrix=scipy.stats.zscore(transform_training_tangent)
# zscore_tangent_matrix=((transform_training_tangent)-mean_tangent_matrix)/std_tangent_matrix

# check the shape for MSDL this must be (39,39) ROIs
zscore_tangent_matrix.shape


# In[62]:


# Figure specifications
# Plotting connectivity matrix

def plot_connectivity_matrix(matrix_data,matrix_name:str):
'''
The connectivity_matrix nilearn implementation allows the visualization of the matrices based on the atlas and matrix selected.

'''
    # Plot the tangent matrix
    # The labels of the MSDL Atlas that we are using 
    # Data from the atlas used (in the given example MSDL)
    atlas = datasets.fetch_atlas_msdl()
    # Loading atlas data stored in 'labels'
    labels = atlas['labels']
    # Loading atlas coordinates
    coords = atlas.region_coords
    
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 12}
    
    tt = plt.figure(1,figsize=(7,6))

 
    np.fill_diagonal(matrix_data, 0)
    plt.imshow(matrix_data, interpolation='None', cmap='RdYlBu_r', vmax=.000002, vmin=-.000002)
    plt.yticks(range(len(atlas.labels)),labels, fontsize=10, weight='bold');
    plt.xticks( range(len(atlas.labels)), labels, rotation=90, fontsize=10, weight='bold');
    plt.title(str(matrix_name)+'_msdl',fontdict=font)
    plt.colorbar(shrink=0.8)
    
    tt2 = plt.figure(2)   
    view=plotting.view_connectome(matrix_data,coords, node_size=5.0, edge_threshold='99.5%')    
    

    plt.show()    
   
    return view

# Retrieving the visualizations and saving the plots 
for i in tangent.keys():
    
    if i=="fmri_msdl":
  
        
        tt = plt.figure(1,figsize=(7,6))
        plot_connectivity_matrix(np.mean(tangent[i], axis=0), "Tangent")
        
        tt2 = plt.figure(1,figsize=(7,6))
        plot_connectivity_matrix(np.mean(correlation[i], axis=0), "Correlation")
      
        
        
        pp = PdfPages("msdl.pdf")
        pp.savefig(tt, dpi = 300, transparent = True)
        pp.savefig(tt2, dpi = 300, transparent = True)
        pp.close()
        


# In[61]:



def viz_connectome (regressor,imagename:str):
'''
The plotting.plot_connectome nilearn tool allows to visualize the nodes and connections among regions of the brain 
based on the selected regressor (for this project we use the mean). This value will define the strength of the connectivity
'''


    # List of colors 
    colors_df=pd.read_excel(open('colors_8.xlsx','rb'))
    colors_df['lower_color'] = map(lambda x: x.lower(), colors_df['color'])

    # Plot the tangent matrix
    # The labels of the MSDL Atlas that we are using 
    # Data from the atlas used (in the given example MSDL)
    atlas = datasets.fetch_atlas_msdl()
    # Loading atlas data stored in 'labels'
    labels = atlas['labels']
    # Loading atlas coordinates
    coords = atlas.region_coords

    # Plot of the connectome based on the nilearn plotting.plot_connectome package
    fig = plt.figure(figsize=(6,7))
    display=plotting.plot_connectome(regressor,coords,node_size =40,
                             edge_threshold="99.5%", display_mode="ortho",  title="Tangent-ASD", alpha=1,
                                     colorbar=True, annotate=False)

    values =list(colors_df["color"])
    keys = labels
    colors_labels = dict(zip(keys, values))

    patchList = []
    fontP = FontProperties()
    fontP.set_size('small')

    for key in colors_labels:

            data_key = mpatches.Patch(color=colors_labels[key], label=key)
            patchList.append(data_key)
   

    plt.legend(handles=patchList,prop=fontP, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=8, fancybox=True, shadow=True)

    #depends on the data to viz
    plt.savefig(imagename+".png", dpi=300)

    plt.close(fig)
    plt.show()
viz_connectome (abc,"asd")


# In[ ]:




