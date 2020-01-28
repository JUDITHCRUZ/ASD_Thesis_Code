'''
Functions used locally to parse and concatenate the results.
Some of the exploratory analysis of the work were performed simultaneously using the same scripts but 
changing the parameters appropriately. 

Note: locations make reference to the local directories where the results were saved (the are not meant to be run).
'''

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 

### 
#Function used to reformat the data to visualize the classification resutls

location=r"C:\Users\Ju\Documents\classifier_results\Changes_thesis\Results_after_feature_selection\tangent\train\fmri_basc122_logreg_elnet.csv"
def parsing_data(location:str,atlas:str,matrix,fms:str):
    data=pd.read_csv(location)
#     print(data.columns)#     
    data = data.rename(columns={'Unnamed: 0': 'CV'})
#     del data['Unnamed: 0']
#     print(data.head())
    data.set_index('CV', inplace=True)
#     print(data.head())
    data['Atlas']=atlas
    data["Matrix"]=matrix
    data['FSM']=fms
    print(data.head())
    data.to_csv(location)    
    
    
    


# In[29]:


####
#Function used to reformat the data to visualize the classification results


import pandas as pd 
location=r"C:\Users\Ju\Documents\classifier_results\Changes_thesis\Results_after_feature_selection\tangent\all_matrix\all_basc064\all_msdl\fmri_msdl_logreg_elnettangent.csv"
atlas="MSDL"
fms="elnet"
matrix="tangent"
data1="fMRI"
model1="logreg"


data=pd.read_csv(location)
data = data.rename(columns={'Unnamed: 0': 'CV'})
data.set_index('CV', inplace=True)
data['Atlas']=atlas
data["Matrix"]=matrix
data['FSM']=fms
data['Data']=data1
data['Classifier']=model1

#


# In[30]:


data


# In[31]:


data.to_csv(location)


# In[32]:


from glob import iglob
from os.path import join
import pandas as pd


####
# code to read multiple csv files and concatenate the classification information to be visualized

path=r"C:\Users\Ju\Documents\classifier_results\Changes_thesis\Results_after_feature_selection\tangent\all_matrix\all_basc064\all_msdl"
def read_df_rec(path, fn_regex=r'*.csv'):
    return pd.concat((pd.read_csv(f,index_col="CV") for f in iglob(
        join(path, '**', fn_regex), recursive=True)), ignore_index=False,axis=0,sort=False)
har_oxf_atlas_matrix=read_df_rec(path)


# In[33]:


har_oxf_atlas_matrix


# In[34]:


#####
# create a file with all the classification results
har_oxf_atlas_matrix.to_csv(r"C:\Users\Ju\Documents\classifier_results\Changes_thesis\Results_after_feature_selection\tangent\all_matrix\all_basc064\all_msdl\har_oxf_atlas_matrix_df.csv")


# In[ ]:




