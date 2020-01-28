'''
Stratification based on age and  Multivariate Regression Analysis based
on demographic  variables.

'''

import os
from sklearn.preprocessing import LabelEncoder 
from seaborn import countplot
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy


def data_fuctions(df):
    data_train_with_labels=df
    # make the variables numbers
    data_train_with_labels ['participants_sex'] = LabelEncoder().fit_transform(data_train['participants_sex'])
    ## remove the fmri data
    # data_train.dtypes==object
    # fmri_basc064, fmri_basc122, fmri_basc197, fmri_craddock_scorr_mean, fmri_harvard_oxford_cort_prob_2mm            
    # fmri_motions, fmri_msdl, fmri_power_2011 
    data_train_with_labels['condition'] = labels_train
    data_train_with_labels['Group'] = labels_train
    data_train_with_labels['Group'].replace({ 0: 'TC',1: 'ASD'}, inplace= True)
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data_train_with_labels.select_dtypes(include=numerics)    
    data_train_with_labels["participants_age"]=data_train_with_labels["participants_age"].round(0).astype(int)
    
    data_train_anatomy_area = data_train_with_labels[[col for col in data_train.columns if  col.endswith('site') or col.endswith('area') or col.startswith('Group')]]
    data_train_anatomy_tickness = data_train_with_labels[[col for col in data_train.columns if col.endswith('thickness') or col.startswith('Group')]]

   
   
    return data_train_with_labels,data_train_anatomy_area,data_train_anatomy_tickness

def grouping_by_age(df, age_a, age_b):    
    sub_set_data= df[["participants_age", "condition", "Group", "participants_site", "participants_sex"]].dropna()
    effect_grouping =  sub_set_data[(sub_set_data.participants_age >= age_a) & (sub_set_data.participants_age <= age_b)]
    return effect_grouping
	
	
def viz_count_plots_age(df, file_name):
        
    fig=figure(figsize=(22.20,6.80))
    countplot(data=df, x="participants_age", hue="condition", palette="BuPu")

    #change labels size, fonts and styles
    plt.ylabel('Subject Count',  fontsize=14,  fontweight='bold')
    plt.xlabel('Subject Age',  fontsize=14, fontweight='bold')
    plt.legend( ('TC', 'ASD'), loc="upper right")
    plt.xticks(fontname = "Times New Roman", fontsize=16)
    plt.yticks(fontname = "Times New Roman", fontsize=16)
    plt.show()
    # Saving in high resolution
    fig.savefig(file_name,dpi = 300)
	
def regression_model_age(df):
    reg_object= smf.logit(formula="condition ~ participants_sex", data=df).fit()
    print("Odd Ratios of the Effect", "----------") 
    # Odd Ratios confidential intervals to compare to the population values
    params=reg_object.params
    conf=reg_object.conf_int()
    conf['OR']=params
    conf.columns =["Lower CI", "Upper CI", "OR"]
    
    print(numpy.exp(reg_object.params), "\n", numpy.exp(conf),"\n","\n", 
          reg_object.summary())
		  
		  
def regression_model_age_sex(df):
    reg_object= smf.logit(formula="condition ~ participants_age + participants_sex", data=df).fit()
    print("Odd Ratios of the Effect", "----------") 
    # Odd Ratios confidential intervals to compare to the population values
    params=reg_object.params
    conf=reg_object.conf_int()
    conf['OR']=params
    conf.columns =["Lower CI", "Upper CI", "OR"]
    
    print(numpy.exp(reg_object.params), "\n", numpy.exp(conf),"\n","\n", 
          reg_object.summary())