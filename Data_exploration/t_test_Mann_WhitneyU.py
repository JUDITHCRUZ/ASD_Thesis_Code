''' 
## T-test
Like every test, this inferential statistic test has assumptions. The assumptions that the data must meet in order for the test results to be valid are:

* The samples are independently and randomly drawn
* The distribution of the residuals between the two groups should follow the normal distribution
* The variances between the two groups are equal
* The dependent variable (outcome being measured) should be continuous which is measured on an interval or ratio scale.
    * If any of these assumptions are violated then another test should be used. 
'''
import seaborn as sns
import matplotlib.pylab as plt
plt.style.use('ggplot')
import pandas as pd
from scipy import stats
import numpy as np
pd.set_option('display.notebook_repr_html', True)
import os, sys
import warnings
warnings.filterwarnings('ignore')
import researchpy as rp
import numpy as np
from copy import deepcopy
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt



from problem import get_train_data
data_train, labels_train = get_train_data()

### Add labels to the data for visualization purposes 
data_train['Group'] = labels_train 
data_train['Group'].replace({ 0: 'TC',1: 'ASD'}, inplace= True)


#Cast all numeric values to  float
def cast_int_val(df):
    
    df['Group'] = data_train['Group'].astype(str)
    df['participants_site'] = data_train['participants_site'].astype(str)
    df['fmri_select'] = data_train['fmri_select'].astype(str)
    df['anatomy_select'] = data_train['anatomy_select'].astype(str)
    df['repetition_time'] = data_train['repetition_time'].astype(str)
    
    int_col = df.select_dtypes(include = ['int64']) 
    for col in int_col.columns.values:
        df[col] = data_train[col].astype('float64')
        
        
    
    return(df)
data_train_all_float= cast_int_val(data_train)


#Create table for missing data analysis
 
def missing_data_check(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
# print(missing_data_check(data_train))

### No missing values


###### Checking for Coliniarities 
sns.set(style="white")

corr =data_train_all_float.loc[:,data_train_all_float.dtypes == 'float64'].corr('spearman')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heat-map with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


## Grouping by TC and ASD
ASD = data_train[(data_train['Group'] == 'ASD')]
ASD.reset_index(inplace= True)

Control = data_train[(data_train['Group'] == 'TC')]
Control.reset_index(inplace= True)

## Normality Test

def normality_values(df):

	"""
	Perform the Shapiro-Wilk test for normality using stats.shapiro package
	Documentation of the method  can be found at https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html

	The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.

	Parameters
	df: {dataframe}shape (n_samples, n_features)
        input data
	

	Returns
	Non-normal: {dictionary} where 
		keys: feature name
		values: p-value result
	Normal:{dictionary} where 
		keys: feature name
		values: p-value result
	
	"""
    normality_test_negative=dict()
    normality_test_positive=dict()
    
    for i in group:
        if group[i].dtype =="float64":
            normality_test= stats.shapiro(df[i].dropna())
                        
            if normality_test[1]>=0.05:                
                normality_test_positive[i]= normality_test[1]
            else:
                normality_test_negative[i]= normality_test[1]
                
               
    return(dict(sorted(normality_test_negative.items(), key=lambda x: x[1])),dict(sorted(normality_test_positive.items(), key=lambda x: x[1])))
Non-normal,Normal=normality_values(data_train_all_float)




def covert_df(dictionary):
	"""
	Auxiliary function to convert dictionary in data-frame
	"""
    df_group= pd.DataFrame(list(zip(dictionary.keys(),dictionary.values())),
    columns =['Feature', 'p-value']) 
    return df_group



## Homogeneity of variances
def homogeneity_variance(df,df_ASD,df_TC):
	'''
	This function computes the Levene test tests on the samples
		
		Input
		----------
		df: {data-frame}, shape (n_samples, n_features)
			input data
		df_ASD: {data-frame} shape (n_samples, n_features)
			subset of features for ASD label
		df_TC: {data-frame} shape (n_samples, n_features)
			subset of features for Control group (TC) label
	   Output

	** The  the null hypothesis that all input samples are from populations with equal variances.
	 Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations from normality.

	As p > 0.05 the data sets conform to the variance criterion
	'''
    homogeneity_of_variance = dict()
    stat_positive={}
    
    for i in df.loc[:,df.dtypes == 'float64']:
        statistic, p_value = stats.levene(df_ASD[i], df_TC[i])
        homogeneity_of_variance[i]=  p_value
        if p_value>=0.05:
            stat_positive[i]=p_value
    return homogeneity_of_variance, stat_positive



### Visual of the homogeneity_of_variance
ncols= 4
nrows= int((len(homo_positive['Feature'])+1)/ncols)


fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=False, constrained_layout=True)

# width and height
fig.set_size_inches(40,100)
fig.subplots_adjust(0.3)
fig.subplots_adjust(hspace = 1, wspace= .09)

axes_list = [item for sublist in axes for item in sublist]

axs = axes.ravel()

for i in homo_positive['Feature']:
    try:        
        ax = axes_list.pop(0)
        diff = Control[i]-ASD[i]
        ax.hist(diff, histtype ='bar')
        ax.set_title(i)
        
        
    except IndexError as e:
        pass
		
		
		
#### T-test calculation 
import statsmodels.stats.multitest as smm
def t_test(df,df_ASD, df_TC):
	    """
		This function computes the Mann-Whitney rank test on ASD-TC samples
		based on the normal variables calculated with Shapiro Wilk Test.   

		Input
		----------
		df: {dataframe}, shape (n_samples, n_features)
			input data
		df_ASD: {dataframe} shape (n_samples, n_features)
			subset of features for ASD label
		df_TC: {dataframe} shape (n_samples, n_features)
			subset of features for Control group (TC) label
	   Output
		----------
		features_dict_p_val_05: {dictionary}
			features with significant p values based on t-test
			Documentation for the method used can be found at https://researchpy.readthedocs.io/en/latest/ttest_documentation.html
		features corrected based on Bonferroni
			Documentation can be found at https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
		reject_t: true for hypothesis that can be rejected for given alpha
		pvals_corrected_t: p-values corrected for multiple tests
		corrected_hypotesis: corrected alpha for Bonferroni method
		rejected_t_dic: dictionary filtered by reject_t values equals True
		   
    """

   features_pv_05=dict()
    print(type(features_pv_05))
    exceptions=dict()

    for i in data_train_anatomy._get_numeric_data():
        try:  
            descriptives, results = rp.ttest(df_ASD[i], df_TC[i])     

            p_val=(results["results"][3])                
            if (p_val<=0.05) and (p_val!=0.0):   
                features_pv_05[i]=p_val

        except ZeroDivisionError:
            exceptions[i]=p_val
            
    features_dict_p_val_05=(dict(sorted(features_pv_05.items(), key = lambda x : x[1])))
    ## Correction Bonferroni 
    reject_t, pvals_corrected_t,alphacSidak_t, alphacBonf_t = smm.multipletests((list(features_dict_p_val_05.values())),alpha=0.05, method='b',returnsorted=True)
    corrected_hypotesis= dict((key, value) for (key, value) in zip(features_pv_05.keys(), reject_t))
    rejected_t_dic = {k: v for k, v in corrected_hypotesis.items() if v==True}
	corrected_pvalues=dict((key, value) for (key, value) in zip(rejected_t_dic.keys(), pvals_corrected_t))
    corrected_p_dic = {k: v for k, v in corrected_pvalues.items() }
    t_val_dict=dict((key, value) for (key, value) in zip(rejected_t_dic.keys(),stats_features.values()))
    
    
    return  features_dict_p_val_05, reject_t, pvals_corrected_t, corrected_hypotesis,rejected_t_dic

p_v_significant = t_test(data_train_anatomy,ASD,Control)

#




def mannwhitneyU(dict_non_normal,df_ASD,df_TC):
    """
		This function computes the Mann-Whitney rank test on ASD-TC samples
		based on the non-normal variables calculated with Shapiro Wilk Test.   

		Input
		----------
		dict_non_normal: {dictionary}, keys: feature_names, values: p-value
			input data
		df_ASD: {dataframe} shape (n_samples, n_features)
			subset of features for ASD label
		df_TC: {dataframe} shape (n_samples, n_features)
			subset of features for Control group (TC) label
	   Output
		----------
		 features_pv_05: {dictionary}
			features with significant p values based on mann_whitneyU
		reject: true for hypothesis that can be rejected for given alpha
		pvals_corrected: p-values corrected for multiple tests
		alphacBonf: corrected alpha for Bonferroni method
		rejected_dic: dictionary filtered by reject_t values equals True
       
    """


    mann_whitneyU_p=dict()
    stats_dict=dict()

    for i in list(dict_non_normal.keys()):
        try:  
            stat, p = mannwhitneyu( df_ASD[i],df_TC[i]) 

            alpha= 0.05

            if p<=alpha:
                
                mann_whitneyU_p[i]=p

        except ZeroDivisionError:    
            exceptions[i]=p
    features_pv_05=(dict(sorted( mann_whitneyU_p.items(), key = lambda x : x[1])))
            
    reject_t, pvals_corrected_t,alphacSidak_t, alphacBonf_t =  smm.multipletests((list(features_pv_05.values())),alpha=0.05, method='b', is_sorted=True,returnsorted=True)        
    corrected_values= dict((key, value) for (key, value) in zip(features_pv_05.keys(), reject_t))    
    rejected_t_dic = {k: v for k, v in corrected_values.items() if v==True}
    reject_pvalues=dict((key, value) for (key, value) in zip( rejected_t_dic.keys(), pvals_corrected_t))
    return features_pv_05, reject,  pvals_corrected,  alphacBonf, rejected_dic,reject_pvalues

# t-test feature visualization
def viz_features(df, rejected_features:dict, columns:list):
     """
		This function visualizes the features that presented significant p values after t-test analysis
		and Bonferroni correction given the total data-set(data_train),
		rejected features (dictionary with the features rejected),
		and the list of columns that will not be included. 
		

		Input
		----------
		df: {dataframe}, shape (n_samples, n_features)
			input data
		rejected_features: {dictionary}
			input features with significant difference ASD-TC
	   Output
    ----------
		Visualization: {plt}
       
    """

    papapap=data_train[list(rejected_features.keys())].rename(columns = lambda x: x.replace('anatomy_', ''))
    # papapap.head()
    papapap1=df[["Group"]]
    papapap.drop(papapap.columns[columns], axis=1, inplace=True)
    pdList = [papapap1,papapap]  # List of your data-frames
    df_with_group = pd.concat(pdList,axis=1)
    flatui = ["#e74c3c", "#34495e"]
    sns.set_context("notebook", font_scale=2.0, rc={"lines.linewidth": 2.5})
    sns.set_palette(flatui)
    sns_plot = sns.pairplot(df_with_group, hue='Group', kind ='reg',size=5.5,height=1.0)
    plt.xticks(rotation=30)
    plt.show()




def viz_feature_pval(df,dictionary):
    cols_pv_001=list(dictionary.keys())
    df[cols_pv_001].hist(bins=20, figsize=(15,10))
    plt.show()
viz_feature_pval(data_train,p_v_significant) 