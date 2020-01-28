# ASD_Thesis_Project


Autism Spectrum Disorder (ASD) is a developmental condition of early onset,
composed by disorders of diverse etiology but with overlapping diagnosis criteria.
Traditional diagnosis has as limitation to not accurately asses patients in early
childhood, which is critical to successfully treat the condition.
This project has as goal to explore the fMRI and MRI data pre-processed by the IMaging-PsychiAtry Challenge
organizers to classify ASD patients using Machine Learning models.

As suggested by the IMaging-PsychiAtry Challenge organizers, the retrieval of the 
pre-processed data requires Python and the following dependencies:

* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `matplolib`
* `seaborn`
* `nilearn`
* `jupyter`
* `ramp-workflow`

The project was developed using jupyter notebooks. However, the final reported code is presented as .py files due to convenience
to double check the content. Therefore, it will be advisable to check the code using notebooks, understanding that 
`nilearn` and `ramp-workflow` are not included by default in the Anaconda out of the box installation.

An easy way to get the requirements set up is to execute the jupyter notebook, from the root directory using:
```
jupyter notebook autism_starting_kit.ipynb
``` 
The pre-processed data can be reached under the autism_starting_kit. However, the Atlases data used to collect the time-series information
is not available anymore.
* `The starting kit can be found at https://github.com/ramp-kits/autism ` the autism-master folder contains all the 
data files (except the Atlases) by cloning or downloading this zip the data, img, preprocesing, submissions folders are available locally.
Challenge Organization details can be found at https://paris-saclay-cds.github.io/autism_challenge/
