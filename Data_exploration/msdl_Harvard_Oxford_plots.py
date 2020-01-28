#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#Adapted from http://nilearn.github.io/auto_examples/01_plotting/plot_prob_atlas.html


get_ipython().run_line_magic('matplotlib', 'inline')


# 
# Visualizing 4D probabilistic atlas maps
# =======================================
# 
# This example shows how to visualize probabilistic atlases made of 4D images.
# There are 3 different display types:
# 
# 1. "contours", which means maps or ROIs are shown as contours delineated by     colored lines.
# 
# 2. "filled_contours", maps are shown as contours same as above but with     fillings inside the contours.
# 
# 3. "continuous", maps are shown as just color overlays.
# 
# A colorbar can optionally be added.
# 
# The :func:`nilearn.plotting.plot_prob_atlas` function displays each map
# with each different color which are picked randomly from the colormap
# which is already defined.
# 
# 
# 
# 

# In[11]:


# Load 4D probabilistic atlases
from nilearn import datasets

# Harvard Oxford Atlasf
harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')


# Multi Subject Dictionary Learning Atlas
msdl = datasets.fetch_atlas_msdl()

# Craddock
craddock = datasets.fetch_atlas_craddock_2012
# Power
power_2011=datasets.fetch_coords_power_2011()
# Visualization
from nilearn import plotting

atlas_types = {'Harvard_Oxford': harvard_oxford.maps,               
               'MSDL': msdl.maps, 
               
               }

for name, atlas in sorted(atlas_types.items()):
    plotting.plot_prob_atlas(atlas, title=name, colorbar=True)


print('ready')
plotting.show()


# In[ ]:




