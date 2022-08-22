#!/usr/bin/env python
# coding: utf-8

# ## Xarray example Jupyter Notebook
# 
# Runs in _jupyter lab_ notebook.
# 

# ## Xarray instalation
# Install Xarray and some of its dependencies if not already installed.
# 
# ``` conda install -c conda-forge xarray==0.20.2 dask netCDF4 bottleneck pooch```
# 
# It may take a while resolving installation environments.
# If it is successful, will install other package dependecies.
# 
# Xarray comes with a collection of datasets to explore: [xarray.tutorial.open_dataset](https://docs.xarray.dev/en/stable/generated/xarray.tutorial.open_dataset.html)
# 
# Available datasets:
# 
# `"air_temperature"`: NCEP reanalysis subset
# 
# `"air_temperature_gradient"`: NCEP reanalysis subset with approximate x,y gradients
# 
# `"basin_mask"`: Dataset with ocean basins marked using integers
# 
# `"ASE_ice_velocity"`: MEaSUREs InSAR-Based Ice Velocity of the Amundsen Sea Embayment, Antarctica, Version 1
# 
# `"rasm"`: Output of the Regional Arctic System Model (RASM)
# 
# `"ROMS_example"`: Regional Ocean Model System (ROMS) output
# 
# `"tiny"`: small synthetic dataset with a 1D data variable
# 
# `"era5-2mt-2019-03-uk.grib"`: ERA5 temperature data over the UK
# 
# `"eraint_uvz"`: data from ERA-Interim reanalysis, monthly averages of upper level data
# 
# `"ersstv5"`: NOAA’s Extended Reconstructed Sea Surface Temperature monthly averages
# 

# In[1]:


# Load required libraries

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
import pooch
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the air_temperature dataset and define a xarray datastructure
# 4 x Daily Air temperature in degrees K at sigma level 995 
# (2013-01-01 to 2014-12-31)
# Spatial Coverage
# 2.5 degree x 2.5 degree global grids (144x73) [2.5 degree = 172.5 miles]
# 0.0E to 357.5E, 90.0N to 90.0S

ds = xr.tutorial.open_dataset('air_temperature')
#ds.info()


# In[3]:


# Show the components of the xarray.Dataset
ds


# In[4]:


# To see the values of  data variable: air
ds.air


# In[5]:


# Direct access to values
ds.air.values


# In[6]:


# Underneath is a Numpy N dimensional Array
type(ds.air.values)


# In[7]:


# Show dimensions
ds.dims


# In[8]:


# Show coordinates
ds.coords


# In[9]:


# Show attributes
ds.attrs


# We can use [xarray.DataArray.groupby](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.groupby.html) to caculate average monthly temperatures and anomalies.

# In[10]:



# calculate monthly climatology
climatology = ds.groupby('time.month').mean('time')

# calculate anomalies 
anomalies = ds.groupby('time.month') - climatology


# In[11]:


climatology


# In[12]:


anomalies


# We can make plots directly with the DataArray

# In[13]:


ds.air.isel(time=1).plot(x="lon");


# In[14]:


# We can take time average of air temperature over all coordinates 
ds.air.mean("time")


# In[ ]:





# 

# In[15]:


df = ds.to_dataframe()
df.head()


# In[16]:


df.tail()


# In[17]:


df.describe()


# ### Positional indexing
# 

# In[18]:


da = xr.DataArray(
     np.random.rand(4, 3),
     [
         ("time", pd.date_range("2000-01-01", periods=4)),
         ("space", ["IA", "IL", "IN"]),
     ],
   )


# In[19]:


da


# In[20]:


da[:2]


# In[21]:



da[0,0]


# In[22]:


da[:, [2, 1]]


# ### Indexing with dimension names

# In[23]:


# index by integer array indices
da.isel(space=0, time=slice(None, 2))


# In[24]:


# index by dimension coordinate labels
da.sel(time=slice("2000-01-01", "2000-01-02"))


# In[25]:


# Same as: da[0,0]
da.isel(space=[0], time=[0])


# In[26]:


da.sel(time="2000-01-01")


# In[ ]:




