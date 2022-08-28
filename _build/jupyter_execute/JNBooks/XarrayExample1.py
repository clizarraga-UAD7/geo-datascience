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

plt.rcParams['figure.figsize'] = (8,5)


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


# Show the values of  data variable: air
ds.air


# In[5]:


# Direct access to values
#ds.air.values


# In[6]:


# Underneath is a Numpy N dimensional Array
type(ds.air.values)


# ### DataArray Properties

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


# Show the resulting Dataset
climatology


# In[12]:


# Show the sesulting Dataset
anomalies


# 
# Much like in Pandas, Xarray includes an interface to Matplotlib that we can access through the `.plot()` method of every DataArray.
# 
# Plotting the air temperature for the first time value, using longitude as the x variable.

# In[13]:


ds.air.isel(time=1).plot(x="lon");


# In[14]:


# We can take time average of air temperature over all coordinates 
ds.air.mean("time").plot(x="lon");


# ## Indexing and selecting data

# ### Positional indexing
# 

# In[15]:


# Create the following Dataset 

da = xr.DataArray(
     np.random.rand(4, 3),
     [
         ("time", pd.date_range("2000-01-01", periods=4)),
         ("space", ["IA", "IL", "IN"]),
     ],
   )
# Show the Dataset
da


# In[16]:


# Show Dataset dimensions
da.dims


# In[17]:


# Select the first 2 values of first variable (time)
da[:2]


# In[18]:


# Select the first values of (time, space)
da[0,0]


# In[19]:


# get all the values of the `time` variable and 
# select the third and second values of the `space` variable in that order.
da[:, [2, 1]]


# ### Indexing with dimension names

# In[20]:


# index by integer array indices
# Select by index the first space variable and first 2 values of time
da.isel(space=0, time=slice(None, 2))


# In[21]:


# index by dimension coordinate labels
da.sel(time=slice("2000-01-01", "2000-01-02"))


# In[22]:


# Same as: da[0,0], but using indexes
da.isel(space=[0], time=[0])


# In[23]:


# Select a specific time slice
da.sel(time="2000-01-01")


# ### Droping labels and dimensions
# The `drop_sel()` method returns a new object with the listed index labels along a dimension dropped:

# In[24]:


da


# In[25]:


# Drop 2 space coordinates using labels
da.drop_sel(space=["IN", "IL"])


# In[26]:


# Drop by index the first values of space and time variables
da.drop_isel(space=[0], time=[0])


# Use drop_vars() to drop a full variable from a Dataset. Any variables depending on it are also dropped:

# In[27]:


# Drop the time coordinate variable reference
da.drop_vars("time")


# ## Example of reading a netCDF file using Xarray
# 
# We will read an [Argo](https://argo.ucsd.edu/about/) data file that describes the temperature and salinity of the water and some of the floats measure other properties that describe the biology/chemistry of the ocean. 
# 
# The Argo robot instruments drift along the ocean and collect data which are stored in netCDF format and can be [acccessed via HTTP and FTP](https://argo.ucsd.edu/data/data-from-gdacs/).
# 
# Fot this example, we download the file: [nodc_4901112_prof.nc](https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/meds/4901112/)

# In[28]:


import numpy as np
import xarray as xr

from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (8,5)


# In[29]:


# Reading an Argo dataset using Xarray
argo_data = xr.open_dataset('data/nodc_4901112_prof.nc')

# Show file keys
argo_data.keys()


# In[30]:


# Show dataset structure
argo_data


# In[31]:


# Show dataset variables
argo_data.dims


# In[32]:


# Disctionary of attributes
argo_data.attrs


# In[33]:


# Show first n=5 rows of temperature values
argo_data.temp_adjusted.head()


# In[34]:


# Quick Timeseries Profile plot of Temperature
argo_data.temp_adjusted.T.plot()
plt.gca().invert_yaxis()


# In[35]:


# Quick Timeseries Profile plot of Salinity
argo_data.psal_adjusted.T.plot()
plt.gca().invert_yaxis()


# In[36]:


# Profile Plot
nprof = 25 #Specify a profile to plot
plt.plot(argo_data.temp_adjusted[nprof], argo_data.pres_adjusted[nprof])

plt.xlabel('Temperature (C)')
plt.ylabel('Pressure (dbar)')
plt.title('Argo Profile from %s' % argo_data.juld[nprof].dt.strftime('%a, %b %d %H:%M').values)
plt.grid()

plt.gca().invert_yaxis() #Flip the y-axis


# In[37]:


# Profile Plot

data = argo_data.copy()

# Subplot example
fig, (ax1,ax2) = plt.subplots(1,2, sharey=True, figsize=(10,6))

nprof = 0 # Fist profile
ax1.plot(data.temp_adjusted[nprof], data.pres_adjusted[nprof], label=data.juld[nprof].dt.strftime('%Y-%m-%d').values)
ax2.plot(data.psal_adjusted[nprof], data.pres_adjusted[nprof])

nprof = 25 # Middle-ish profile
ax1.plot(data.temp_adjusted[nprof], data.pres_adjusted[nprof], label=data.juld[nprof].dt.strftime('%Y-%m-%d').values)
ax2.plot(data.psal_adjusted[nprof], data.pres_adjusted[nprof])

nprof = -1 # Last profile
ax1.plot(data.temp_adjusted[nprof], data.pres_adjusted[nprof], label=data.juld[nprof].dt.strftime('%Y-%m-%d').values)
ax2.plot(data.psal_adjusted[nprof], data.pres_adjusted[nprof])

ax1.set_ylabel('Pressure (dbar)')
ax1.set_xlabel('Temperature (C)')
ax2.set_xlabel('Salinity')
ax1.invert_yaxis()
ax1.legend()

# Add some gridlines
ax1.grid()
ax2.grid()

# Add a super title
fig.suptitle('Argo Float #%d' % data.platform_number[nprof].values, fontweight='bold', fontsize=16);


# ## Temperature Salinity Diagram

# In[38]:


# TS Diagram
nprof = 25 #Selected profile
plt.scatter(data.psal_adjusted[nprof], data.temp_adjusted[nprof])
plt.xlabel('Salinity')
plt.ylabel('Temperature (°C)')
plt.grid()

plt.title('Argo Float #%d' % data.platform_number[nprof].values, fontweight='bold');


# In[39]:


# We can also use a colored scatterplot to show the depth dimension.
# T-S Diagram with depth
plt.figure(figsize=(8,6))

nprof = 25 #Selected profile
plt.scatter(data.psal_adjusted[nprof], data.temp_adjusted[nprof], c=data.pres_adjusted[nprof], cmap='viridis_r')
plt.xlabel('Salinity');
plt.ylabel('Temperature (°C)')
plt.grid()

cbh = plt.colorbar();
cbh.set_label('Pressure (dbar)')


# To calculate density, we will need the wonderful `seawater` library.
# 
# To install it:
# 
# Use: `!pip install seawater`
# 
# Or in Anaconda use: `conda install -c conda-forge seawater`

# In[40]:


import seawater


# In[41]:


# TS Diagram with density contours
plt.figure(figsize=(8,6))

# Calculate the density lines
x = np.arange(33, 35, .1)
y = np.arange(2, 23, .5)
X, Y = np.meshgrid(x, y)
Z = seawater.eos80.dens0(X,Y) - 1000 # Substract 1000 to convert to sigma-t

# Plot the contour lines
CS = plt.contour(X, Y, Z, colors='grey', linestyles='dashed', levels=np.arange(22,30,.5))
plt.clabel(CS, inline=1, fontsize=10, fmt='%0.1f')

# Plot the data
nprof = 25 #Selected profile
plt.scatter(data.psal_adjusted[nprof], data.temp_adjusted[nprof], c=data.pres_adjusted[nprof], cmap='viridis_r')
plt.xlabel('Salinity');
plt.ylabel('Temperature (°C)')
plt.title('Argo Float #%d on %s' % (data.platform_number[nprof].values, data.juld[nprof].dt.strftime('%Y-%m-%d').values), fontweight='bold');

# Add a colorbar
cbh = plt.colorbar(label='Pressure (dbar)');


# ## Float Track Map

# In[42]:


# Simple map of a float track
plt.figure(figsize=(8,8))
plt.plot(data.longitude, data.latitude, c='lightgrey')
plt.scatter(data.longitude, data.latitude, c=data.juld, cmap='RdYlBu')
plt.grid()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('ARGO Robot drift map')

# Crude profile labels
for jj in [1,25,-1]:
  plt.text(data.longitude[jj]+.02, data.latitude[jj]+.02, data.n_prof[jj].values);

# Add a colorbar
cbar = plt.colorbar();

# Fix the colorbar ticks
import pandas as pd # We need pandas for this
cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%Y-%m-%d'));

# Set the aspect ratio to pseudo-Mercator
#plt.gca().set_aspect(1 / np.cos(np.deg2rad( np.mean(plt.ylim()) )))


# In[ ]:




