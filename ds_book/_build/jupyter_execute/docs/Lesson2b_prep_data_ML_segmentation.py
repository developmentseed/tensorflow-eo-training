#!/usr/bin/env python
# coding: utf-8

# # Process dataset for use with deep learning segmentation network
# > A guide for processing raster data and labels into ML-ready format for use with a deep-learning based semantic segmentation.

# ### Setup Notebook

# ```{admonition} **Version control**
# Colab updates without warning to users, which can cause notebooks to break. Therefore, we are pinning library versions.
# ``` 

# In[1]:


# install required libraries
get_ipython().system('pip install -q rasterio==1.2.10')
get_ipython().system('pip install -q geopandas==0.10.2')


# In[2]:


# import required libraries
import os, glob, functools, fnmatch, json, requests
from zipfile import ZipFile
from itertools import product
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image

import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio import features, mask, windows

import geopandas as gpd

from IPython.display import clear_output

import cv2

from timeit import default_timer as timer
from tqdm.notebook import tqdm


# In[3]:


# Mount google drive.
from google.colab import drive
drive.mount('/content/gdrive')


# In[4]:


# set your root directory and tiled data folders
if 'google.colab' in str(get_ipython()):
    root_dir = '/content/gdrive/My Drive/servir-tf-devseed/'  
    print('Running on Colab')
else:
    root_dir = os.path.abspath("./data/servir-tf-devseed")
    print(f'Not running on Colab, data needs to be downloaded locally at {os.path.abspath(root_dir)}')

img_dir = os.path.join(root_dir,'indices/') # or os.path.join(root_dir,'images_bright/') if using the optical tiles
label_dir = os.path.join(root_dir,'labels/')


# In[5]:


get_ipython().run_line_magic('cd', '$root_dir')


# #### Enabling GPU
# ```{Tip}
# This notebook can utilize a GPU and works better if you use one. Hopefully this notebook is using a GPU, and we can check with the following code.
# 
# If it's not using a GPU you can change your session/notebook to use a GPU. See [Instructions](https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=sXnDmXR7RDr2).
# ```

# In[ ]:


get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# ## Raster processing

# 
# <div class="alert alert-block alert-danger">
# &#9888 <b>WARNING</b> &#9888 This section contains helper functions for processing the raw raster composites and is <b>optional yet not recommended</b>, as the ML-ready tiled dataset is written to a shared drive folder, as you’ll see in the section titled <b>Read the data into memory</b>. Any cells with markdown containing an &#8681 just above them are to be skipped during the workshop.
# </div>

# Get the optical, spectral index and label mask images. 
# <div>&#8681</div> 
# 
#  ```python
# def raster_read(raster_dir):
#     print(raster_dir)
#     rasters = glob.glob(os.path.join(raster_dir,'/**/*.tif'),recursive=True)
#     print(rasters)
# 
#     # Read band metadata and arrays
#     # metadata
#     rgb = rasterio.open(os.path.join(raster_dir,'/*rgb.tif*')) #rgb
#     rgbn = rasterio.open(os.path.join(raster_dir,'/*rgbn.tif*')) #rgbn
#     indices = rasterio.open(os.path.join(raster_dir,'/*indices.tif*')) #spectral
#     labels = rasterio.open(os.path.join(raster_dir,'/*label.tif*')) #labels
#     rgb_src = rgb
#     labels_src = labels
#     target_crs = rgb_src.crs
#     print("rgb: ", rgb)
# 
#     # arrays
#     # Read and re-scale the original 16 bit image to 8 bit.
#     rgb = cv2.normalize(rgb.read(), None, 0, 255, cv2.NORM_MINMAX)
#     rgbn = cv2.normalize(rgbn.read(), None, 0, 255, cv2.NORM_MINMAX)
#     indices = cv2.normalize(indices.read(), None, 0, 255, cv2.NORM_MINMAX)
#     labels = labels.read()
#     # Check the label mask values.
#     print("values in labels array: ", np.unique(labels))
#     return raster_dir, rgb, rgbn, indices, labels, rgb_src, labels_src, target_crs
# ```

# Color correction for the optical composite. 
# <div>&#8681</div> 
# 
# ```python
# # function to increase the brightness in an image
# def change_brightness(img, value=30):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     v = cv2.add(v,value)
#     v[v > 255] = 255
#     v[v < 0] = 0
#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     return img
# ```

# Calculate relevant spectral indices
# <div>&#8681</div> 

# **WDRVI**: Wide Dynamic Range Vegetation Index \
# **NPCRI**: Normalized Pigment Chlorophyll Ratio Index \
# **SI**: Shadow Index
# 
# ```python
# # calculate spectral indices and concatenate them into one 3 channel image
# def indexnormstack(red, green, blue, nir):
#     
#     def WDRVIcalc(nir, red): 
#         a = 0.15
#         wdrvi =  (a * nir-red)/(a * nir+red)
#         return wdrvi
#     
#     def NPCRIcalc(red,blue):
#         npcri =  (red-blue)/(red+blue)
#         return npcri
#     
#     def SIcalc(red, green, blue):
#         si = ((1-red)*(1-green)*(1-blue))^(1/3)
#         return si
#     
#     def norm(arr):
#         arr_norm = (255*(arr - np.min(arr))/np.ptp(arr)) 
#         return arr_norm
#     
#     wdrvi = WDRVIcalc(nir,red) 
# 
#     npcri = WRDVIcalc(red,blue)
#     
#     si = SIcalc(red,green,blue) 
#     
#     wdrvi = wdrvi.transpose(1,2,0)
#     npcri = npcri.transpose(1,2,0)
#     si = si.transpose(1,2,0)
# 
#     index_stack = np.dstack((wdrvi, npcri, si))
#     
#     return index_stack
# ```

# If you are rasterizing the labels from a vector file (e.g. GeoJSON or Shapefile).
# <div>&#8681</div> 

# Read label shapefile into geopandas dataframe, check for invalid geometries and set to local CRS. Then, rasterize the labeled polygons using the metadata from one of the grayscale band images. In this fucntion, `geo_1` is used when there are two vector files used for labeling, e.g. Imaflora and Para. The latter is given preference over the former because it overwrites when intersections occur.
# 
# ```python
# def label(geos, labels_src):
#     geo_0 = gpd.read_file(geos[0])
#     # check for and remove invalid geometries
#     geo_0 = geo_0.loc[geo_0.is_valid]
#     # reproject training data into local coordinate reference system
#     geo_0 = geo_0.to_crs(crs={'init': target_crs})
#     #convert the class identifier column to type integer
#     geo_0['landcover_int']  = geo_0.landcover.astype(int)
#     # pair the geometries and their integer class values
#     shapes_0 = ((geom,value) for geom, value in zip(geo_0.geometry, geo_0.landcover_int)) 
#     if len(geos) > 1:
#       geo_1 = gpd.read_file(geos[1])
#       geo_1 = geo_1.loc[geo_1.is_valid]
#       geo_1 = geo_1.to_crs(crs={'init': target_crs})
#       geo_1['landcover_int']  = geo_1.landcover.astype(int)
#       shapes_1 = ((geom,value) for geom, value in zip(geo_1.geometry, geo_1.landcover_int)) 
#     else:
#       continue
# 
#     # get the metadata (height, width, channels, transform, CRS) to use in constructing the labeled image array
#     labels_src_prf = labels_src.profile
#     # construct a blank array from the metadata and burn the labels in
#     labels = features.rasterize(shapes=shapes, out_shape=(labels_src_prf['height'], labels_src_prf['width']), fill=0, all_touched=True, transform=labels_src_prf['transform'], dtype=labels_src_prf['dtype'])
#     if geo_1:
#       labels = features.rasterize(shapes=shapes_0, fill=0, all_touched=True, out=labels, transform=labels_src_prf['transform'])
#     else:
#       continue
#       
#     print("Values in labeled image: ", np.unique(labels))
# 
# 
#     return labels
# ```

# Write the processed rasters to file.
# <div>&#8681</div> 
# 
# ```python
# def save_images(raster_dir, rgb_norm, index_stack, labels, rgb_src, labels_src):
# 
#     rgb_norm_out=rasterio.open(os.path.join(raster_dir,'/rgb_byte_scaled.tif'), 'w', driver='Gtiff',
#                               width=rgb_src.width, height=rgb_src.height,
#                               count=3,
#                               crs=rgb_src.crs,
#                               transform=rgb_src.transform,
#                               dtype='uint8')
# 
#     rgb_norm_out.write(rgb_norm)
#     rgb_norm_out.close()
# 
#     indices_computed = False # change to True if using the index helper function above
#     if indices_computed:
#       index_stack = (index_stack * 255).astype(np.uint8)
#       index_stack_t = index_stack.transpose(2,0,1)
#     else:
#       index_stack_t = index_stack
# 
#     index_stack_out=rasterio.open(os.path.join(raster_dir,'/index_stack.tif'), 'w', driver='Gtiff',
#                               width=rgb_src.width, height=rgb_src.height,
#                               count=3,
#                               crs=rgb_src.crs,
#                               transform=rgb_src.transform,
#                               dtype='uint8')
# 
#     index_stack_out.write(index_stack_t)
#     index_stack_out.close()
#     
#     labels = labels.astype(np.uint8)
#     labels_out=rasterio.open(os.path.join(raster_dir,'/labels.tif'), 'w', driver='Gtiff',
#                               width=labels_src.width, height=labels_src.height,
#                               count=1,
#                               crs=labels_src.crs,
#                               transform=labels_src.transform,
#                               dtype='uint8')
# 
#     labels_out.write(labels, 1)
#     labels_out.close()
#     
#     return os.path.join(raster_dir,'/index_stack.tif'), os.path.join(raster_dir,'/labels.tif')
#   ```

# Now let's divide the optical/index stack and labeled image into 224x224 pixel tiles.
# <div>&#8681</div> 
# 
# ```python
# def tile(index_stack, labels, prefix, width, height, output_dir, brighten=False):
#     tiles_dir = os.path.join(output_dir,'tiled/')
#     img_dir = os.path.join(output_dir,'tiled/indices/')
#     label_dir = os.path.join(output_dir,'tiled/labels/')
#     dirs = [tiles_dir, img_dir, label_dir]
#     for d in dirs:
#         if not os.path.exists(d):
#             os.makedirs(d)
#     
#     def get_tiles(ds):
#         # get number of rows and columns (pixels) in the entire input image
#         nols, nrows = ds.meta['width'], ds.meta['height']
#         # get the grid from which tiles will be made 
#         offsets = product(range(0, nols, width), range(0, nrows, height))
#         # get the window of the entire input image
#         big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#         # tile the big window by mini-windows per grid cell
#         for col_off, row_off in offsets:
#             window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
#             transform = windows.transform(window, ds.transform)
#             yield window, transform
#       
#     tile_width, tile_height = width, height
#     
#     def crop(inpath, outpath, c):
#         # read input image
#         image = rasterio.open(inpath)
#         # get the metadata 
#         meta = image.meta.copy()
#         print("meta: ", meta)
#         # set the number of channels to 3 or 1, depending on if its the index image or labels image
#         meta['count'] = int(c)
#         # set the tile output file format to PNG (saves spatial metadata unlike JPG)
#         meta['driver']='PNG'
#         meta['dtype']='uint8'
#         # tile the input image by the mini-windows
#         i = 0
#         for window, transform in get_tiles(image):
#             meta['transform'] = transform
#             meta['width'], meta['height'] = window.width, window.height
#             outfile = os.path.join(outpath,"tile_%s_%s.png" % (prefix, str(i)))
#             with rasterio.open(outfile, 'w', **meta) as outds:
#                 if brighten:
#                   imw = image.read(window=window)
#                   imw = imw.transpose(1,2,0)
#                   imwb = change_brightness(imw, value=50)
#                   imwb = imwb.transpose(2,0,1)
#                   outds.write(imwb)
#                 else:
#                   outds.write(image.read(window=window))
#             i = i+1
#             
#     def process_tiles(index_flag):
#         # tile the input images, when index_flag == True, we are tiling the spectral index image, 
#         # when False we are tiling the labels image
#         if index_flag==True:
#             inpath = os.path.join(raster_dir,'/*indices_byte_scaled.tif')
#             outpath=img_dir
#             crop(inpath, outpath, 3)
#         else:
#             inpath = os.path.join(raster_dir,'/*label.tif')
#             outpath=label_dir
#             crop(inpath, outpath, 1)
#                 
#     process_tiles(index_flag=True) # tile index stack
#     process_tiles(index_flag=False) # tile labels
#     return tiles_dir, img_dir, label_dir
# ```

# Run the image processing workflow.
# 
# <div class="alert alert-block alert-warning">
# &#9888 <b>Pain point</b> &#9888 The reason for not running this raster processing code in this workshop is due to a limitation of a Google Drive based workflow. Having a VM with a mounted SSD would be a good start to solving these associated latency problems incurred from I/O of data hosted in Google Drive.
# </div>
# 
# <div>&#8681</div> 
# 
# ```python
# process = False
# if process:
#   raster_dir = os.path.join(root_dir,'/rasters/')
# 
#   # If you want to write the files out to your personal drive, set write_out = True, but I recommend trying 
#   # that in your free time because it takes about 1 hour for all composites.
# 
#   write_out = True #False
#   if write_out == True:
#       # read the rasters and scale to 8bit
#       print("reading and scaling rasters...")
#       raster_dir, rgb, rgbn, indices, labels, rgb_src, labels_src, target_crs = raster_read(raster_dir)
# 
#       # Calculate indices and combine the indices into one single 3 channel image
#       print("calculating spectral indices...")
#       index_stack = indexnormstack(rgbn.read(1), rgbn.read(2), rgbn.read(3), rgbn.read(4))
# 
#       # Rasterize labels
#       labels = label([os.path.join(root_dir,'TerraBio_Imaflora.geojson'), os.path.join(root_dir,'TerraBio_Para.geojson')], labels_src)
# 
#       # Save index stack and labels to geotiff
#       print("writing scaled rasters and labels to file...")
#       index_stack_file, labels_file = save_images(personal_dir, rgb, index_stack, labels, rgb_src, labels_src)
# 
#       # Tile images into 224x224
#       print("tiling the indices and labels...")
#       tiles_dir, img_dir, label_dir = tile(index_stack, labels, 'terrabio', 224, 224, raster_dir, brighten=False)
#   else:
#     print("Not writing to file; using data in shared drive.")
# 
# else:
#   print("Using pre-processed dataset.")
# ```

# ## Read the data into memory

# #### Getting set up with the data
# 
# ```{important}
# Create drive shortcuts of the tiled imagery to your own My Drive Folder by Right-Clicking on the Shared folder `servir-tf-devseed`. Then, this folder will be available at the following path that is accessible with the google.colab `drive` module: `'/content/gdrive/My Drive/servir-tf-devseed/'`
# ```
# 
# We'll be working witht he following folders in the `servir-tf-devseed` folder:
# ```
# servir-tf-devseed/
# ├── images/
# ├── images_bright/
# ├── indices/
# ├── indices_800/
# ├── labels/
# ├── labels_800/
# ├── background_list_train.txt
# ├── train_list_clean.txt
# └── terrabio_classes.csv
# ```

# In[6]:


# Read the classes
class_index = pd.read_csv(os.path.join(root_dir,'terrabio_classes.csv'))
class_names = class_index.class_name.unique()
print(class_index) 


# ```{important}
# Normally we would read the image files from the directories and then process forward from there with background removal with the next **three** illustrated functions, however, due to slow I/O in Google Colab we will read the images list with 90% background removal already performed from a pre-saved list in the shared drive.
# ```
# 
# Get lists of image and label tile pairs for training and testing.
# <div>&#8681</div> 
# 
# ```python
# def get_train_test_lists(imdir, lbldir):
#   imgs = glob.glob(os.path.join(imdir,"/*.png"))
#   #print(imgs[0:1])
#   dset_list = []
#   for img in imgs:
#     filename_split = os.path.splitext(img) 
#     filename_zero, fileext = filename_split 
#     basename = os.path.basename(filename_zero) 
#     dset_list.append(basename)
#     
#   x_filenames = []
#   y_filenames = []
#   for img_id in dset_list:
#     x_filenames.append(os.path.join(imdir, "{}.png".format(img_id)))
#     y_filenames.append(os.path.join(lbldir, "{}.png".format(img_id)))
#     
#   print("number of images: ", len(dset_list))
#   return dset_list, x_filenames, y_filenames
# 
# train_list, x_train_filenames, y_train_filenames = get_train_test_lists(img_dir, label_dir)
# 
# ```
# number of images:  37350
# 
# Check for the proportion of background tiles. This takes a while. So we can skip by loading from saved results.
# <div>&#8681</div> 
# 
# ```python
# skip = True
# 
# if not skip:
#   background_list_train = []
#   for i in train_list: 
#       # read in each labeled images
#       # print(os.path.join(label_dir,"{}.png".format(i))) 
#       img = np.array(Image.open(os.path.join(label_dir,"{}.png".format(i))))  
#       # check if no values in image are greater than zero (background value)
#       if img.max()==0:
#           background_list_train.append(i)
#           
#   print("Number of background images: ", len(background_list_train))
# 
#   with open(os.path.join(root_dir,'background_list_train.txt'), 'w') as f:
#     for item in background_list_train:
#         f.write("%s\n" % item)
# 
# else:
#   background_list_train = [line.strip() for line in open("background_list_train.txt", 'r')]
#   print("Number of background images: ", len(background_list_train))
# ```
# Number of background images:  36489
# 
# We will keep only 10% of the total. Too many background tiles can cause a form of class imbalance.
# <div>&#8681</div> 
# 
# ```python
# background_removal = len(background_list_train) * 0.9
# train_list_clean = [y for y in train_list if y not in background_list_train[0:int(background_removal)]]
# 
# x_train_filenames = []
# y_train_filenames = []
# 
# for i, img_id in zip(tqdm(range(len(train_list_clean))), train_list_clean):
#   pass 
#   x_train_filenames.append(os.path.join(img_dir, "{}.png".format(img_id)))
#   y_train_filenames.append(os.path.join(label_dir, "{}.png".format(img_id)))
# 
# print("Number of background tiles: ", background_removal)
# print("Remaining number of tiles after 90% background removal: ", len(train_list_clean))
# ```
# Number of background tiles:  32840
# 
# Remaining number of tiles after 90% background removal:  4510

# ```{important}
# The cell below contains the shortcut read of prepped training image list. 
# ```

# In[7]:


def get_train_test_lists(imdir, lbldir):
  train_list = [line.strip() for line in open("train_list_clean.txt", 'r')]

  x_filenames = []
  y_filenames = []
  for img_id in train_list:
    x_filenames.append(os.path.join(imdir, "{}.png".format(img_id)))
    y_filenames.append(os.path.join(lbldir, "{}.png".format(img_id)))

  print("Number of images: ", len(train_list))
  return train_list, x_filenames, y_filenames


# In[8]:


train_list, x_train_filenames, y_train_filenames = get_train_test_lists(img_dir, label_dir)


# Now that we have our set of files we want to use for developing our model, we need to split them into three sets: 
# * the training set for the model to learn from
# * the validation set that allows us to evaluate models and make decisions to change models
# * and the test set that we will use to communicate the results of the best performing model (as determined by the validation set)
# 
# We will split index tiles and label tiles into train, validation and test sets: 70%, 20% and 10%, respectively.
# 

# In[9]:


x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = train_test_split(x_train_filenames, y_train_filenames, test_size=0.3, random_state=42)
x_val_filenames, x_test_filenames, y_val_filenames, y_test_filenames = train_test_split(x_val_filenames, y_val_filenames, test_size=0.33, random_state=42)

num_train_examples = len(x_train_filenames)
num_val_examples = len(x_val_filenames)
num_test_examples = len(x_test_filenames)

print("Number of training examples: {}".format(num_train_examples))
print("Number of validation examples: {}".format(num_val_examples))
print("Number of test examples: {}".format(num_test_examples))


# ```{warning} **Long running cell** \
# The code below checks for values in train, val, and test partitions. We won't run this since it takes over 10 minutes on colab due to slow IO.
# ``` 
# <div>&#8681</div> 
# 
# ```python
# vals_train = []
# vals_val = []
# vals_test = []
# 
# def get_vals_in_partition(partition_list, x_filenames, y_filenames):
#   for x,y,i in zip(x_filenames, y_filenames, tqdm(range(len(y_filenames)))):
#       pass 
#       try:
#         img = np.array(Image.open(y)) 
#         vals = np.unique(img)
#         partition_list.append(vals)
#       except:
#         continue
# 
# def flatten(partition_list):
#     return [item for sublist in partition_list for item in sublist]
# 
# get_vals_in_partition(vals_train, x_train_filenames, y_train_filenames)
# get_vals_in_partition(vals_val, x_val_filenames, y_val_filenames)
# get_vals_in_partition(vals_test, x_test_filenames, y_test_filenames)
# ```
# 

# ``` python
# print("Values in training partition: ", set(flatten(vals_train)))
# print("Values in validation partition: ", set(flatten(vals_val)))
# print("Values in test partition: ", set(flatten(vals_test)))
# ```
# Values in training partition:  {0, 1, 2, 3, 4, 5, 6, 7, 8}
# 
# Values in validation partition:  {0, 1, 2, 3, 4, 5, 6, 7, 8}
# 
# Values in test partition:  {0, 1, 2, 3, 4, 5, 6, 7, 8}

# ### Visualize the data

# ```{warning} **Long running cell** \
# The code below loads foreground examples randomly. We won't run this since it takes over a while on colab due to slow I/O.
# ```
# <div>&#8681</div> 
# 
# ```python
# display_num = 3
# 
# background_list_train = [line.strip() for line in open("background_list_train.txt", 'r')]
# 
# # select only for tiles with foreground labels present
# foreground_list_x = []
# foreground_list_y = []
# for x,y in zip(x_train_filenames, y_train_filenames): 
#     try:
#       filename_split = os.path.splitext(y) 
#       filename_zero, fileext = filename_split 
#       basename = os.path.basename(filename_zero) 
#       if basename not in background_list_train:
#         foreground_list_x.append(x)
#         foreground_list_y.append(y)
#       else:
#         continue
#     except:
#       continue
# 
# num_foreground_examples = len(foreground_list_y)
# 
# # randomlize the choice of image and label pairs
# r_choices = np.random.choice(num_foreground_examples, display_num)
# ```

# ```{important}
# Instead, we will read and plot a few sample foreground training images and labels from their pathnames. Note: this may still take a few execution tries to work. Google colab in practice takes some time to connect to data in Google Drive, so sometimes this returns an error on the first (few) attempt(s).
# ```

# In[12]:


display_num = 3

background_list_train = [line.strip() for line in open("background_list_train.txt", 'r')]

foreground_list_x = [
                     f'{img_dir}/tile_terrabio_15684.png', 
                     f'{img_dir}/tile_terrabio_23056.png', 
                     f'{img_dir}/tile_terrabio_21877.png'
                     ]

foreground_list_y = [
                     f'{label_dir}/tile_terrabio_15684.png', 
                     f'{label_dir}/tile_terrabio_23056.png', 
                     f'{label_dir}/tile_terrabio_21877.png'
                     ]
                     
# confirm files exist
for fx, fy in zip(foreground_list_x, foreground_list_y):
  if os.path.isfile(fx) and os.path.isfile(fy):
    print(fx, " and ", fy, " exist.")
  else:
    print(fx, " and ", fy, " don't exist.")


num_foreground_examples = len(foreground_list_y)

# randomlize the choice of image and label pairs
#r_choices = np.random.choice(num_foreground_examples, display_num)

plt.figure(figsize=(10, 15))
for i in range(0, display_num * 2, 2):
  #img_num = r_choices[i // 2]
  img_num = i // 2
  x_pathname = foreground_list_x[img_num]
  y_pathname = foreground_list_y[img_num]
  
  plt.subplot(display_num, 2, i + 1)
  plt.imshow(mpimg.imread(x_pathname))
  plt.title("Original Image")
  
  example_labels = Image.open(y_pathname)
  label_vals = np.unique(np.array(example_labels))
  
  plt.subplot(display_num, 2, i + 2)
  plt.imshow(example_labels)
  plt.title("Masked Image")  
  
plt.suptitle("Examples of Images and their Masks")
plt.show()

