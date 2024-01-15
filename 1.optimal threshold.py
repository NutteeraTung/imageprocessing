#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install scikit-image


# In[24]:


from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from pylab import *


# In[31]:


image = io.imread('D:\sample 1\IMG_0665.jpg')
plt.imshow(image);
plt.axis('off')


# In[33]:


# Importing necessary libraries
from skimage import data
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
 
# Setting plot size to 15, 15
plt.figure(figsize=(15, 15))
 
# Sample Image of scikit-image package
image = io.imread('D:\sample 1\IMG_0665.jpg')
gray_image = rgb2gray(image)
 
# Computing Otsu's thresholding value
threshold = filters.threshold_otsu(gray_image)
 
# Computing binarized values using the obtained
# threshold
binarized_image = (gray_image > threshold)*1
plt.subplot(2,2,1)
plt.title("Threshold: >"+str(threshold)) #ได้ค่า threshold ที่ดีที่สุดออกมา 
 
# Displaying the binarized image
plt.imshow(binarized_image, cmap = "gray")
 
# Computing Ni black's local pixel
# threshold values for every pixel
threshold = filters.threshold_niblack(gray_image)
 
# Computing binarized values using the obtained
# threshold
binarized_image = (gray_image > threshold)*1
plt.subplot(2,2,2)
plt.title("Niblack Thresholding")
 
# Displaying the binarized image
plt.imshow(binarized_image, cmap = "gray")
 
# Computing Sauvola's local pixel threshold
# values for every pixel - Not Binarized
threshold = filters.threshold_sauvola(gray_image)
plt.subplot(2,2,3)
plt.title("Sauvola Thresholding")
 
# Displaying the local threshold values
plt.imshow(threshold, cmap = "gray")
 
# Computing Sauvola's local pixel
# threshold values for every pixel - Binarized
binarized_image = (gray_image > threshold)*1
plt.subplot(2,2,4)
plt.title("Sauvola Thresholding - Converting to 0's and 1's")
 
# Displaying the binarized image
plt.imshow(binarized_image, cmap = "gray")


# In[38]:


#กำหนด Sauvola Thresholding
# Setting plot size to 10, 10
plt.figure(figsize=(10, 10))
# Sample Image of scikit-image package
image = io.imread('D:\sample 1\IMG_0665.jpg')
gray_image = rgb2gray(image)
# Computing Otsu's thresholding value
threshold = filters.threshold_otsu(gray_image)
# Computing Sauvola's local pixel threshold
# values for every pixel - Not Binarized
threshold = filters.threshold_sauvola(gray_image)
plt.subplot(2,2,3)
plt.title("Sauvola Thresholding")
 
# Displaying the local threshold values
plt.imshow(threshold, cmap = "gray")
plt.tight_layout()
plt.axis('off')


# In[ ]:




