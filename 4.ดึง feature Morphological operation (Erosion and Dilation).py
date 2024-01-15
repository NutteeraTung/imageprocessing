#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
import aspose.words as aw
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import tiffile as tiff
import os
import pandas as pd
import cv2
from skimage import data
from skimage import io
from skimage import morphology
from skimage import segmentation
from skimage.color import rgb2hsv
from skimage.color import rgb2gray
from skimage.draw import bezier_curve
from pylab import *
from PIL import Image
from matplotlib import transforms


# In[16]:


#Erosion


# In[17]:


#reading sample image
img=cv2.imread("D:\sample 1\img_bwa.png")


# In[18]:


#Display image
window_name='imagefirst'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
plt.figure(figsize=(15, 15))
cv2.imshow(window_name,img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[19]:


#5x5 kernel with full of ones
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)


# In[20]:


#Display image
window_name='erosion'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name,erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


#Dilation


# In[11]:


img_third=cv2.imread("D:\sample 1\img_bwa.png")


# In[12]:


window_name='imagethird'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name,img_third)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


kernel_3 = np.ones((5,5),np.uint8)
dilation = cv2.dilate(img_third,kernel_3,iterations = 1)


# In[14]:


window_name='imagedialte'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name,dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




