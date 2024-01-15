#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install scikit-image


# In[2]:


from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[3]:


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


# In[4]:


from skimage import io
image = io.imread(r'D:\sample 1\img_bwa.tiff') #D:\sample 1\crop1.tif
plt.imshow(image);


# In[5]:


import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color


# In[6]:


image_gray = color.rgb2gray(image) 
image_show(image_gray);


# In[7]:


#Supervised segmentation


# In[8]:


#active contour segmentation (snakes)
def circle_points(resolution, center, radius):
    """
    Generate points which define a circle on an image.Centre refers to the centre of the circle
    """   
    radians = np.linspace(0, 2*np.pi, resolution)
    c = center[1] + radius*np.cos(radians)#polar co-ordinates
    r = center[0] + radius*np.sin(radians)
    
    return np.array([c, r]).T
# Exclude last point because a closed path should not have duplicate points
points = circle_points(1000,[200, 200], 50) #[:-10]


# In[9]:


#แสดงผลบริเวณพื้นที่วงกลมสีแดง
fig, ax = image_show(image)
ax.plot(points[:,0], points[:,1], '--r', lw=3)


# In[10]:


import numpy
from matplotlib import pyplot, transforms
snake = seg.active_contour(image_gray, points)
fig, ax = image_show(image)
#the base transformation of the data points is needed
snake = seg.active_contour(image_gray, points,alpha=0.6,beta=3)
ax.plot(points[:, 1], points[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=1.5);
#alpha will make this snake contract faster while beta makes the snake smoother


# In[11]:


#random walker segmentation
image_labels = np.zeros(image_gray.shape, dtype=np.uint8)


# In[12]:


indices = draw.circle_perimeter(200, 200,20)#from here
image_labels[indices] = 1
image_labels[points[:, 1].astype(np.int), points[:, 0].astype(np.int)] = 2
image_show(image_labels);


# In[13]:


image_segmented = seg.random_walker(image_gray, image_labels)
# Check our results
fig, ax = image_show(image_gray)
ax.imshow(image_segmented == 1, alpha=0.5);


# In[14]:


image_segmented = seg.random_walker(image_gray, image_labels, beta = 1000)
# Check our results
fig, ax = image_show(image_gray)
ax.imshow(image_segmented == 1, alpha=0.5);


# In[ ]:


#unsuperviesd segmentation

