#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#วิธีที่ 1 overlay


# In[1]:


import cv2
import numpy as np
img1 = cv2.imread(r'D:\sample 1\overlay0665\mask.png',-4500) #ข้างหลังขยายรูป
img2 = cv2.imread(r'D:\sample 1\overlay0665\origi.png',-4500)

img_bwa = cv2.bitwise_and(img1,img2)
img_bwo = cv2.bitwise_or(img1,img2)
img_bwx = cv2.bitwise_xor(img1,img2)

cv2.imshow("Bitwise AND of Image 1 and 2", img_bwa)
cv2.imshow("Bitwise OR of Image 1 and 2", img_bwo)
cv2.imshow("Bitwise XOR of Image 1 and 2", img_bwx)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


isWritten = cv2.imwrite('D:\sample 1\img_bwa.png', img_bwa)
if isWritten:
 print('Image is successfully saved as file.')


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import pandas as pd

img = tiff.imread('D:\sample 1\img_bwa.tiff')
img_array = np.array(img)
plt.imshow(img)
plt.axis('off')


# In[4]:


type(img)


# In[6]:


from PIL import Image
im = Image.open('D:\sample 1\img_bwa.tiff')


# In[ ]:


im.show()


# In[4]:


imarray


# In[ ]:


#วิธีที่ 2 ไม่ต้อง overlay


# In[2]:


import numpy as np
import glob
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters


# In[3]:


# load the image
image = skimage.io.imread("D:\sample 2\IMG_0719.jpg")

fig, ax = plt.subplots()
plt.imshow(image)
plt.axis('off')


# In[4]:


# convert the image to grayscale
gray_image = skimage.color.rgb2gray(image)

# blur the image to denoise
blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)

fig, ax = plt.subplots()
plt.imshow(blurred_image, cmap="gray")
plt.axis('off')


# In[5]:


# create a histogram of the blurred grayscale image
histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))

fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim(0, 1.0)
plt.show()


# In[6]:


# create a mask based on the threshold
t = 0.4
binary_mask = blurred_image > t

fig, ax = plt.subplots()
plt.imshow(binary_mask, cmap="gray")
plt.show()


# In[10]:


# use the binary_mask to select the "interesting" part of the image
selection = image.copy()
selection[~binary_mask] = 0

fig, ax = plt.subplots()
plt.imshow(selection)
plt.tight_layout()
plt.axis('off')


# In[ ]:




