#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Importing Necessary Libraries
from skimage import data
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import io
# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))
 
# Sample Image of scikit-image package
img = io.imread('D:\sample 1\IMG_0665.jpg')
plt.subplot(1, 2, 1)
 
# Displaying the sample image
plt.imshow(img)
 
# Converting RGB image to Monochrome
gray_img = rgb2gray(img)
plt.subplot(1, 2, 2)
 
# Displaying the sample image - Monochrome
# Format
plt.imshow(gray_img, cmap="gray")


# In[6]:


# Importing Necessary Libraries
from skimage import data
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
 
# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))
 
# Sample Image of scikit-image package
img = io.imread('D:\sample 1\IMG_0665.jpg')
plt.subplot(1, 2, 1)
 
# Displaying the sample image
plt.imshow(img)
 
# Converting RGB Image to HSV Image
hsv_img = rgb2hsv(img)
plt.subplot(1, 2, 2)
 
# Displaying the sample image - HSV Format
hsv_img_colorbar = plt.imshow(hsv_img)
 
# Adjusting colorbar to fit the size of the image
plt.colorbar(hsv_img_colorbar, fraction=0.046, pad=0.04)


# In[7]:


# Importing Necessary Libraries
# Displaying the sample image - Monochrome Format
from skimage import data
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
 
# Sample Image of scikit-image package
img = io.imread('D:\sample 1\IMG_0665.jpg')
gray_img = rgb2gray(img)
 
# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))
 
for i in range(10):
   
  # Iterating different thresholds
  binarized_gray = (gray_img > i*0.1)*1
  plt.subplot(5,2,i+1)
   
  # Rounding of the threshold
  # value to 1 decimal point
  plt.title("Threshold: >"+str(round(i*0.1,1)))
   
  # Displaying the binarized image
  # of various thresholds
  plt.imshow(binarized_gray, cmap = 'gray')
   
plt.tight_layout()


# In[8]:


import matplotlib.pyplot as plt
from skimage import io
from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation

# Input data
img = io.imread('D:\sample 1\IMG_0665.jpg')

# Compute a mask
lum = color.rgb2gray(img)
mask = morphology.remove_small_holes(
    morphology.remove_small_objects(
        lum > 0.4, 500),
    500)

mask = morphology.opening(mask, morphology.disk(3))

# SLIC result
slic = segmentation.slic(img, n_segments=200, start_label=1)

# maskSLIC result
m_slic = segmentation.slic(img, n_segments=100, mask=mask, start_label=1)

# Display result
fig, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
ax1, ax2, ax3, ax4 = ax_arr.ravel()

ax1.imshow(img)
ax1.set_title('Original image')

ax2.imshow(mask, cmap='gray')
ax2.set_title('Mask')

ax3.imshow(segmentation.mark_boundaries(img, slic))
ax3.contour(mask, colors='red', linewidths=1)
ax3.set_title('SLIC')

ax4.imshow(segmentation.mark_boundaries(img, m_slic))
ax4.contour(mask, colors='red', linewidths=1)
ax4.set_title('maskSLIC')

for ax in ax_arr.ravel():
    ax.set_axis_off()

plt.tight_layout()
plt.show()


# In[9]:


#กำหนด threshold > 0.4 เอง
from skimage import data
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import io
# Sample Image of scikit-image package
img = io.imread('D:\sample 1\IMG_0665.jpg')
gray_img = rgb2gray(img)
plt.figure(figsize=(15, 15))
binarized_gray = (gray_img > 4*0.1)*1
plt.subplot(5,2,3+1)
plt.title("Threshold: >"+str(round(4*0.1,1)))
plt.imshow(binarized_gray, cmap = 'gray')
plt.tight_layout()


# In[10]:


pip install scikit-image


# In[22]:


#กำหนด mask SLIC
import matplotlib.pyplot as plt
from skimage import io
from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation

# Input data
img = io.imread('D:\sample 1\IMG_0665.jpg')

# Compute a mask
lum = color.rgb2gray(img)
mask = morphology.remove_small_holes(
    morphology.remove_small_objects(
        lum > 0.4, 500),
    500)

mask = morphology.opening(mask, morphology.disk(3))

# maskSLIC result
m_slic = segmentation.slic(img, n_segments=100, mask=mask, start_label=1)

# Display result
fig, ax4 = plt.subplots(sharex=True, sharey=True, figsize=(10, 10))

ax4.imshow(segmentation.mark_boundaries(img, m_slic))
ax4.contour(mask, colors='red', linewidths=1)
ax4.set_title('maskSLIC')
ax.set_axis_on()

plt.tight_layout()
plt.axis('off')
plt.show()


# In[12]:


#กำหนด SLIC 
import matplotlib.pyplot as plt
from skimage import io
from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation

# Input data
img = io.imread('D:\sample 1\IMG_0665.jpg')

# Compute a mask
lum = color.rgb2gray(img)
mask = morphology.remove_small_holes(
    morphology.remove_small_objects(
        lum > 0.4, 500),
    500)

mask = morphology.opening(mask, morphology.disk(3))

# SLIC result
slic = segmentation.slic(img, n_segments=200, start_label=1)

# Display result
fig, ax3 = plt.subplots(sharex=True, sharey=True, figsize=(10, 10))

ax3.imshow(segmentation.mark_boundaries(img, slic))
ax3.contour(mask, colors='red', linewidths=1)
ax3.set_title('SLIC')
ax.set_axis_on()

plt.tight_layout()
plt.show()


# In[13]:


pip install aspose-words


# In[14]:


# save as tiff threshold
import aspose.words as aw

doc = aw.Document()
builder = aw.DocumentBuilder(doc)

shape = builder.insert_image("D:\sample 1\download.png")
shape.image_data.save("threshold.tiff")


# In[15]:


# save as tiff maskSLIC
import aspose.words as aw

doc = aw.Document()
builder = aw.DocumentBuilder(doc)

shape = builder.insert_image("D:\sample 1\download1.png")
shape.image_data.save("maskSLIC.tiff")


# In[1]:


#กำหนด mask
import matplotlib.pyplot as plt
from skimage import io
from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation

# Input data
img = io.imread('D:\sample 1\IMG_0665.jpg')

# Compute a mask
lum = color.rgb2gray(img)
mask = morphology.remove_small_holes(
    morphology.remove_small_objects(
        lum > 0.4, 500),
    500)

mask = morphology.opening(mask, morphology.disk(3))


# Display result
fig, ax2 = plt.subplots(sharex=True, sharey=True, figsize=(10, 10))

ax2.imshow(mask, cmap='gray')
#ax2.set_title('Mask')

plt.tight_layout()
plt.axis('off')
plt.show()


# In[1]:


#original
import matplotlib.pyplot as plt
from skimage import io
from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation

# Input data
img = io.imread('D:\sample 1\IMG_0665.jpg')

# Compute a mask
lum = color.rgb2gray(img)
mask = morphology.remove_small_holes(
    morphology.remove_small_objects(
        lum > 0.4, 500),
    500)

mask = morphology.opening(mask, morphology.disk(3))

# Display result
fig,ax1= plt.subplots(sharex=True, sharey=True, figsize=(10, 10))
ax1.imshow(img)
#ax1.set_title('Original image')

plt.tight_layout()
plt.axis('off')
plt.show()


# In[ ]:




