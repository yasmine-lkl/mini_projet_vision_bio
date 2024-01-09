#!/usr/bin/env python
# coding: utf-8

# In[4]:


# TP 1 VISION PAR ORDINATEUR


# In[4]:


#importation de toutes les bibliothèques necessaires 
import skimage 
from skimage import data
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.transform import resize
from skimage.draw import ellipse


from skimage.color import rgb2hsv
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
skimage.__version__
from sklearn.cluster import KMeans
from collections import Counter 
import numpy as np
import math
import cv2 as cv
from PIL import Image
import sys


# In[5]:


plt.figure()
image=plt.imread("CT_SubCortInf.jpg")
plt.title("image du cerveau original")
plt.imshow(image)
plt.show()


# In[6]:


#affichage de l histogramme
plt.figure(figsize=(15,15))
plt.subplot(2,2,1) #creer 2 graphiques dans des multiples fenteres
plt.imshow(image,cmap='gray') #afficher les donnees sous forme d'image (sur un raster régulier 2D)
plt.title('image')
plt.xticks([])
plt.yticks([])
#construction de l'histogramme
img=image[...,1]
plt.subplot(2,2,2)
plt.hist(image.ravel(),20,[0,255]) #la bib hist permet de tracer l'histogramme
plt.title('histogramme') #afficher le titre vv

plt.show()


# In[7]:


"""la premiere harmonique correspond au niveau du gris le plus chaud, donc ça represente la matiere grise MG
la 2eme harmonique (image>=250) correspond au niveau du gris le plus clair==blanc, donc represente la matiere blanche
entre l'intervalle [60,160] : ça represente le cerveau = liquide céphalo-rachidien

(si le niveau de gris est plus petit que 50 on le met a 0)
"""


# In[8]:


"""Conversion du format de couleur RVB au format de couleur HSV """
# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))
 
# Sample Image of scikit-image package
#image = data.image()
plt.subplot(1, 2, 1)
 
# Displaying the sample image
plt.imshow(image)
 
# Converting RGB Image to HSV Image
hsv_image = rgb2hsv(image)
plt.subplot(1, 2, 2)
 
# Displaying the sample image - HSV Format
hsv_image_colorbar = plt.imshow(hsv_image)
 
# Adjusting colorbar to fit the size of the image
plt.colorbar(hsv_image_colorbar, fraction=0.046, pad=0.04)


# In[9]:


"""L'espace colorimétrique RVB(code couleur) décrit les proportions
de rouge, de vert et de bleu dans une couleur. Dans le système de couleurs HSV, 
les couleurs sont définies en termes de teinte, de saturation et
de valeur.

La fonction skimage.color.rgb2hsv() est utilisée pour convertir 
une image RVB au format HSV"""


# In[10]:


"""on peut remarquer que la partie droite afficher quelques petites taches en noir"""


# In[11]:


"""Ensuite, on affiche les trois parties de l'image: la tête entière, la boîte cranienne et le cerveau seul  """


# In[12]:


#tete entiere :
#on affiche l'image initiale 
plt.figure(figsize=(15,15))  #dimsensions de l'image
plt.subplot(2,2,1)
plt.title('tête entière')
plt.imshow(img, cmap='gray')
#boîte cranienne
boite = (img > 240)
plt.subplot(2,2,2)
plt.title("Boîte cranienne")
plt.imshow(boite, cmap = "gray")
#cerveau seul
#on enleve la boite cranienne, ce qui nous permet de procéder au calcul pour pouvoir afficher le cerveau seul, donc on prend la boite qui correspond au boite cranienne moins notre image
cerveau=boite-img
plt.subplot(2,2,3)
plt.title('cerveau seul')
plt.imshow(cerveau, cmap='gray')


# In[13]:


"""on remarque bien que l'image du cerveau seul n'est pas parfaite, il faut se débarsser du contour qui n'est pas inclu dans la partie cerveau
on pourra réaliser un masque qui correspondra seulement à l'image serveau seul"""


# In[14]:


kernel = np.ones((15,15),np.uint8) #definir le kernel
closing=cv.morphologyEx(cerveau, cv.MORPH_OPEN, kernel, iterations=1)
"""le gradient morphologique applique d'abord l'érosion
et la dilatation individuellement sur l'image, puis calcule la différence
entre l'image érodée et dilatée. La sortie sera un aperçu de l'image 
donnée
"""

print(closing)
for i in range(len(closing)):
    for j in range (len(closing[i])):
        if closing[i,j]<20: #verifier si les lignes et colonnes sont inferieurs a 20
            cerveau[i,j]=0 #on met le niveau de gris le plus petit à 0
mask=cerveau #on filtre l'image
plt.figure(figsize=(5,5))  
plt.imshow(cerveau,cmap='gray')
plt.title('cerveau nettoyé')
plt.show()


# In[15]:


"""on remarque qu'on a bien réussi à enlever le contour du cerveau non désiré"""


# In[16]:


mask = cv.cvtColor(cerveau, cv.COLOR_BGR2RGB)

# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = cerveau.reshape((-1,3))
 
# Convert to float type
pixel_vals = np.float32(pixel_vals)


# In[17]:


"""On peut retirer 3 Clusters pour voir comment l'image change"""

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
#on effectue un clustering k-means avec un nombre de clusters égal à 3
k = 3
#des centres aléatoires sont également initialement choisis pour le clustering k-means
retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
 
#convertir les dimensions en 8 bits 
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
 
# remodeler les données dans les dimensions de l'image d'origine
segmented_image = segmented_data.reshape((cerveau.shape))
plt.title('segmented image')
plt.imshow(segmented_image)


# In[18]:


"""passons maintenant à la ligne de symétrie du cerveau"""


# In[19]:


Y=[]
X=[]
regions = regionprops(cerveau)
for props in regions:
    
    y0, x0 = props.centroid
    if math.isnan(y0)==False:
        Y.append(y0)
    if math.isnan(x0)==False: 
        X.append(x0)

Y=np.mean(Y)
X=np.mean(X)
x=np.linspace(0,400,2)
y=-10*x+3550 #equiation qui permet de passer par le centroide

plt.plot(x, y, color="red", linewidth=1)
plt.plot(X,Y, marker=".", color="red")
plt.title('axe passant par protubérance occipitale')
plt.imshow(cerveau,cmap='gray')


# In[26]:


from PIL import Image
image=cv.imread('cerveau.jpg')
original_image=Image.open("cerveau.jpg")
rotated_image1=original_image.rotate(180) #faire une rotation de 180deg
rotated_image2=original_image.transpose(Image.ROTATE_90) #transposer l'image (angle=90deg)
rotated_image3=original_image.rotate(60) #angle =60deg
rotated_image1.show()
rotated_image2.show()
rotated_image3.show()

images=[cerveau,rotated_image1,rotated_image2,rotated_image3]

f,axarr=plt.subplots(2,2)
axarr[0,0].imshow(images[0]) 
axarr[0,1].imshow(images[1]) #affiche rotated_image1
axarr[1,0].imshow(images[2]) #affiche rotated_image2
axarr[1,1].imshow(images[3]) #affiche rotated_image3
plt.show()


# In[21]:


#passons maintenant à la comparaison ente les elements gauche-droite
a=[300,300] #dimensions de l'axe bleu qui passe bien par le centre 
b=[0,1000]


plt.figure(figsize=(5,5))

plt.plot(x, y, color="red", linewidth=2)
plt.plot(a,b,color="blue",linewidth=2)
show=rotate(segmented_image, angle=np.arctan(11))
plt.imshow(show)


# In[22]:


#couper l'image en deux
plt.figure(figsize=(5,5))
sh=rotate(segmented_image, angle=np.arctan(11))

cropped1 = sh[:700, 12:300]
cropped2 = sh[:700, 300:]

res = resize(cropped2, (722, 588))
r = resize(segmented_image, (722, 588))

plt.figure(figsize=(10, 10))
plt.subplot(2,2,1)
plt.title('partie gauche')
plt.imshow(cropped1)
plt.subplot(2,2,2)
plt.title('partie droite')
plt.imshow(cropped2)


# In[31]:


image=cv.imread("cerveau.jpg")
(h,w,c)=image.shape
img2D=image.reshape(h*w,c)
print(img2D)
print(img2D.shape)
plt.imshow(image)


# In[36]:


#clusters=k
kmeans=KMeans(k)
labels=kmeans.fit_predict(img2D) #fit_predict est une méthode qui renvoie les étiquettes de l'ensemble de données 
print(labels)


# In[39]:


counter=Counter(labels)
print(counter)
print(kmeans.cluster_centers_)


# In[65]:


rgb=kmeans.cluster_centers_.round(0).astype(int)
rgb[1][0:3]=0
rgb[1][1]=254
rgb[2][0:3]=0
rgb[2][0]=254
print(rgb)


# In[66]:


img=np.reshape(rgb[labels],(h,w,c))
plt.imshow(img)


# In[ ]:




