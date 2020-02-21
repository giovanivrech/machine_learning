#!/usr/bin/env python
# coding: utf-8

# In[69]:


import cv2
import matplotlib.pyplot as plt
import shutil
import numpy as np

from os import listdir, path, makedirs
from os.path import isfile, join
from sklearn.metrics import accuracy_score


# In[2]:


img_face_1 = cv2.imread('images/cropped_faces/s01_01.jpg')
img_face_1 = cv2.cvtColor(img_face_1, cv2.COLOR_BGR2RGB)

img_face_2 = cv2.imread('images/cropped_faces/s02_01.jpg')
img_face_2 = cv2.cvtColor(img_face_2, cv2.COLOR_BGR2RGB)

img_face_3 = cv2.imread('images/cropped_faces/s03_03.jpg')
img_face_3 = cv2.cvtColor(img_face_3, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,10))

plt.subplot(131)
plt.title('People 01')
plt.imshow(img_face_1)

plt.subplot(132)
plt.title('People 02')
plt.imshow(img_face_2)

plt.subplot(133)
plt.title('People 03')
plt.imshow(img_face_3)

plt.show()


# In[3]:


img_face_1.shape


# In[4]:


img_face_2.shape


# In[5]:


img_face_3.shape


# In[6]:


faces_directory = "images/cropped_faces/"
list_arq_faces = [f for f in listdir(faces_directory) if isfile(join(faces_directory, f))]


# In[7]:


# separating images for training and testing

faces_path_train = "images/train/"
faces_path_test = "images/test/"

if not path.exists(faces_path_train):
    makedirs(faces_path_train)
    
if not path.exists(faces_path_test):
    makedirs(faces_path_test)
    
for arq in list_arq_faces:
    people = arq[1:3]
    number = arq[4:6]
    
    if int(number) <= 10:
        shutil.copyfile(faces_directory + arq, faces_path_train + arq)
    else:
        shutil.copyfile(faces_directory + arq, faces_path_test + arq)


# In[29]:


def standardize_image(path_img):
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_LANCZOS4)    
    return img


# In[13]:


list_faces_train = [f for f in listdir(faces_path_train) if isfile(join(faces_path_train, f))]
list_faces_test = [f for f in listdir(faces_path_test) if isfile(join(faces_path_test, f))]


# In[20]:


list_faces_train[0]


# In[15]:


list_faces_test[0]


# In[31]:


data_train, people = [], []

for i, arq in enumerate(list_faces_train):
    img_path = faces_path_train + arq
    img = standardize_image(img_path)
    data_train.append(img)
    subject = arq[1:3]
    people.append(int(subject))


# In[32]:


len(data_train)


# In[33]:


len(people)


# In[34]:


data_test, people_test = [], []

for i, arq in enumerate(list_faces_test):
    img_path = faces_path_test + arq
    img = standardize_image(img_path)
    data_test.append(img)
    subject = arq[1:3]
    people_test.append(int(subject))


# In[35]:


len(data_test)


# In[36]:


len(people_test)


# In[37]:


plt.imshow(data_train[0], cmap = "gray")
plt.title(people[0])


# In[38]:


plt.imshow(data_test[0], cmap = "gray")
plt.title(people_test[0])


# In[42]:


# training model with classifier Eingenface
people = np.asarray(people, dtype = np.int32)
people_test = np.asarray(people_test, dtype = np.int32)

model_eingenfaces = cv2.face.EigenFaceRecognizer_create()
model_eingenfaces.train(data_train, people)


# In[44]:


plt.figure(figsize = (20, 10))

plt.subplot(121)
plt.title("People " + str(people_test[6]))
plt.imshow(data_test[6], cmap = "gray")

plt.subplot(122)
plt.title("People " + str(people_test[7]))
plt.imshow(data_test[7], cmap = "gray")

plt.show()


# In[48]:


predict = model_eingenfaces.predict(data_test[6])
predict


# In[49]:


predict = model_eingenfaces.predict(data_test[7])
predict


# In[50]:


# training model with classifier Fisherfaces
model_fisherfaces = cv2.face.FisherFaceRecognizer_create()
model_fisherfaces.train(data_train, people)


# In[51]:


plt.figure(figsize = (20, 10))

plt.subplot(121)
plt.title("People " + str(people_test[13]))
plt.imshow(data_test[13], cmap = "gray")

plt.subplot(122)
plt.title("People " + str(people_test[19]))
plt.imshow(data_test[19], cmap = "gray")

plt.show()


# In[52]:


predict = model_fisherfaces.predict(data_test[13])
predict


# In[53]:


predict = model_fisherfaces.predict(data_test[19])
predict


# In[54]:


# training model with classifier LBPH
model_lbph = cv2.face.LBPHFaceRecognizer_create()
model_lbph.train(data_train, people)


# In[55]:


plt.figure(figsize = (20, 10))

plt.subplot(121)
plt.title("People " + str(people_test[21]))
plt.imshow(data_test[21], cmap = "gray")

plt.subplot(122)
plt.title("People " + str(people_test[27]))
plt.imshow(data_test[27], cmap = "gray")

plt.show()


# In[56]:


predict = model_lbph.predict(data_test[21])
predict


# In[57]:


predict = model_lbph.predict(data_test[27])
predict


# In[66]:


# accuracy classifier Eingenface
y_pred_eigenfaces = []

for item in data_test:
    y_pred_eigenfaces.append(model_eingenfaces.predict(item)[0])
    
accuracy_eigenfaces = accuracy_score(people_test, y_pred_eigenfaces)
accuracy_eigenfaces


# In[67]:


# accuracy classifier Fisherfaces
y_pred_fisherfaces = []

for item in data_test:
    y_pred_fisherfaces.append(model_fisherfaces.predict(item)[0])
    
accuracy_fisherfaces = accuracy_score(people_test, y_pred_fisherfaces)
accuracy_fisherfaces


# In[68]:


# accuracy classifier LBPH
y_pred_lbph = []

for item in data_test:
    y_pred_lbph.append(model_lbph.predict(item)[0])
    
accuracy_lbph = accuracy_score(people_test, y_pred_lbph)
accuracy_lbph


# In[ ]:




