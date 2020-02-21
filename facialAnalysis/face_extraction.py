#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt

from scipy.spatial import distance as dist
from io import BytesIO
from IPython.display import clear_output, Image, display
from PIL import Image as Img


# In[2]:


img = cv2.imread("images/px-woman-smilings.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize = (20, 10))
plt.imshow(img)


# In[3]:


classifier_dlib_68_path = "classifiers/shape_predictor_68_face_landmarks.dat"
classifier_dlib = dlib.shape_predictor(classifier_dlib_68_path)
face_detector = dlib.get_frontal_face_detector()


# In[4]:


def mark_face(img):
    rectangle = face_detector(img, 1)
    
    if len(rectangle) == 0:
        return None
    
    for k, d in enumerate(rectangle):
        print("Face index " + str(k))
        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 0), 2)
        
    return img


# In[5]:


mark_img = img.copy()
mark_img = mark_face(mark_img)


# In[6]:


plt.figure(figsize = (20, 10))
plt.imshow(mark_img)


# In[7]:


def facial_landmarks_points(img):
    rectangle = face_detector(img, 1)
    
    if len(rectangle) == 0:
        return None
    
    landmarks = []
    
    for ret in rectangle:
        landmarks.append(np.matrix([[p.x, p.y] for p in classifier_dlib(img, ret).parts()]))
        
    return landmarks


# In[8]:


facial_landmarks = facial_landmarks_points(img)


# In[9]:


len(facial_landmarks)


# In[10]:


len(facial_landmarks[0])


# In[11]:


def mark_facial_landmarks(img, landmarks):
    for landmark in landmarks:
        for idx, point in enumerate(landmark):
            center = (point[0, 0], point[0, 1])
            cv2.circle(img, center, 3, (255, 255, 0), -1)
            cv2.putText(img, str(idx), center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
    return img


# In[12]:


mark_img = img.copy()
mark_img = mark_facial_landmarks(mark_img, facial_landmarks)


# In[13]:


plt.figure(figsize = (20, 10))
plt.imshow(mark_img)


# In[14]:


img_people = cv2.imread("images/px-man-happy.jpg")
img_people = cv2.cvtColor(img_people, cv2.COLOR_BGR2RGB)


# In[15]:


plt.figure(figsize=(20,10))
plt.imshow(img_people)


# In[43]:


mark_img = img_people.copy()
facial_landmarks = facial_landmarks_points(mark_img)
mark_img = mark_facial_landmarks(mark_img, facial_landmarks)


# In[17]:


plt.figure(figsize=(20,10))
plt.imshow(mark_img)


# In[ ]:





# In[18]:


# define landmarks
FACE = list(range(17, 68))
FACE_COMPLETE = list(range(0, 68))
MOUTH = list(range(48, 61))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 35))
JAW = list(range(0, 17))


# In[19]:


def eyes_reason_aspect(eye_points):
    a = dist.euclidean(eye_points[1], eye_points[5])
    b = dist.euclidean(eye_points[2], eye_points[4])
    c = dist.euclidean(eye_points[0], eye_points[3])
    
    reason_aspect = (a + b) / (2.0 * c)
    
    return reason_aspect


# In[20]:


def mark_landmarks_convex_hull(img, landmarks):
    rectangle = face_detector(img, 1)
    
    if len(rectangle) == 0:
        return None
    
    for idx, ret in enumerate(rectangle):
        landmark = landmarks[idx]
        
        points = cv2.convexHull(landmark[LEFT_EYE])
        cv2.drawContours(img, [points], 0, (0, 255, 0), 2)
        
        points = cv2.convexHull(landmark[RIGHT_EYE])
        cv2.drawContours(img, [points], 0, (0, 255, 0), 2)
        
    return img


# In[21]:


def mark_landmarks_convex_hull_mouth(img, landmarks):
    rectangle = face_detector(img, 1)
    
    if len(rectangle) == 0:
        return None
    
    for idx, ret in enumerate(rectangle):
        landmark = landmarks[idx]
        
        points = cv2.convexHull(landmark[MOUTH])
        cv2.drawContours(img, [points], 0, (0, 255, 0), 2)
        
    return img


# In[44]:


mark_img = img_people.copy()
mark_img = mark_landmarks_convex_hull(mark_img, facial_landmarks)


# In[45]:


plt.figure(figsize = (20, 10))
plt.imshow(mark_img)


# In[24]:


value_left_eye = eyes_reason_aspect(facial_landmarks[0][LEFT_EYE])
value_left_eye


# In[25]:


value_right_eye = eyes_reason_aspect(facial_landmarks[0][RIGHT_EYE])
value_right_eye


# In[26]:


people_serious_img = cv2.imread("images/px-man-serious.jpg")
people_serious_img = cv2.cvtColor(people_serious_img, cv2.COLOR_BGR2RGB)


# In[27]:


plt.figure(figsize = (20, 10))
plt.imshow(people_serious_img)


# In[28]:


facial_landmarks = facial_landmarks_points(people_serious_img)

mark_img = people_serious_img.copy()
mark_img = mark_landmarks_convex_hull(mark_img, facial_landmarks)


# In[29]:


plt.figure(figsize = (20, 10))
plt.imshow(mark_img)


# In[30]:


value_left_eye = eyes_reason_aspect(facial_landmarks[0][LEFT_EYE])
value_left_eye


# In[31]:


value_right_eye = eyes_reason_aspect(facial_landmarks[0][RIGHT_EYE])
value_right_eye


# In[32]:


def standardize_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (500,400))
    return frame


# In[33]:


def show_video(frame):
    img = Img.fromarray(frame, "RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    display(Image(data=buffer.getvalue()))
    clear_output(wait=True)


# In[34]:


try:
    video = cv2.VideoCapture("videos/expressoes.mov")
    while(True):
        capture_ok, frame = video.read()
        if capture_ok:
            frame = standardize_image(frame)
            show_video(frame)
except KeyboardInterrupt:
    video.release()
    print("interrupted")


# In[35]:


def mouth_reason_aspect(mouth_points):
    a = dist.euclidean(mouth_points[3], mouth_points[9])
    b = dist.euclidean(mouth_points[2], mouth_points[10])
    c = dist.euclidean(mouth_points[4], mouth_points[8])
    d = dist.euclidean(mouth_points[0], mouth_points[6])
    
    reason_aspect = (a + b + c) / (3.0 * d)
    
    return reason_aspect


# In[51]:


try:
    ar_max = 0
    video = cv2.VideoCapture("videos/bocejo.mov")
    while(True):
        capture_ok, frame = video.read()
        if capture_ok:
            frame = standardize_image(frame)
            facial_landmarks = facial_landmarks_points(frame)
            
            if facial_landmarks is not None:
                ar_mouth =  mouth_reason_aspect(facial_landmarks[0][MOUTH])
                ar_mouth = round(ar_mouth, 3)
                
                if ar_mouth > ar_max:
                    ar_max = ar_mouth
                
                info = "mouth " + str(ar_mouth) + " max " + str(ar_max)
                cv2.putText(frame, info, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                frame = mark_landmarks_convex_hull_mouth(frame, facial_landmarks)
                
            show_video(frame)
            
except KeyboardInterrupt:
    video.release()
    print("interrupted")


# In[50]:


try:
    min_left_eye = 1
    min_right_eye = 1
    video = cv2.VideoCapture("videos/olhos-fechados.mov")
    while(True):
        capture_ok, frame = video.read()
        if capture_ok:
            frame = standardize_image(frame)
            facial_landmarks = facial_landmarks_points(frame)
            
            if facial_landmarks is not None:
                ar_left_eye =  eyes_reason_aspect(facial_landmarks[0][LEFT_EYE])
                ar_right_eye =  eyes_reason_aspect(facial_landmarks[0][RIGHT_EYE])
                
                ar_left_eye = round(ar_left_eye, 3)
                ar_right_eye = round(ar_right_eye, 3)
                
                if ar_left_eye < min_left_eye:
                    min_left_eye = ar_left_eye
                    
                if ar_right_eye < min_right_eye:
                    min_right_eye = ar_right_eye
                    
                frame = mark_landmarks_convex_hull(frame, facial_landmarks)
                
                info_left_eye = "Left eye " + str(ar_left_eye) + " min " + str(min_left_eye)
                info_right_eye = "Right eye " + str(ar_right_eye) + " min " + str(min_right_eye)
                
                cv2.putText(frame, info_left_eye, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, info_right_eye, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
            show_video(frame)
            
except KeyboardInterrupt:
    video.release()
    print("interrupted")


# In[ ]:




