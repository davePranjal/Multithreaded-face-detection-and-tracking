## Face detection code - taken from OpenCV documentation ##

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:\\Users\\pranj\\anaconda\\pkgs\\libopencv-4.2.0-py37_6\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

img = cv2.imread('speaker1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(tuple(faces))
for (x,y,w,h) in faces:
    print(tuple((x,y,w,h)))
    print((x,y,w,h))
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
#cv2.waitKey(0)
cv2.destroyAllWindows()
