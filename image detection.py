import cv2
import time
import random

# Image capture

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# reading the imagie
img = cv2.imread('i.png')

# showing the image
#cv2.imshow('shaswat face detection ',img)


# making image to gray scale as black and white
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# cv2.imshow('shaswat face detection ',grayscaled_img)
# cv2.waitKey()

# detecting the image
# return top left and bottom right points
faces = classifier.detectMultiScale(grayscaled_img)


print(faces)
# faces[0][0] = x faces[0][0] = y faces[0][0] = w faces[0][0] = h
# rectangle (image , (x,y),(w+x,h+y),color,thickness)
# cv2.rectangle(img , (87,114),(361+87,361+114) ,(255,0,0) , 2)

#[(x ,y,w,h)] = faces
for (x,y,w,h) in faces:
    b = random.randint(0,256)
    g = random.randint(0,256)
    r = random.randint(0,256)
    cv2.rectangle(img, (x,y) , (x+w,y+h ) , (b,g,r), 2)

cv2.imshow('shaswat face detection ',img)

cv2.waitKey()
