import cv2
import random

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Now i am going to capture video to do real time Face detection

webcam = cv2.VideoCapture(0)


while True:
    frame_read , frame  = webcam.read()

    video = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces= classifier.detectMultiScale(video)
    
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        b = random.randint(0,256)
        g = random.randint(0,256)
        r = random.randint(0,256)
        cv2.rectangle(frame, (x,y) , (x+w,y+h ) , (b,g,r), 2)
    
    cv2.imshow('window bar',frame)

    
    k = cv2.waitKey(1)
    # ascii of q = 113 and Q = 81
    if k == 81 or k == 113:
        break

webcam.release()

print("Code completed Successfully ")
