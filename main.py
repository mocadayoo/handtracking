import cv2 as cv

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

camera = cv.VideoCapture(2)

while(camera.isOpened()):
    ret, frame = camera.read()
    if ret == True:
        cv.imshow('webcam', frame)
        if cv.waitKey(1) == ord('l'): # leaveの頭文字 "l"
            break
    else:
        break

camera.release()
cv.destroyAllWindows()