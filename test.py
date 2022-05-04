import cv2
import numpy as np
import pandas as pnd

cap = cv2.VideoCapture(0)


mouseX = 0
mouseY = 0


capCol=(141, 143, 143)

def onEvent(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        global capCol
        a=sqrSz
        capCol=cv2.mean(frame[y-a:y+a,x-a:x+a])
        
        print(capCol)


rf, frame = cap.read()
cv2.imshow("main", frame)
cv2.setMouseCallback("main", onEvent)

sqrSz=10

while True:
    
    rf, frame = cap.read()
    
    key = cv2.waitKey(1)
    if key >= 0:
        print(key)

    cv2.rectangle(frame,(mouseX-sqrSz, mouseY-sqrSz),(mouseX+sqrSz, mouseY+sqrSz),capCol,5)
    #cv2.circle(frame, (mouseX-sqrSz, mouseY-sqrSz), 25, capCol, -1)
    cv2.imshow("main", frame)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
