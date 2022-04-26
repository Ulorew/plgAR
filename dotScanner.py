import cv2
import random as rd
import pandas as pd


cap = cv2.VideoCapture(0)

def findCenter (frame, color_low, color_high, out):
    color_mask = cv2.inRange(frame, color_low, color_high)

    cont, h = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        cont = sorted(cont, key=cv2.contourArea, reverse=True)[0]

    
        moments = cv2.moments(cont, 1)

        m00 = moments['m00']
        m01 = moments['m01']

        
        
        m10 = moments['m10']
        if m00 >0:
            x = int(m10/m00)
            y = int(m01/m00)

            cv2.circle(out, (x, y), 5, (120,0,60), -1)

            return (x, y), color_mask
        return (0,0), color_mask
    except IndexError:
        return (0,0), color_mask


df=pd.DataFrame({'p1x':[],'p1y':[],'p2x':[],'p2y':[],'p3x':[],'p3y':[],'p4x':[],'p4y':[]})

saveId=0

while True:
    rf, frame = cap.read()
    
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    cdR, mask = findCenter (frame_HSV, (170,110,70), (180,255,255), frame)

    cdY, mask = findCenter (frame_HSV, (15,110,110), (40,255,255), frame)

    cdG, mask = findCenter (frame_HSV, (40,100,50), (80,255,255), frame)

    cdP, mask = findCenter (frame_HSV, (135,100,70), (165,255,255), frame)

    key = cv2.waitKey(1)
    if (key>=0):
        print(key)
    if key == 112:
        newCol=pd.DataFrame({'p1x':[cdR[0]],'p1y':[cdR[1]],'p2x':[cdY[0]],'p2y':[cdY[1]],
                             'p3x':[cdG[0]],'p3y':[cdG[1]],'p4x':[cdP[0]],'p4y':[cdP[1]]})
        df=pd.concat([df,newCol])
        saveId+=1
    
    
    '''
    cv2.line(frame, cdY, cdR, (200,0,255), 1 )

    cv2.line(frame, cdG, cdR, (255,0,200), 1 )

    cv2.line(frame, cdY ,(cdY[0], cdG[1]), (255,0,200), 1 )
    cv2.line(frame, (cdY[0], cdG[1]), cdG, (200,0,255), 1 )
    '''

   
   # try:
    #    cv2.imshow('mask', mask)
    #except:
    #    pass
    cv2.imshow('main', frame)
    if key == 27:
        break
df.to_csv('plgData.csv')
cv2.destroyAllWindows()
cap.release()

