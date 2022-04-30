import cv2
import numpy as np
import pandas as pnd
from Perceptron_2 import pnUp, pnDown, pnLeft, pnRight

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


df=pnd.DataFrame({'px':[],'py':[], 'x_':[],'y_':[]})

while True:
    rf, frame = cap.read()
    
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cols=[[(150,90,70), (180,255,255)],[(15,110,110), (40,255,255)],[(30,30,40), (80,255,255)],[(135,100,70), (165,255,255)]]
    pts=[]
    for i in cols:
        pts.append(list(findCenter (frame_HSV, i[0], i[1], frame)[0]))

    pts=np.asarray(pts)

   
    cdC=[0,0]
        
    for i in pts:
        cdC[0]+=i[0]
        cdC[1]+=i[1]
    cdC[0]/=len(pts)
    cdC[1]/=len(pts)

    relpts=pts[:]-cdC
        
    for pos, relpos in zip(pts,relpts):
        status=''
        vUp, vDown, vRight, vLeft=pnUp.predict(relpos), pnDown.predict(relpos), pnRight.predict(relpos), pnLeft.predict(relpos)        
        if vUp>0:
            status+='top '
        if vDown>0:
            status+='bottom '
        if vRight>0:
            status+='right '
        if vLeft>0:
            status+='left '
        cv2.putText(frame,status,(pos[0],pos[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    


    key = cv2.waitKey(1)
    if (key>=0):
        print(key)
    if key == 112:
        pl=min(pts[:,0])
        pr=max(pts[:,0])
        pu=min(pts[:,1])
        pd=max(pts[:,1])
           
        h=pd-pu
        w=pr-pl



        newPt=pnd.DataFrame({'px':pts[:,0],'py':pts[:,1],'x_':pts[:,0]/w,'y_':pts[:,1]/h})
        df=pnd.concat([df,newPt])
        
    
    
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
if (len(df)>3):
    df.to_csv('plgData.csv')
cv2.destroyAllWindows()
cap.release()

