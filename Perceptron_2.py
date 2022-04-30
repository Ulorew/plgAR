import numpy as np
import os
from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron():
    
    def __init__(self,eta=0.01,n_iter=50, random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state

    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,
                            size=1+X.shape[1])
        self.errors_=[]

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors += int(update!=0.0)
            self.errors_.append(errors)
        return self

    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X)>=0.0,1,-1)


df=pd.read_csv('plgData.csv',header=0)
df.iloc[1:]=df.astype(float)
print(df.head(10))

trainSz=len(df)
#trainSz=int(len(df)*3/4)

X=df.iloc[1:trainSz,[1,2]].values

iters=10

pnUp=Perceptron(eta=0.1,n_iter=iters)
pnDown=Perceptron(eta=0.1,n_iter=iters)
pnLeft=Perceptron(eta=0.1,n_iter=iters)
pnRight=Perceptron(eta=0.1,n_iter=iters)

y=df.iloc[1:trainSz,[3,4]].values

yUp=np.where(y[:,1]>=0,1,-1)
yDown=np.where(y[:,1]<0,1,-1)
yRight=np.where(y[:,0]>=0,1,-1)
yLeft=np.where(y[:,0]<0,1,-1)

pnUp.fit(X,yUp)
pnDown.fit(X,yDown)
pnRight.fit(X,yRight)
pnLeft.fit(X,yLeft)



if False:
    XDownP=np.array([v.tolist() for i, v in zip(range(len(X)),X) if yDown[i]>=0])
    XDownN=np.array([v.tolist() for i, v in zip(range(len(X)),X) if yDown[i]<0])
    plt.scatter(XDownP[:,0],XDownP[:,1],color='red',marker='o',label='bottom thing')
    plt.scatter(XDownN[:,0],XDownN[:,1],color='blue',marker='x',label='other')
    plt.legend(loc='Downpers')
    plt.show()

    XRightP=np.array([v.tolist() for i, v in zip(range(len(X)),X) if yRight[i]>=0])
    XRightN=np.array([v.tolist() for i, v in zip(range(len(X)),X) if yRight[i]<0])
    plt.scatter(XRightP[:,0],XRightP[:,1],color='red',marker='o',label='right thing')
    plt.scatter(XRightN[:,0],XRightN[:,1],color='blue',marker='x',label='other')
    plt.legend(loc='rights')
    plt.show()


'''plt.plot(range(1,len(pnUp.errors_)+1),pnUp.errors_,marker='o')
plt.show()'''





         

