import numpy as np
import os
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

pnUp=Perceptron(eta=0.1,n_iter=10)
pnDown=Perceptron(eta=0.1,n_iter=10)
pnLeft=Perceptron(eta=0.1,n_iter=10)
pnRight=Perceptron(eta=0.1,n_iter=10)

y=df.iloc[:trainSz,[3,4]].values

yUp=np.where(y[1]<0,1,-1)
yDown=np.where(y[1]>=0,1,-1)
yRight=np.where(y[0]>=0,1,-1)
yLeft=np.where(y[0]<0,1,-1)


pnUp.fit(X,yUp)
pnDown.fit(X,yDown)
pnRight.fit(X,yRight)
pnLeft.fit(X,yLeft)


'''
ppnY=Perceptron(eta=0.1,n_iter=10)
X=df.iloc[:trainSz,[1,2]].values
y=np.rint(df.iloc[:trainSz,[4]].values)

ppnY.fit(X,y)'''
y=(df.iloc[:trainSz,[3]].values)
#Xp=pd.DataFrame(list(map(np.ravel, [v for i, v in zip(range(len(X)),X) if True])))
Xp=np.array([v.tolist() for i, v in zip(range(len(X)),X) if y[i]>=0])
Xn=np.array([v.tolist() for i, v in zip(range(len(X)),X) if y[i]<0])

print('Xp and Xn:')
print(Xp)
print(Xn)

plt.scatter(Xp[:,0],Xp[:,1],color='red',marker='o',label='top thing')
plt.scatter(Xn[:,0],Xn[:,1],color='blue',marker='x',label='down thing')
plt.legend(loc='upper left')
plt.show()


plt.plot(range(1,len(ppnX.errors_)+1),ppnX.errors_,marker='o')
plt.show()

'''plt.plot(range(1,len(ppnY.errors_)+1),ppnY.errors_,marker='o')
plt.show()'''

'''y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
X=df.iloc[0:100,[0,2]].values
print(X)
print(y)

plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
plt.scatter(X[50:,0],X[50:,1],color='blue',marker='x',label='ne setosa')
plt.legend(loc='upper left')
plt.show()


ppn=Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.show()
'''
         

