import numpy as n
import pandas as p
import matplotlib.pyplot as plt
class LinearRegression:
    
    def __init__(self,lr=0.01,iter=1000):
        self.lr=lr
        self.iter=iter
        self.weight=None
        self.bias=None
    
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weight=n.zeros(n_features)
        self.bias=0
        for _ in range(self.iter):
            y_pred=n.dot(X,self.weight)+self.bias
            dw=(1/n_samples)*n.dot(X.T,(y_pred-y))*2
            db=(1/n_samples)*n.sum(y_pred-y)
            self.weight-=self.lr*dw
            self.bias-=self.lr*db
    
    def predict(self,X):
        return n.dot(X,self.weight)+self.bias
    
    def meanSquareError(self,Y,Y_pred):
        return n.mean((Y-Y_pred)**2)
    

d_x=p.read_csv("Linear_X_Train.csv").values
d_y=p.read_csv("Linear_Y_Train.csv").values
d_y=d_y.reshape((3750,))
d_x_test=p.read_csv("Linear_X_Test.csv")
r=LinearRegression()
r.fit(d_x,d_y)
y_pred=r.predict(d_x_test)
plt.scatter(d_x,d_y)
plt.plot(d_x_test,y_pred,color="black")
plt.show()
print(y_pred.shape)
