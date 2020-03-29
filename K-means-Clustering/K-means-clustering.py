import numpy as n
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    
    def __init__(self,k=5,n_iter=20):
        self.k=k
        self.n_iter=n_iter
        self.clusters=[[] for i in range(k)]
        self.centers=[]
    
    def predict(self,x,y):
        self.x=x
        self.y=y
        n_samples,n_features=x.shape
        self.centers=[10*(2*n.random.random(n_features)-1) for i in range(self.k)]
        for i in range(self.n_iter):
            self.clusters=self.getCluster()
            self.plot()
            self.centers=self.getNewCenter()
            self.plot()
    
    def distance(self,x1,x2):
        return n.sqrt(sum((x1-x2)**2))
    
    def getCluster(self):
        clusters=[[] for _ in range(self.k)]
        for i in range(self.x.shape[0]):
            d=[]
            for j in range(self.k):
                d.append(self.distance(self.x[i],self.centers[j]))
            index=n.argmin(d)
            clusters[index].append(self.x[i])
        return clusters
            
    def getNewCenter(self):
        centers=[]
        for i in range(self.k):
            mean=n.mean(self.clusters[i],axis=0)
            centers.append(mean)
        return centers
    
    def plot(self):
        center_x=[]
        center_y=[]
        for i in range(self.k):
            center_x.append(self.centers[i][0])
            center_y.append(self.centers[i][1])
        plt.scatter(self.x[:,0],self.x[:,1],c=y)
        plt.scatter(center_x,center_y,marker="*",color="black")
        plt.show()

X,y=make_blobs(n_samples=200,n_features=2,centers=5,random_state=10)
k=KMeans()
k.predict(X,y)
