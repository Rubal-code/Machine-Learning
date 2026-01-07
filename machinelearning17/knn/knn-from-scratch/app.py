from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans


centroids=[(-5,-5),(5,5)]
cluster_std=[1,1]

X,y=make_blobs(n_samples=100,cluster_std=cluster_std,centers=centroids,n_features=2,random_state=5)

km=KMeans(2,100)
km.fit_predict(X)

plt.scatter(X[:,0],X[:,1])
plt.show()