'''
k-Means python implementation
'''

import pandas as pd
import numpy as np 
import random as rd
import matplotlib.pyplot as plt
from sklearn import datasets 
from collections import defaultdict
import seaborn as sns


def euclidean_distance(center,point):
    dist = np.linalg.norm(center-point)
    return dist

### K-Means Algorithm

def kmeans(data,k,iters) :
    
    ##Step 1: Choose number K of clusters 
    K = k
    
    ##Step 2: Select K random points from the data as centroids
    
    random_indices = np.random.choice(len(data), size=K, replace=False)
    centroids = data[random_indices,:]
    
    ##Step 3: Assign all points to the closest cluster centroid  
    
    counter_1 = 0 #count number of iters where centroids don't change
    iteration = 0
    while True:
        
        # for every point calculate distance from every centroid
        distances = defaultdict(dict)
        
        for i,center in enumerate(centroids):
            for j,point in enumerate(data):
                distances[j][i] = euclidean_distance(center,point)
        
        # assign every point to the closest centroid
        points = {}
        for i in range(len(data)):
            points[i] = list({k: v for k, v in sorted(distances[i].items(), key=lambda item: item[1])}.keys())[0]
        
        # group points of every centroid  
        cluster_points = {}
        for i in range(K):
             cluster_points[i] = list({k for (k,v) in points.items() if v==i})        
        
        ##Step 4: Recompute the centroids of newly transformed clusters as the mean of their points
        centroids_cached = [tuple(c) for c in centroids]
        centroids = []
        for i in range(K):
            centroids.append(np.array([data[i] for i in cluster_points[i]]).transpose().mean(axis=1))

        ##Step 5: Repeat steps 3,4 until a stopping criterion is met (max iterations or centroids remain the same for 3 iterations)
        
        if ([tuple(c) for c in centroids] == centroids_cached and counter_1==3):
            break
            return centroids,cluster_points
        elif ([tuple(c) for c in centroids] == centroids_cached and counter_1<3):
            counter_1 +=1
            
        if iteration == iters:
            break
        else:
            iteration += 1
            
    points = np.array([tup[1] for tup in points.items()]) #return cluster indices per point

    return centroids, cluster_points, points

        
### Find K using Elbow Method

def inertia (centroids,cluster_points,data):
    cost=0
    for i,centroid in enumerate(centroids):
        for point in cluster_points[i]:
            cost += euclidean_distance(centroid,data[point])
    return cost

def elbow_method(data,k):
    
    cost_list = []
    for k in range(1, k+1):
      
        centroids, cluster_points,points = kmeans(data, k,iters = 20)
        cost = inertia(centroids, cluster_points,data)
        cost_list.append(cost)
        
    sns.lineplot(x=range(1,k+1), y=cost_list, marker='o')
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.show()
  
if __name__ == "__main__":
    
    X, y = datasets.make_blobs(n_samples=100, centers=3, n_features=3,random_state=0)
    K=3
    iters = 20
    centroids, cluster_points,points = kmeans(X,K,iters)
