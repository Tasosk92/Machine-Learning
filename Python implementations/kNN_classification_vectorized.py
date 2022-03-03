'''
k-Nearest Neighbors (vectorized) python implementation
'''

import numpy as np 
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time

### K-Nearest Neighbors Algorithm - Classification - Vectorised implementation


## Steps 1 : Calculate and Sort Distances 

def get_neighbors(X_train,X_test) :
    
    dist = np.sqrt(((X_train[:, :, None] - X_test[:, :, None].T) ** 2).sum(1))
    sorted_distance = np.argsort(dist, axis = 0)

    return sorted_distance

    
## Step 2: Get Nearest Neighbors and Make Predictions

def  predict(X_train,X_test, y_train, n_neighbors):
    
    sorted_distance = get_neighbors(X_train,X_test)
    y_pred = np.zeros(y_test.shape)
    
    for row in range(len(X_test)):
        votes = list(y_train[sorted_distance[:,row][:n_neighbors]])
        y_pred[row] = max(set(votes), key=votes.count)
    
    return y_pred
 
if __name__ == "__main__":
    
    X,y = datasets.make_classification(n_samples=2000, n_features=10,n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
    
    start = time.time()
    y_pred =  predict(X_train,X_test, y_train, n_neighbors=5)
    print(accuracy_score(y_test, y_pred))
    end = time.time()
    print("Time elapsed: ",end - start)
    
    
    #Compare with sklearn
    
    start = time.time()
    
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    end = time.time()
    print("Time elapsed: ",end - start)



