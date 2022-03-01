'''
k-Nearest Neighbors algorithm python implementation
'''

import numpy as np 
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

### K-Nearest Neighbors Algorithm - Classification

## Step 1 : Calculate Euclidean Distance

def euclidean_distance(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

# ## Step 2: Get Nearest Neighbors

def get_neighbors(X,test_point,n_neighbors) :
    distances,neighbors = [],[]
    for i,train_point in enumerate(X):
        dist = euclidean_distance(train_point, test_point)
        distances.append((i,dist))
    distances.sort(key=lambda tup:tup[1])
    neighbors = [tup[0] for tup in distances[:n_neighbors]]
    return neighbors
        
### Step 3: Make Predictions

def  predict(X, y, test_point, n_neighbors):
    neighbors = get_neighbors(X, test_point, n_neighbors)
    votes = [y[idx] for idx in neighbors]
    return max(set(votes), key=votes.count)
  
if __name__ == "__main__":
    
    X,y = datasets.make_classification(n_samples=200, n_features=4,n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
    
    predictions = []
    for test_point in X_test:
        predictions.append(predict(X_train,y_train,test_point,n_neighbors=5))
    
    target_names = ['class 0', 'class 1']
    print(classification_report(y_test, predictions,target_names=target_names))
