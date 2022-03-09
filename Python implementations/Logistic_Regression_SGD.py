"""
Logistic Regression (Stochastic Gradient Descent) implemented in Python 
 
Χ: input data
y: target value
y_hat: prediction
w: weights
b: bias


"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class LogReg():
    
    def __init__(self,X,y,iters=1000, batch_size=32, learning_rate=10):
        self.X = X
        self.y = y 
        self.m,self.n = self.X.shape # m: training examples, n:number of features
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iters = iters
        
    def sigmoid(self, x):
        z = 1/(1+np.exp(-x))
        return z

    def cost_function(self, y, y_hat):
        cost = (np.dot((-y.T), np.log(y_hat)) - np.dot((1-y).T, np.log(1-y_hat))) / self.m
        return cost
    
    def grads(self, X, y, y_hat):

        # Gradient of weights
        dw = (1/self.m)*np.dot(X.T, (y_hat - y))
        
        # Gradient of bias
        db = (1/self.m)*np.sum((y_hat - y)) 
        
        return dw, db
        
    def fit(self):
        
        #initialize weights and bias with zero
        self.w = np.zeros((self.n,1)) 
        self.b = 0

        #reshape y
        self.y = self.y.reshape(self.m,1) 

        #normalize X
        self.X = self.X / np.linalg.norm(self.X) 

        #split data in batches
        X_batch = np.array_split(self.X, self.batch_size)
        y_batch = np.array_split(self.y, self.batch_size)

        losses = []
        
        for epoch in range(self.iters):
            for i,x_batch in enumerate(X_batch):
                
                y_hat = self.sigmoid(np.dot(x_batch,self.w) + self.b)
                
                dw,db = self.grads(x_batch, y_batch[i], y_hat)
               
                #update parameters
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
                
            cost = self.cost_function(self.y, self.sigmoid(np.dot(self.X, self.w) + self.b))
            losses.append(cost) 
        return losses
    
    def predict(self,X):
        
        #normalize X
        X = X / np.linalg.norm(X)
 
        preds = self.sigmoid(np.dot(X,self.w) + self.b)
        
        predictions = [1 if pred>0.5 else 0 for pred in preds]
        
        return predictions
    


if __name__ == '__main__':
    
    X, y = make_classification(n_samples=10000,n_features=5, n_redundant=0, 
                           n_informative=2, random_state=1, 
                           n_clusters_per_class=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.3, random_state=7)

    LR = LogReg(X_train, y_train)
    LR.fit()
    y_pred = LR.predict(X_test)
    print(classification_report(y_test,y_pred))
    
