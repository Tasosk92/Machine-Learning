import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class LogReg():
      """
      Logistic Regression (Gradient Descent) implemented in Python 

      Χ: input data
      y: target value
      y_hat: prediction
      w: weights
      b: bias
      m: training examples
      n:number of features
      """
    
    def __init__(self, X, y, iters=1000, learning_rate=10):
        self.X = X
        self.y = y 
        self.m, self.n = self.X.shape
        self.learning_rate = learning_rate
        self.iters = iters
        
    def sigmoid(self, x):
        z = 1/(1+np.exp(-x))
        return z

    def cost_function(self, y, y_hat):
        cost = -np.mean((y*np.log(y_hat))-((1-y)*np.log(1-y_hat)))
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

        ovr_cost = []
        
        for iteration in range(self.iters):
            
            #calculate predictions
            y_hat = self.sigmoid(np.dot(self.X,self.w) + self.b)
            
            #calculate gradients
            dw,db = self.grads(self.X, self.y, y_hat)
           
            #update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            #calculate cost
            cost = self.cost_function(self.y, self.sigmoid(np.dot(self.X, self.w) + self.b)) 
            
            ovr_cost.append(round(cost,5))
          
        return ovr_cost
    
    def predict(self,X):
        
        #normalize X
        X = X / np.linalg.norm(X)
 
        preds = self.sigmoid(np.dot(X,self.w) + self.b)
        
        return [1 if pred>0.5 else 0 for pred in preds]

    


if __name__ == '__main__':
    
    X, y = make_classification(n_samples=1000,n_features=5, n_redundant=1, 
                           n_informative=2, random_state=7)
    
    X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=7)

    LR = LogReg(X_train, y_train)
    LR.fit()
    y_pred = LR.predict(X_test)   
    accuracy = np.mean(y_test==y_pred)
    
    print("Accuracy: {:.2%}".format(accuracy))
