import numpy as np

#include bias
X_train = np.array([[1,0,0],[1,1,0],[1,0,1],[1,1,1]])
y_train = np.array([[0],[0],[0],[1]])

class Perceptron:
  def __init__(self,X_train,y_train):
    self.X = X_train
    self.y = y_train
    #include bias weight
    self.W = np.array([[np.random.randn()],[np.random.randn()],[np.random.randn()]])
    self.alpha = 1e-1

  def getShape(self):
    print(self.W.shape)

  def sigmoid(self,z):
    return 1/(1+np.exp(-z))
  
  def sigmoid_prime(self,z):
    return np.exp(-z)/((1+np.exp(-z))**2)

  def forward_propagation(self,x):
    self.z1 = np.dot(x,self.W)
    yHat = self.sigmoid(self.z1)
    return yHat 

  def back_propagation(self):
    for _ in range(100):
      yHat = self.forward_propagation(self.X)

      #gradient of cross entry loss function for logistic classification
      dJdyHat = ((-1 * self.y) / yHat) + ((1- self.y)/(1-yHat))
      
      delta1 = np.multiply(dJdyHat,self.sigmoid_prime(self.z1))

      dJdW = np.dot(self.X.transpose(),delta1)

      self.W = self.W - self.alpha * dJdW

  def predict(self,x):
    yHat = self.forward_propagation(x)
    norm = []
    for p in yHat:
      if p >= .5:
        norm.append([1])
      else:
        norm.append([0])
    return norm

P = Perceptron(X_train,y_train)
P.back_propagation()

#manually included 1 as bias term in test
test = np.array([[1,1,1]])
#np.insert(test[0],0,1)

print(P.predict(test))

