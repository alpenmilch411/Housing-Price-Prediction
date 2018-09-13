
# coding: utf-8

# # King County, Washington State, USA - Housing Prices

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import optimize
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import optimize
from matplotlib import pyplot
import sys
from tqdm import tqdm as tqdm
from matplotlib.pyplot import figure

from sklearn.preprocessing import PolynomialFeatures



# In[3]:


data = pd.read_csv("Data/kc_house_data.csv")
plt.figure()
data.hist(figsize=(20, 15), bins=50)
#scatter_matrix(data)
plt.show()


# # Data Preparation

# In[4]:


data.info()


# In[5]:


def data_preprocessing(data):
    #Returns processed data in numpy matrix
    #Drops ID field and converts date to year only
    data_w = data
    data_w = data.drop(columns=['id'])
    data_w["date"] = pd.to_datetime(data_w["date"]).dt.year
    
    #move price to the end of df
    price = data_w.pop('price')
    data_w['price']=price
    return data_w.values


# In[6]:


#Split data into train and test set
data_X = data_preprocessing(data)
X, X_test, y, y_test = train_test_split(data_X[:, :18], data_X[:, 19], test_size=0.20, random_state=23)

#Set aside test set
test_set = np.concatenate([X_test, y_test[:, None]], axis=1)
np.savetxt("Data/test_data.txt", test_set, fmt='%s')

#Split data into train and cross validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=23)


# # Linear Regression Model

# In[7]:


# Cost Function
def linearRegCostFunction(X, y, theta, lambda_):
    
    # Initialize some useful values
    m, n = X.shape # X matrix shape
    
    
    #Add intercept
    #print(np.ones(m).reshape(m,1).shape, X)
    X = np.concatenate([np.ones(m).reshape(m,1), X], axis=1)
    
    #Compute h
    h = X @ theta
    
    #Regularized Cost Function
    J = np.sum((h - y)**2)/(2*m)
    reg_term = (lambda_/(2*m)) * np.sum(theta[1:]**2)
    J = J + reg_term
    
    #Gradient Computation
    #Simple Gradient without reg for bias
    grad = (1/m) * ((h - y) @ X)
    
    # Compute gradient with reg for non-bias
    grad[1:] = grad[1:] + (lambda_/m)*theta[1:]

    return J, grad

def trainLinearReg(linearRegCostFunction, X, y, lambda_=0.0, maxiter=200):
    # Useful values
    m, n = X.shape # X matrix shape
    # Initialize Theta
    initial_theta = np.zeros(n+1)

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)

    # Now, costFunction is a function that takes in only one argument
    options = {'maxiter': maxiter}

    # Minimize using scipy
    res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
    
    return res.x

def learningCurve(X_train, y_train, X_val, y_val, lambda_=0):
    # Number of training examples
    m = y_train.size

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val   = np.zeros(m)

    #Compute error_train
    for i in tqdm(range(1, m+1)):
        theta_train = trainLinearReg(linearRegCostFunction, X_train[:i,:], y_train[:i], lambda_)
        
        error_train[i-1], _ = linearRegCostFunction(X_train[:i, :], y_train[:i], theta_train, 0)
        error_val[i-1], _ = linearRegCostFunction(X_val, y_val, theta_train, 0)
        
    return error_train, error_val

def featureNormalize(X):

    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm, mu, sigma


# In[8]:


## testing cost function
#test theta with ones
theta_test = np.ones(18+1)
lambda_ = 0
# Compute cost with test theta
J, grad = linearRegCostFunction(X_train, y_train, theta_test, lambda_)
J, grad


# In[9]:


#Train Model
res = trainLinearReg(linearRegCostFunction, X_train, y_train, lambda_=0.0, maxiter=200)
trained_theta = res


# In[ ]:


#Compute Training and Validation error for increasing number of training data used
error_train, error_val = learningCurve(X_train, y_train, X_val, y_val, lambda_=0)


# # ML Diagnostics
# ## Learning Curve

# In[86]:


#Plot Learning Curves
m = y_train.size

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(10):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))


figure(num=1, figsize=(20, 5))
pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=0.6)
pyplot.title('Learning curve for linear regression')
pyplot.legend(['Train', 'Cross Validation'])
pyplot.xlabel('Number of training examples')
pyplot.ylabel('Error')
pyplot.axis([0, m, 0, 161361944543])

pyplot.show()

    


# ## Learning Curve - Interpretation
# 
# - Graph Properties
#     - Low training set size: J_CV is high & J_train is low
#     - Large training set size: J_CV ≃ J_train
#     
# -> Model does not seem to suffer from high variance but from high bias
# 
# Steps to address high bias situation:
# - Adding features: <span style="color:orange">Not possible, look into feauture engineering</span> 
# - Adding polynomial features: <span style="color:green">Next Step</span> 
# - After polynomials: determine best lambda: <span style="color:green">Next Step</span> 

# # Adding Polynomial Features

# In[10]:


#Polynomial dimensions
p = 2

#Add polynomial features
poly = PolynomialFeatures(p, include_bias=False)

X_test_p = poly.fit_transform(X_test)
X_train_p = poly.fit_transform(X_train)
X_val_p = poly.fit_transform(X_val)

#Normalize X_test_p and extract sigma & mu
X_train_p, mu, sigma = featureNormalize(X_train_p)

#Normalize X_test_p & X_val_p using sigma & mu
X_test_p -= mu
X_test_p /= sigma
X_val_p -= mu
X_val_p /= sigma


# # Plot learning curve

# In[ ]:


#Compute Training and Validation error for increasing number of training data used
error_train, error_val = learningCurve(X_train_p, y_train, X_val_p, y_val, lambda_=0)


# In[ ]:


#Plot Learning Curves
m = y_train.size

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(10):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))


figure(num=1, figsize=(20, 5))
pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=0.6)
pyplot.title('Learning curve for linear regression')
pyplot.legend(['Train', 'Cross Validation'])
pyplot.xlabel('Number of training examples')
pyplot.ylabel('Error')
pyplot.axis([0, m, 0, 161361944543])

pyplot.show()

