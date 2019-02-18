import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn import cross_validation, neighbors

data_train=pd.read_csv("train_upd.csv")
#data_test=pd.read_csv("test_upd.csv")

X=np.array(data_train.drop(data_train.columns[[0,2,3,-2,-1]],axis=1))
Y=np.array(data_train[data_train.columns[-1]]).reshape((data_train.shape[0]),1)

#X_test=np.array(data_test.drop(data_test.columns[[0,2,3,-1]],axis=1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, np.ravel(Y_train,order='C')) 
gnb_predictions = gnb.predict(X_test) 
  
# accuracy on X_test 
accuracy = gnb.score(X_test, Y_test) 
print (accuracy) 
  
# creating a confusion matrix 
cm = confusion_matrix(Y_test, gnb_predictions) 

