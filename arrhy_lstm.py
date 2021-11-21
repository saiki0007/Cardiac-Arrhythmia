
# coding: utf-8

# In[ ]:


#importing libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


#function for predicting testset accuracy
def accuracy(y,pred):
	count=0.0
	for i in range(0,y.shape[0]):
		if(y[i][0]==pred[i]):
			count=count+1
	return count*100/y.shape[0]

#importing the dataset
X=pd.read_csv("dataset_original_wcolns.csv")
X=X.iloc[:,:].values
X.shape
Y=pd.read_csv("target_output.csv")
Y=Y.iloc[:,:].values
#splitting the dataset in train and test datasets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.05, random_state=0)
#feature scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
#converting train data into 3-d Array for input to LSTM layer
trainX = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
testX = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
#checking its dimensions
trainX.ndim
#checking its shape
trainX.shape
#initialising the LSTM model
classifier= Sequential()
model = Sequential()
model.add(LSTM(300,input_shape=(1,279)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, Y_train, epochs=1000, batch_size=20, verbose=2)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
#checking the RootMeanSquareError
import math
trainScore = math.sqrt(mean_squared_error(Y_train, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Y_test, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
#predicting test sets results
Y_pred=model.predict(testX)
Y_test_op=((np.argmax(Y_pred,axis=1)+1))
np.reshape(Y_test_op,(Y_pred.shape[0],1))
print(accuracy(Y_test,Y_test_op))








