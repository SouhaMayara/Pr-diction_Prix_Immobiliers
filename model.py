# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# Importing the dataset
dataset = pd.read_csv('Data - Test.csv',delimiter=';')


X = dataset.iloc[:,2:-1].values
y = dataset.iloc[:,-1].values


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =80, random_state = 1000)
regressor.fit(X, y)
#regressor.score(X, y)

y_pred = regressor.predict(X)
print(regressor.score(X, y))
#print("y_pred")
#print(y_pred)
#print("y")
#print(y)


#Saving model
pickle.dump(regressor, open('model.pkl','wb'))

#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
