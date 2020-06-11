# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:40:42 2019

@author: hp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data= pd.read_csv('train_.csv')
test_data = pd.read_csv('test_.csv')
y = train_data.iloc[:,9].values
train_data = train_data.drop(['Item_Outlet_Sales'],axis = 1)

data = train_data.append(test_data, ignore_index=True)

data = data.fillna(data.mean())

fat_content = data['Item_Fat_Content'].value_counts()
data['Item_Fat_Content'] = data['Item_Fat_Content'].str.replace("LF", "Low Fat")
data['Item_Fat_Content'] = data['Item_Fat_Content'].str.replace("low fat", "Low Fat")
data['Item_Fat_Content'] = data['Item_Fat_Content'].str.replace("reg", "Regular")

data = pd.get_dummies(data, columns=["Item_Fat_Content"]) 

mapping2 = {'Small': 1,'Medium': 2,'High': 3,'nan': 2}
data = data.replace({'Outlet_Size': mapping2})

mapping_tier = {'Tier 1': 1,'Tier 2': 2,'Tier 3': 3}
data = data.replace({'Outlet_Location_Type': mapping_tier})

outlet_type_count = data['Outlet_Type'].value_counts()
data = pd.get_dummies(data, columns=["Outlet_Type"]) 

data = data.drop(['ID'],axis = 1)

itemType = data['Item_Type'].value_counts()
from sklearn.preprocessing import LabelBinarizer
lb_style = LabelBinarizer()
lb = lb_style.fit_transform(data["Item_Type"])
data = data.drop(['Item_Type'],axis = 1)

data = data.fillna(data.median())

X = data.iloc[:,:].values
X = np.hstack((data,lb))

train = X[:8523,:]
test = X[8523:,:]

from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
train_pca = pca.fit_transform(train)
variance = pca.explained_variance_ratio_  
variance

from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
test_pca = pca.fit_transform(test)
variance_pca = pca.explained_variance_ratio_  


from sklearn.linear_model import LinearRegression
regressor_pca = LinearRegression()
regressor_pca.fit(train_pca, y)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train, y)

from sklearn.tree import DecisionTreeRegressor
regressorDT = DecisionTreeRegressor(random_state = 0)
regressorDT.fit(train, y)


from sklearn.ensemble import RandomForestRegressor
regressorRF = RandomForestRegressor(n_estimators = 10, random_state = 0,max_features = "log2",oob_score = True)
regressorRF.fit(train, y)

ypred =regressor.predict(test)
ypred_rf = regressorRF.predict(test)
ypred_dt = regressorDT.predict(test)
ypred_lr_pca = regressor_pca.predict(test_pca)

#these chnge made in VS Code 