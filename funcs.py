import pandas as pd 
import time 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error
import seaborn as sns 
import matplotlib.pyplot as plt

# def nullRemoval(df) :
#     print(f'null values in your data : \n{df.isna().sum()}')
#     ans = input('Wanna remove all null value records ? Y/n :  ')
    
#     for i in range(1,10):
#         print('_' * i , end= ' ', flush = True)
#         time.sleep(0.10)
#     if ans in ('Y' , 'y'):
#         df.dropna(inplace = True)
#         print('\n')
#         print(df.isna().sum())
#     else : 
#         print('Removing the rows which are completely full of null values !')
#         df.dropna(how = 'all' , inplace = True )
#         print('\n')
#         print(df.isna().sum())



def nullRemoval(df):
    ans = input('Do you wanna remove them ? Y/n : ')
    if ans in ('Y' , 'y'):
        df.dropna(inplace = True)
        print(f'Removing Them -----~~*')
    else :
        print('Removing the rows and columns which are completely null ')
        df.dropna(how = 'all' , inplace = True , axis = 1)
        df.dropna(how = 'all' , inplace = True , axis = 0)
        print(f'Removing Them -----~~*')
    
def scaling(df):
    ans = input('Do you wanna scale your dataset ? Y/n')
    if ans in ('Y' , 'y'):
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        return df 
    else :
        print('Okaay moving to next step --* ')
    
def dimensionalityReduction(df):
    ans = input('do you wanna reduce the dimension of your data ? Y/n')
    if ans in('Y', 'y'):
        pca = PCA(n_components = 2)
        df = pca.fit_transform(df)
        df = pd.DataFrame(df)
        print(f'your data looks like : \n{df.head()}')
    else :
        print('Okaay moving to next step !-----**')



        
 

def ModelSelection(df):
    inpt = input('Tell me the type of your data means Regression or Classification')
    if inpt =='Regression':
        models = {
            'Linear Regression' : LinearRegression(),
            'Lasso Regression' : Lasso(),
            'Ridge Regression' : Ridge(),
            'RandomForestRegressor' : RandomForestRegressor(),
            'DecisionTreeRegressor' : DecisionTreeRegressor(),
            'SVR' : SVR()  
        }
        DependentVar = input('Enter your output feature name : ')
        print('Dividing your dataset in train and test split --- * ')
        x_train , x_test , y_train , y_test = train_test_split(df.drop(DependentVar, axis =1 ), df[DependentVar] , train_size=0.7 , random_state=42)
        for model_name , model in models.items():
            print(f'\nModel name : {model_name} --------------------** \n')
            model.fit(x_train , y_train)
            y_pred = model.predict(x_test)
            r2Score = r2_score(y_test  , y_pred)
            mse = mean_squared_error(y_test , y_pred)
            print(f'\nPredicted values : {y_pred}')
            print(f'Model evaluation')
            print(f'R2 score : {r2Score} \nMean_squared_error : {mse}')


       
        

    