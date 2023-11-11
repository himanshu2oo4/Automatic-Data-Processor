import pandas as pd 
import time 
import numpy as np 
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
        time.sleep(3)
    else :
        print('Removing the rows and columns which are completely null ')
        df.dropna(how = 'all' , inplace = True , axis = 1)
        df.dropna(how = 'all' , inplace = True , axis = 0)
        print(f'Removing Them -----~~*')
        time.sleep(3)
    

 
    