import pandas as pd 
from funcs import nullRemoval, scaling , dimensionalityReduction , ModelSelection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys 
arguments = sys.argv
import matplotlib.pyplot as plt 
import seaborn as sns 
# data = r'E:\Projects\Diwali sales analysis\Diwali Sales Data.csv'
data = arguments[1]
df = pd.read_csv(data, encoding= 'unicode_escape')




inpt = input('What type of model you are trying to build ? Supervised/unsupervised ')
if inpt == 'Supervised' :
    ls = []
    for i, j in zip(df.isna().sum().keys() ,df.isna().sum().values ):
        if j > 0 :
        # print(f'you have some missing values in column : {i} ')
            ls.append(i)
    print(f'our automatic cleaner detects some missing values in columns : {ls}')
    nullRemoval(df)
    

    # numericall check ---
    print('for applying the ml model your data should be in a numerical format !! ')
    print('We are getting the columns which have only numerical values in it ')
    ls1 = []
    for i , j in zip(df.dtypes.index ,df.dtypes.values):
        if j != 'object':
            ls1.append(i)
    print(f'our automatic cleaner detects columns which have numerical vlaues are : {ls1}')
    inp = input('Do you wanna remove columns other than this ? Y/n')
    notls = [i for i in df.columns if i not in ls1]
    if inp in('Y','y'):
        df.drop(notls , axis = 1 , inplace = True)
    else :
        print('Okaay lets move on to next step !--- *')


    print(f'Your final Dataframe : \n{df.head()}')
    print(f'Dimension of your data : {df.shape}')

    # Scaling the dataset
    scaling(df)
    ModelSelection(df)




    # df.dropna(inplace = True)
    # print('Null values get removed from your data !')

# plt.bar(x= df['Gender'].value_counts().index , height =df['Gender'].value_counts().values  )




pcommand = input('You wanna save this into a CSV file ? Y/n :  ')
if pcommand in ('Y' , 'y'):
    df.to_csv('Results.csv')
    print('Check the directory ! you get your file there !! ')
else :
    print('Okaay byee ')


