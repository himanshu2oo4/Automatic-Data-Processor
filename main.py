import pandas as pd 
from funcs import nullRemoval
import sys 
arguments = sys.argv
import matplotlib.pyplot as plt 
import seaborn as sns 
# data = r'E:\Projects\Diwali sales analysis\Diwali Sales Data.csv'
data = arguments[1]
df = pd.read_csv(data, encoding= 'unicode_escape')




# while True :
#     print('\n\n')
#     print('----------- Enter your choice : ----------')
#     print('''
#     1. Remove the null values 
#     2. q for quit 
#     ''')
#     inp = input('Enter your choice : ')
#     match inp:
#         case '1':
#             nullRemoval(df)
#         case 'q' :
#             break 

ls = []
for i, j in zip(df.isna().sum().keys() ,df.isna().sum().values ):
    if j > 0 :
        # print(f'you have some missing values in column : {i} ')
        ls.append(i)
print(f'our automatic cleaner detects some missing values in columns : {ls}')
nullRemoval(df)
    # df.dropna(inplace = True)
    # print('Null values get removed from your data !')

# plt.bar(x= df['Gender'].value_counts().index , height =df['Gender'].value_counts().values  )


print(sns.countplot(x = df['Gender']))
plt.show()


print(f'Your final Dataframe : \n{df.head()}')
pcommand = input('You wanna save this into a CSV file ? Y/n :  ')
if pcommand in ('Y' , 'y'):
    df.to_csv('Results.csv')
    print('Check the directory ! you get your file there !! ')
else :
    pass 



