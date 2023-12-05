import pandas as pd 
import time 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression , Lasso , Ridge , LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor , DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor , RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error , classification_report , accuracy_score, confusion_matrix , ConfusionMatrixDisplay
import warnings 
warnings.filterwarnings('ignore')
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sc
from kneed import KneeLocator


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
        dims = int(input('Enter the dimension you want your data into : '))
        pca = PCA(n_components = dims)
        df = pca.fit_transform(df)
        df = pd.DataFrame(df)
        print(f'your data looks like : \n{df.head()}')
    else :
        print('Okaay moving to next step !-----**')



        
 

def SupervisedModel(df):
    answers = {}
    inpt = input('Tell me the type of your data means Regression or Classification : ')
    if inpt in ('Regression' , 'regression'):
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
            answers[model_name] = r2Score
            print(f'\nPredicted values : {y_pred}')
            print(f'Model evaluation')
            print(f'R2 score : {r2Score} \nMean_squared_error : {mse}')

        print(f'Acc. to me your best model is : {max(answers.items(), key=lambda x: x[1])}')

    elif inpt in ('Classification', 'classification'):
        models = {
            'Logistic Regression' : LogisticRegression(),
            'Naive bayes' : GaussianNB(),
            'Linear SVC' : SVC(),
            'DecisionTreeClassifier' : DecisionTreeClassifier(),
            'RandomForestClassifier' : RandomForestClassifier()
        }
        DependentVar = input('Enter your output feature name : ')
        print('Dividing your dataset in train and test split --- * ')
        x_train , x_test , y_train , y_test = train_test_split(df.drop(DependentVar, axis =1 ), df[DependentVar] , train_size=0.7 , random_state=42)
        for model_name , model in models.items():
            print(f'\nModel name : {model_name} --------------------** \n')
            model.fit(x_train , y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test  , y_pred)
            
            answers[model_name] = accuracy
            print(f'\nPredicted values : {y_pred}')
            print(f'Model evaluation')
            print(f'Model Accuracy : {accuracy}')
            print(f'Classification Report : \n{classification_report(y_test , y_pred)}')
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(cmap='viridis', values_format='d')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.show()    



def unsupervisedModel(df):
    inpp = input('which type of model you are trying to build : (kmeans /hierarchical) : ')
    if inpp in ('kmeans' , 'Kmeans'):        
        x_train , x_test   = train_test_split(df, test_size = 0.3 , random_state = 42)
        wcss = []
        for k in range(1,11):
            kmeans = KMeans(n_clusters = k ,init = 'k-means++' )
            kmeans.fit(x_train)
            wcss.append(kmeans.inertia_)

        
        kl = KneeLocator(range(1,11) , wcss , curve = 'convex', direction ='decreasing')
        # kl.elbow  returns the number of clusters to be form !! 

        model = KMeans(n_clusters = kl.elbow,init = 'k-means++' )
        y_label = model.fit_predict(x_train)
        score = silhouette_score(x_train, kmeans.labels_)
        print(f'Predicted labels : \n{y_label}')
        plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=y_label)
        plt.title('KMeans Clustering')
        plt.ylabel(f'Accuracy (Silhouette score) : {score}')
        plt.show()

    elif inpp in ('Hierarchical' , 'hierarchical'):
        # bcoz our data is already scaled 
        x_train , x_test   = train_test_split(df, test_size = 0.3 , random_state = 42)
         
        plt.figure(figsize = (15,7))
        sc.dendrogram(sc.linkage(x_train, method = 'ward'))  # this line is fixed for creating a dendrogram
        # this ward argument tells us try to use the eucledian distance

        plt.ylabel('Eucledian Distance')
        plt.show()

        inpt = int(input('Enter the number of clusters you get from the dendrogram : '))
        cluster = AgglomerativeClustering(n_clusters = inpt , affinity = 'euclidean' , linkage = 'ward')
        cluster.fit(x_train)

        print(f'clusters labels : {cluster.labels_}')
        plt.scatter(x_train.iloc[:,0], x_train.iloc[:,1] , c = cluster.labels_)
        plt.title('Hierarchical clustering')
        plt.show()



        



                
def DatatypeHandler(df):
    colnames = {}
    inpp = input('Wanna change the datatype of any attribute ? Y/n ')
    if inpp in ('Y', 'y'):
        colnums = int(input('Enter the number of columns you wanna change name of (Numeric): '))
        for i in range(colnums):
            tp = input(f'Enter the name {i+1} : ')
            dt = input('Enter the datatype to change in : ')
            colnames[tp]= dt 

        for i,j in colnames.items():
            df[i]  = df[i].astype(j)
    
        print(f'Final results after changes in datatypes : ---- ** \n{df.dtypes}')
    else :
        print('Okaay moving to next step !!------------- ** ')


def colremover(df):
    agyaa = input('wanna see data feature names ? Y/n  : ')
    if agyaa in ('y', 'Y'):
        print(f'columns in your dataframe : \n{df.columns}\n')
        inp = input('do you wanna drop any column from these ? Y/n  : ')
        if inp in ('Y','y'):
            cols = []
            nums = int(input('Enter the number of columns to drop : '))
            for i in range(nums):
                name = input(f'Col {i+1} : ')
                cols.append(name)
            df.drop(cols,axis = 1 , inplace =True )
            print(f'\n{cols} are permanently removed from your data ! \n')
        else :
            print('\nOkaay no problem moving to next step -------------- **\n')



       
        

    