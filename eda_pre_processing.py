import numpy as np # just in case, I most probably wont use this
import matplotlib.pyplot as plt # to show plots
import seaborn as sns # for plotting
import pandas as pd # to read and manipulate the csv file
from sklearn.model_selection import train_test_split # to splitting data
import pickle

# reading the dataset from kaggle 
# kagggle link: https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset
df=pd.read_csv('Skillorbit_student_performance_prediction_dashboard/data/Student_performance_data _.csv')

# print relevent information of the dataset 
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# to check for duplicates, if the sum is zero, it means there are no duplicates
print(df.duplicated().sum()) 

# checking for null values
print(f'Nulls before{df.isnull().sum()}') 

# the dataset here has no duplicates or null/empty spots so no necessary actions to remove nulls/duplicates are not required
# to understand how other factor effect academic performance (GPA) we see the heatmap
# corr=df.corr(numeric_only=True)
# plt.figure(figsize=(10,10))
# sns.heatmap(corr,annot=False,cmap='coolwarm') # a heatmap for all columns (since all cols are numerical anyway).
# plt.show() # annot is set to false for less congestion in the figure

# checking for outliers
# for col in df.columns:
#     plt.figure(figsize=(10,10))
#     sns.boxplot(df[col])
#     plt.title(col)
#     plt.show()
# almost no outliers, dataset is clean. no need for capping,trimming etc.

# pre processing
# splitting the feature columns and the target column
x=df.drop(['GradeClass','GPA','StudentID'],axis=1) # reason for this will be given in the report
y1=df['GPA'] # i will be using RandomTreeregressor for this
x_train,x_test,y_train,y_test=train_test_split(x,y1,test_size=0.2,random_state=42)
# label encoding not required as all columns are numerical
# I am not going to normailze the data for speed and the RandomTreeRegressor is not effect by the scale of the cols as it's output is produced purely by comparision.

processed_dataset=(x_train,x_test,y_train,y_test)

with open('Skillorbit_student_performance_prediction_dashboard/model/processed_data.pkl','wb') as file:
    pickle.dump(processed_dataset,file)

