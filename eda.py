import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# reading the dataset from kaggle 
# kagggle link: https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset
df=pd.read_csv('Skillorbit_student_performance_prediction_dashboard/data/Student_performance_data _.csv')

# print relevent information of the dataset 
print(df.head())
print(df.shape)
df.info()
df.describe()
columns=df.columns

# to check for duplicates, if the sum is zero, it means there are no duplicates
print(df.duplicated().sum()) 

# checking for null values
print(f'Nulls before{df.isnull().sum()}') 

# the dataset here has no duplicates or null/empty spots so no necessary actions to remove nulls/duplicates are not required
# to understand how other factor effect academic performance (GPA) we check the heatmap and remove those factors with no correlation with GPA
corr=df.corr(numeric_only=True)
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=False,cmap="coolwarm") # a heatmap for all columns (since all cols are numerical anyway).
plt.show() # annot is set to false for less congestion in the figure

# the columns which are uncorrelated to GPA and will be removed
# Volunteering
# StudentID
# Age
# Gender
# Ethnicity

df.drop(columns=['Volunteering','StudentID','Age','Gender','Ethnicity'], inplace=True)

# checking for outliers
for col in columns:
    plt.figure(figsize=(10,10))
    sns.boxplot(df[col])
    plt.title(col)
    plt.show()
# almost no outliers, dataset is clean. no need for capping,trimming etc.


