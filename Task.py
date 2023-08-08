import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

titanic_data = sns.load_dataset('titanic')
print("dataset Information")
print(titanic_data.info())
print(titanic_data.describe())
# Missing Values
print("\n Missing value counts :")
print(titanic_data.isnull().sum())
# data cleaning
titanic_data['age'].fillna(titanic_data['age'].median(), inplace=True)
titanic_data['embarked'].fillna(titanic_data['embarked'].mode()[0], inplace=True)
# EDA 
plt.figure(figsize=(8,6))
sns.countplot(x='class',data=titanic_data)
plt.show()
# age distribution 
plt.figure(figsize=(8,6))
sns.histplot(x='age',data=titanic_data,bins=20,kde=False)
plt.title('Age Distribution of Passengers')
plt.show()
# heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = titanic_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()