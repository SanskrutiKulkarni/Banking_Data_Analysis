# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:45:19 2024

@author: DELL
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the specified location
file_path = r'E:\code\Finlatics project\DsResearch\Banking\banking_data.csv'
data = pd.read_csv(file_path)

# Now you can proceed with the analysis using the loaded dataset
# For example:
print(data.head())  # Display the first few rows of the dataset

# Distribution of age Q1
plt.figure(figsize=(10,6))
sns.histplot(data['age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Job distribution Q2
plt.figure(figsize=(10,6))
sns.countplot(y='job', data=data)
plt.title('Distribution of Job Types')
plt.xlabel('Count')
plt.ylabel('Job')
plt.show()

# Marital status distribution Q3
plt.figure(figsize=(8,6))
sns.countplot(x='marital', data=data)
plt.title('Marital Status Distribution')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()

# Education level distribution Q4
plt.figure(figsize=(10,6))
sns.countplot(y='education', data=data)
plt.title('Education Level Distribution')
plt.xlabel('Count')
plt.ylabel('Education Level')
plt.show()

# Proportion of clients with credit in default Q5
default_count = data['default'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(default_count, labels=default_count.index, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Clients with Credit in Default')
plt.show()

# Distribution of average yearly balance Q6
plt.figure(figsize=(10,6))
sns.histplot(data['balance'], bins=20, kde=True)
plt.title('Distribution of Average Yearly Balance')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.show()

# Count of clients with housing loans Q7
housing_count = data['housing'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(housing_count, labels=housing_count.index, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Clients with Housing Loans')
plt.show()

# Count of clients with personal loans Q8
loan_count = data['loan'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(loan_count, labels=loan_count.index, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Clients with Personal Loans')
plt.show()

# Communication types distribution Q9
plt.figure(figsize=(8,6))
sns.countplot(y='contact', data=data)
plt.title('Communication Types Distribution')
plt.xlabel('Count')
plt.ylabel('Contact Type')
plt.show()

# Distribution of the last contact day of the month Q10
plt.figure(figsize=(10,6))
sns.histplot(data['day'], bins=31, kde=True)
plt.title('Distribution of Last Contact Day of the Month')
plt.xlabel('Day of Month')
plt.ylabel('Frequency')
plt.show()

# Last contact month distribution Q11
plt.figure(figsize=(10,6))
sns.countplot(y='month', data=data, order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
plt.title('Distribution of Last Contact Month')
plt.xlabel('Count')
plt.ylabel('Month')
plt.show()

# Distribution of duration of the last contact Q12
plt.figure(figsize=(10,6))
sns.histplot(data['duration'], bins=20, kde=True)
plt.title('Distribution of Duration of Last Contact')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()

# Distribution of contacts performed during the campaign Q13
plt.figure(figsize=(10,6))
sns.histplot(data['campaign'], bins=20, kde=True)
plt.title('Distribution of Contacts Performed During the Campaign')
plt.xlabel('Number of Contacts')
plt.ylabel('Frequency')
plt.show()

# Distribution of days passed since last contact from a previous campaign Q14
plt.figure(figsize=(10,6))
sns.histplot(data['pdays'], bins=20, kde=True)
plt.title('Distribution of Days Passed Since Last Contact from Previous Campaign')
plt.xlabel('Number of Days')
plt.ylabel('Frequency')
plt.show()

# Distribution of contacts performed before the current campaign Q15
plt.figure(figsize=(10,6))
sns.histplot(data['previous'], bins=20, kde=True)
plt.title('Distribution of Contacts Performed Before Current Campaign')
plt.xlabel('Number of Contacts')
plt.ylabel('Frequency')
plt.show()

# Previous marketing campaign outcomes distribution Q16
plt.figure(figsize=(8,6))
sns.countplot(y='poutcome', data=data)
plt.title('Previous Marketing Campaign Outcomes Distribution')
plt.xlabel('Count')
plt.ylabel('Outcome')
plt.show()


# Distribution of clients who subscribed to a term deposit vs. those who did not Q17
plt.figure(figsize=(6,6))
sns.countplot(x='y', data=data)
plt.title('Subscription to Term Deposit')
plt.xlabel('Subscribed to Term Deposit')
plt.ylabel('Count')
plt.show()

# Correlation matrix Q18
# Drop non-numeric columns
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
corr_matrix = numeric_data.corr()

# Visualize the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
