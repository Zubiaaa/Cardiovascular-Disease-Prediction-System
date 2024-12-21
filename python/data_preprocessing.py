#!/usr/bin/env python
# coding: utf-8

# # Data Pre-Processing

# In[1]:


### Importing relevant libraries ###

import pandas as pd
import numpy as np
from statistics import mode
import scipy
from scipy.stats import skew

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot


# In[2]:


### Importing Heart Disease dataset ###

# Fill in  '?' value with NaN, as this is a value not provided
missing_values = ["?"]

# Importing dataset and converting '?' values to NAN
dataset = pd.read_csv('original_heart_disease_dataset.csv', na_values = missing_values)
dataset.head()


# ### a) Exploratory data analysis (EDA)

# In[3]:


### Printing information regarding the heart disease dataset ###

dataset.info()


# In[4]:


### Returning dimensionality of the Dataset ###

dataset.shape


# In[5]:


### Function to plot the percentage of missing values ###

# This will plot only the columns that have missing values.

def plot_null(df: pd.DataFrame):

    # Box plot have been selected for this purpose.
    if df.isnull().sum().sum() != 0:
        null_df = (df.isnull().sum() / len(df)) * 100
        null_df = null_df.drop(null_df[null_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' :null_df})
        plots =  missing_data.plot(kind = "bar", color = 'darkslategrey', figsize=(8,8))

        #Annotate text in bar plot
        for bar in plots.patches:
            plots.annotate('{:,.1f}'.format(bar.get_height())+ "%",
                           (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           ha='center', va='center',
                           size=10, xytext=(0, 8),
                           textcoords='offset points')

        plt.show()

    else:
        print('No NAs found')

plot_null(dataset)


# In[6]:


### Correlation Heat map of our original dataset ###

# a two dimensional plot of the amount of correlation (measure of dependence) between variables.

def plot_corrheatmap(df: pd.DataFrame):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap='flare', annot=True)


plot_corrheatmap(dataset)


# In[7]:


### Counting values in the target variable ###

# drawing count plot
sns.countplot(data= dataset, x='goal')
plt.title('Count of target\n')


# ### b) Cleaning Columns

# In[8]:


# There are no null values found in age, sex, cp, and goal(target) columns.


# In[9]:


# Rest of the columns should be cleaned


# In[10]:


### Displaying basic statistical details (description about the dataset) ###

dataset.describe()


# #### Age Column

# In[11]:


### Displaying a histogram (distribution) for age column ###

def plot_age(df: pd.DataFrame):
    # Creating a list of age sorted in ascending order.
    age = df["age"].tolist()

    # Creating a dataframe of sorted age
    sorted_age = df[["age"]].sort_values(by=['age'])

    ### Plotting a histogram to represent the distribution of age column:

    # Calculating median for the age column
    median_age = sorted_age["age"].median()

    plt.hist(age, bins=10, edgecolor='black')
    plt.axvline(median_age, color='red', label="Median Age")
    plt.xlabel("Age")
    plt.ylabel("Age Disribution")
    plt.title("Age Distribution Histogram")
    plt.legend()
    plt.show()


plot_age(dataset)


# In[12]:


# Calculate the skewness 
print(skew(dataset['age']))


# #### Sex Column

# In[13]:


### Checking values in "sex" column ###

# Sex column have two values - 0 and 1 (binary column).
print(dataset["sex"].value_counts())


# #### CP Column

# In[14]:


### Checking values in "cp" column ###

# "cp" column have four values - 1, 2, 3 and 4 (categorical column).

print(dataset["cp"].value_counts())


# #### Trestbps Column

# In[15]:


### Checking values in "trestbps" column ###

# Problem - the value of trestbps cannot be 0, as resting bp cannot be 0.

print(dataset["trestbps"].value_counts().index.tolist())


# In[16]:


### Checking how many values are "0" ###

missing = 0
for value in (dataset["trestbps"].tolist()):
    if value == 0.0:
        missing += 1

print(missing)


# In[17]:


# Only one value is 0


# In[18]:


### Checking percentage of missing values if 0 gets replaced with NAN ###

null_check_df = dataset.copy(deep=True)
null_check_df.trestbps.replace(to_replace = 0.0, value=np.nan, inplace = True)
plot_null(null_check_df)


# In[19]:


### Replacing 0 with NAN (missing value) in the original dataset ###

# As resting blood pressure cannot be 0,
# it means it is a missing value / value not provided / missing due to error.
# Hence substitute it with the NAN value

dataset.trestbps.replace(to_replace = 0.0, value=np.nan, inplace = True)
print(dataset["trestbps"].isnull().sum())


# ##### Checking data distribution

# In[20]:


### Checking how data is distributed ###
# to decide the best method for imputation (mean or median)

# Plotting histogram plot
sns.histplot(dataset['trestbps'], kde=True, color='yellowgreen')
# Removing color of the edges of histogram bars
plt.rcParams['patch.edgecolor'] = 'none'


# Plotting mean, median and mode
plt.axvline(dataset['trestbps'].mean(), color='blue', linestyle='dashed', linewidth=1, label="Mean: " + str(round(dataset['trestbps'].mean(), 2)))
plt.axvline(dataset['trestbps'].median(), color='red', linestyle='dashed', linewidth=1, label="Median: " + str(round(dataset['trestbps'].median(), 2)))
plt.axvline(dataset['trestbps'].mode()[0], color='black', linestyle='dashed', linewidth=1, label="Mode: " + str(round(dataset['trestbps'].mode()[0], 2)))
# Displaying legend
plt.legend(loc='upper right')


# In[21]:


# Calculate the skewness 
print(skew(dataset['trestbps'], nan_policy='omit'))


# The data in this column is a positively skewed distribution. Hence Median Imputation method will be suitable

# ##### Median Imputation

# In[22]:


### Replacing missing values with median ###

# Trestbps column contains 6.4% missing values, as shown in the visual above.
# After converting 0 -> null values, missing values increased to 6.5%.
# Filling missing values with the median value, as the column is continuous

dataset['trestbps'].fillna(value = round(dataset['trestbps'].median(), 1), inplace=True)
dataset["trestbps"].astype(int)


# In[23]:


### Checking if any missing value is left ###

se = dataset["trestbps"].tolist()
for i in se:
    if i == 0 or i == np.nan:
        p='true'
    else:
        p='false'
print(p)


# In[24]:


# All missing value are imputed in Trestbps column


# #### Chol Column

# In[25]:


### Checking values in "chol" column ###

# Problem - the value of chol cannot be 0, as serum cholesterol cannot be 0

print(dataset["chol"].value_counts().index.tolist())


# In[26]:


### Checking how many values are "0" ###

missing = 0
for value in (dataset["chol"].tolist()):
    if value == 0.0:
        missing += 1

print(missing)


# In[27]:


### Checking percentage of missing values if 0 gets replaced with NAN ###

null_check_df.chol.replace(to_replace = 0.0, value=np.nan, inplace = True)
plot_null(null_check_df)


# In[28]:


### Replacing 0 with NAN (missing value) in the original dataset ###

# As serum cholesterol cannot be 0,
# it means it is a missing value / value not provided / missing due to error.
# Hence substitute it with the NAN value

null_check_df = dataset.chol.replace(to_replace=0.0, value=np.nan, inplace=True)
print(dataset["chol"].isnull().sum())


# ##### Checking data distribution

# In[29]:


### Checking how data is distributed ###
# to decide the best method for imputation (mean or median)

# Plotting histogram plot
sns.histplot(dataset['chol'], kde=True, color='yellowgreen')
# Removing color of the edges of histogram bars
plt.rcParams['patch.edgecolor'] = 'none'
# Plotting mean, median and mode
plt.axvline(dataset['chol'].mean(), color='blue', linestyle='dashed', linewidth=1, label="Mean: " + str(round(dataset['chol'].mean(), 2)))
plt.axvline(dataset['chol'].median(), color='red', linestyle='dashed', linewidth=1, label="Median: " + str(round(dataset['chol'].median(), 2)))
plt.axvline(dataset['chol'].mode()[0], color='black', linestyle='dashed', linewidth=1, label="Mode: " + str(round(dataset['chol'].mode()[0], 2)))
# Displaying legend
plt.legend(loc='upper right')


# In[30]:


# Calculate the skewness 
print(skew(dataset['chol'], nan_policy='omit'))


# The data in this column is highly positively skewed distribution. Hence Median Imputation method will be suitable. 

# ##### Median Imputation

# In[31]:


### Replacing missing values with median ###

# Chol column contains 3.3% missing values, as shown in the visual above.
# After converting 0 -> null values, missing values increased to 22%.
# Filling missing values with the median value, as the column is continuous.

dataset['chol'].fillna(value = round(dataset['chol'].median(), 1), inplace=True)
dataset["chol"].astype(int)


# In[32]:


### Checking if any missing value is left ###

se = dataset["chol"].tolist()
for i in se:
    if i == 0.0 or i == np.nan:
        p='true'
    else:
        p='false'
print(p)


# In[33]:


# All missing values are now imputed in "chol" column


# #### Fbs Column

# In[34]:


### Checking values in "fbs" column ###

# The person's fasting blood sugar
# (> 120 mg/dl) 1 = true, 0 = false

print(dataset["fbs"].value_counts().index.tolist())


# ##### Mode Imputation

# In[35]:


### Replacing missing values with mode ###

# fbs column contains 9.8% missing values, as shown in the visual above.
# Filling missing values with the mode value, as the column is binary.

dataset['fbs'] = dataset['fbs'].fillna(value = dataset['fbs'].mode()[0])
dataset["fbs"].astype(int)
dataset.info()


# #### Restecg Column

# In[36]:


### Checking values in "restecg" column ###

# Resting electrocardiographic measurement
# 0 = normal,
# 1 = having ST-T wave abnormality,
# 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria.

print(dataset["restecg"].value_counts().index.tolist())


# ##### Mode Imputation

# In[37]:


### Replacing missing values with mode ###

# Restecg column contains 0.2% missing values, as shown in the visual above.
# Filling missing values with the mean value, as the column is categorical.

dataset['restecg'].fillna(value = dataset['restecg'].mode()[0], inplace=True)
dataset["restecg"].astype(int)


# #### Thalach Column

# In[38]:


### Checking values in "thalach" column ###

# The person's maximum heart rate achieved.

print(dataset["thalach"].value_counts().index.tolist())


# In[39]:


### Checking how many values are "0" ###

se = dataset["thalach"].tolist()
for i in se:
    if i == 0.0:
        p='true'
    else:
        p='false'
print(p)


# ##### Checking data distribution

# In[40]:


### Checking how data is distributed ###
# to decide the best method for imputation (mean or median)

# Plotting histogram plot
sns.histplot(dataset['thalach'], kde=True, color='yellowgreen')
# Removing color of the edges of histogram bars
plt.rcParams['patch.edgecolor'] = 'none'
# Plotting mean, median and mode
plt.axvline(dataset['thalach'].mean(), color='blue', linestyle='dashed', linewidth=1, label="Mean: " + str(round(dataset['thalach'].mean(), 2)))
plt.axvline(dataset['thalach'].median(), color='red', linestyle='dashed', linewidth=1, label="Median: " + str(round(dataset['thalach'].median(), 2)))
plt.axvline(dataset['thalach'].mode()[0], color='black', linestyle='dashed', linewidth=1, label="Mode: " + str(round(dataset['thalach'].mode()[0], 2)))
# Displaying legend
plt.legend(loc='upper right')


# In[41]:


# Calculate the skewness 
print(skew(dataset['thalach'], nan_policy='omit'))


# The distribution of the variable is approximately normal (symmetrical). Hence Mean Imputation method will be suitable.

# ##### Mean Imputation

# In[42]:


### Replacing missing values with mean ###

# Thalach column contains 6.0% missing values, as shown in the visual above.
# Filling missing values with the mean value, as the column is continuous.

dataset['thalach'].fillna(value = dataset['thalach'].mean().round(1), inplace=True)
dataset["thalach"].astype(int)


# #### Exang Column

# In[43]:


### Checking values in "exang" column ###

# Exercise induced angina
# 1 = yes, 0 = no

print(dataset["exang"].value_counts().index.tolist())


# ##### Mode Imputation

# In[44]:


### Replacing missing values with mode ###

# Exang column contains 6.0% missing values, as shown in the visual above
# Filling missing values with the mode value, as the column is binary
dataset['exang'].fillna(value = dataset['exang'].mode()[0], inplace=True)
dataset["exang"].astype(int)


# #### Oldpeak Column

# In[45]:


### Checking values in "oldpeak" column ###

# ST depression induced by exercise relative to rest.
# 'ST' relates to positions on the ECG plot.

print(dataset["oldpeak"].value_counts().index.tolist())


# ##### Checking data distribution

# In[46]:


### Checking how data is distributed ###
# to decide the best method for imputation (mean or median)

# Plotting histogram plot
sns.histplot(dataset['oldpeak'], kde=True, color='yellowgreen')
# Removing color of the edges of histogram bars
plt.rcParams['patch.edgecolor'] = 'none'
# Plotting mean, median and mode
plt.axvline(dataset['oldpeak'].mean(), color='blue', linestyle='dashed', linewidth=1, label="Mean: " + str(round(dataset['oldpeak'].mean(), 2)))
plt.axvline(dataset['oldpeak'].median(), color='red', linestyle='dashed', linewidth=1, label="Median: " + str(round(dataset['oldpeak'].median(), 2)))
plt.axvline(dataset['oldpeak'].mode()[0], color='black', linestyle='dashed', linewidth=1, label="Mode: " + str(round(dataset['oldpeak'].mode()[0], 2)))
# Displaying legend
plt.legend(loc='upper right')


# In[47]:


# Calculate the skewness 
print(skew(dataset['oldpeak'], nan_policy='omit'))


# The data in this column is highly positively skewed distribution. Hence Median Imputation method will be suitable

# ##### Median Imputation

# In[48]:


### Replacing missing values with median ###

# Oldpeak column contains 6.7% missing values, as shown in the visual above
# Filling missing values with the median value, as the column is continuous

dataset['oldpeak'].fillna(value = round(dataset['oldpeak'].median(), 1), inplace=True)
dataset["oldpeak"].astype(int)


# #### Slope Column

# In[49]:


### Checking values in "slope" column ###

# The slope of the peak exercise ST segment.
# 1: upsloping, 2: flat, 3: downsloping.

print(dataset["slope"].value_counts().index.tolist())


# ##### Mode Imputation

# In[50]:


### Replacing missing values with mode ###

# Slope column contains 33.6% missing values, as shown in the visual above.
# Filling missing values with the mode value, as the column is categorical.

dataset['slope'].fillna(value = dataset['slope'].mode()[0], inplace=True)
dataset["slope"].astype(int)


# #### CA Column

# In[51]:


### Checking values in "CA" column ###

# The number of major vessels (0-3).

print(dataset["ca"].value_counts().index.tolist())


# In[52]:


### Deleting CA column ###

# CA column contains 66.4% missing values, as shown in the visual above.
# As the missing values are more than 65%, it is better to delete the column [1].
# [1] Beginner’s Guide to Missing Value Ratio and its Implementation, analyticsvidhya.com. [Online]. Available: https://www.analyticsvidhya.com/blog/2021/04/beginners-guide-to-missing-value-ratio-and-its-implementation/.

dataset.drop("ca", axis=1, inplace=True)
dataset.head()


# In[53]:


# Now 13 columns are there for further pre-processing


# #### Thal Column

# In[54]:


### Checking values in "thal" column ###

# A blood disorder called thalassemia.
# 3 = normal, 6 = fixed defect, 7 = reversable defect.

print(dataset["thal"].value_counts().index.tolist())


# ##### Mode Imputation

# In[55]:


### Replacing missing values with mode ###

# Thal column contains 52.8% missing values, as shown in the visual above.
# Filling missing values with the mode value, as the column is categorical.

dataset['thal'].fillna(value = dataset['thal'].mode()[0], inplace=True)
dataset["thal"].astype(int)


# #### Goal  Column (Target)

# In[56]:


### Checking values in "goal" column (target variable) ###

# Target variable - diagnosis of heart disease (angiographic disease status).
# 0 = no presence of heart disease, [1, 2, 3, 4] = different levels of heart disease presence.

print(dataset["goal"].value_counts().index.tolist())


# In[57]:


### Returning a tuple representing the dimensionality of the Dataset ###

dataset.shape


# In[58]:


# Returning the column labels of the DataFrame

dataset.columns


# In[59]:


### Checking if all the columns are now clean ###

dataset.info()


# In[60]:


### Checking percentage of missing values ###

plot_null(dataset)


# In[61]:


# No missing value is now there


# ### c) Replacing column values

# #### Thal column

# In[62]:


### Converting the datatype of Thal column

dataset["thal"].astype(int)


# In[63]:


### Checking values in Thal column ###

dataset['thal'].head()


# In[64]:


# Thal column values need replacing, as they are hard to interpret
# Current values: 3 = normal, 6 = fixed defect, 7 = reversable defect.
# New values: 1= normal, 2= fixed defect, 3= reversable defect

dataset['thal'].replace([3, 6, 7],[1, 2, 3], inplace=True)


# In[65]:


### Checking values in Thal column ###

dataset['thal'].head()


# #### Goal column (target)

# In[66]:


### Checking values in 'goal' column ###

dataset[["goal"]].value_counts()


# In[67]:


# Current values: 0 = absence of heart disease, [1,2,3,4] = different levels of heart disease presence.
# Currently the goal column is a multi-class classification.
# Goal column values needs replacing.
# 1-4 values (different levels of heart disease) are not required to answer my research questions.
# For my research questions, the goal column needs to be converted to a binary classification.
# New values: 0 = absence of heart disease, 1= presence of heart disease


# In[68]:


dataset['goal'].replace([2, 3, 4],[1, 1, 1], inplace=True)


# In[69]:


### Checking values in 'goal' column ###

dataset["goal"].value_counts()


# In[70]:


### Displaying number/count of patients/records in each class ###

sns.countplot(data= dataset, x='goal')
plt.title('Count of Heart Disease Classes')


# In[71]:


### Checking how many classes we now have ###

len(dataset['goal'].value_counts().index)


# In[72]:


### Displaying percentage of patients/records in each class ###

# Create a figure and a set of subplots
fig, ax = plt.subplots()

### Create a pie chart ###
wedges, texts, autotexts = ax.pie(dataset['goal'].value_counts().values, autopct = '%1.1f%%')

### Stating labels ###
label = ["0: Absence of heart disease", "1: Presence of heart disease"]

### Giving the above stated labels to our pie chart ###
ax.legend(wedges, label,
          title="Target",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

### Setting the text properties of pie chart ###
plt.setp(autotexts, size=11, weight="bold")

### Settinng title of the plot ###
ax.set_title("Instances of the target variable")

### Showing plot ###
plt.show()


# In[73]:


# The goal column is now a binary column.
# Number of instances that belong to class 0 are 410 and 1 are 508 (balanced binary classification).


# ### d) Renaming column names

# In[74]:


# As the column names are ambiguous, renaming columns is a good practice here


# In[75]:


# Renaming all columns in a single line using a dictionary inside the "rename" function
dataset.rename(columns={'cp': 'chest_pain_type', 'trestbps': 'resting_bp', 'chol': 'cholesterol', 'fbs': 'fasting_blood_sugar', 'thalach': 'max_heart_rate', 'exang': 'ex_induced_angina', 'oldpeak': 'st_depression', 'thal': 'thalassemia'}, inplace=True)


# In[76]:


### Displaying data ###

dataset.head()


# ### e) Checking Outliers

# In[77]:


# Creating Function:
### Checking outliers using histogram, quantile-quantile and box plots for all numerical columns ###

def checking_outliers(df: pd.DataFrame):
    
    # Initialise the subplot function using number of rows and columns 
    figure, axes = plt.subplots(nrows= 5, ncols=3, figsize = (10, 15), constrained_layout = True)
    figure.suptitle('Checking Outliers of Columns', size=15, y=1.03)
    
    ### Age column ###
    # Checking how the data is distributed:
    sns.histplot(df['age'], kde=True, ax=axes[0, 0])
    # Verifying the distribution using a quantile-quantile plot:
    qqplot(df['age'], fit=True, line="45", ax=axes[0, 1])
    # Box plot to detect outliers
    sns.boxplot(y = 'age', data = df, ax=axes[0, 2])
    
    ### resting_bp column ###
    # Checking how the data is distributed:
    sns.histplot(df['resting_bp'], kde=True, ax=axes[1, 0])
    # Verifying the distribution using a quantile-quantile plot:
    qqplot(df["resting_bp"], fit=True, line="45", ax=axes[1, 1])
    # Box plot to detect outliers
    sns.boxplot(y = 'resting_bp', data = df, ax=axes[1, 2])
    
    ### cholesterol column ###
    # Checking how the data is distributed:
    sns.histplot(df['cholesterol'], kde=True, ax=axes[2, 0])
    # Verifying the distribution using a quantile-quantile plot:
    qqplot(df["cholesterol"], fit=True, line="45", ax=axes[2, 1])
    # Box plot to detect outliers
    sns.boxplot(y = 'cholesterol', data = df, ax=axes[2, 2])
    
    ### max_heart_rate column ###
    # Checking how the data is distributed:
    sns.histplot(df['max_heart_rate'], kde=True, ax=axes[3, 0])
    # Verifying the distribution using a quantile-quantile plot:
    qqplot(df["max_heart_rate"], fit=True, line="45", ax=axes[3, 1])
    # Box plot to detect outliers
    sns.boxplot(y = 'max_heart_rate', data = df, ax=axes[3, 2])
    
    ### st_depression column ###
    # Checking how the data is distributed:
    sns.histplot(df['st_depression'], kde=True, ax=axes[4, 0])
    # Verifying the distribution using a quantile-quantile plot:
    qqplot(df["st_depression"], fit=True, line="45", ax=axes[4, 1])
    # Box plot to detect outliers
    sns.boxplot(y = 'st_depression', data = df, ax=axes[4, 2])
    
    
    plt.show()
    
checking_outliers(dataset)


# #### age column

# In[78]:


# A normal distribution it is and no outliers found.


# #### resting_bp column

# In[79]:


# Outliers have been detected.
# Positively skewed distribution it is, hence will use IQR to remove outliers.


# ##### IQR

# In[80]:


### Finding the IQR ###

q1 = dataset['resting_bp'].quantile(0.25)
q3 = dataset['resting_bp'].quantile(0.75)

iqr = q3 - q1


# In[81]:


### Finding the upper and lower limits ###

upper_limit = q3 + 1.5 * iqr
lower_limit = q1 - 1.5 * iqr


# In[82]:


### Finding outliers ###

dataset[dataset['resting_bp'] > upper_limit]
dataset[dataset['resting_bp'] < lower_limit]


# In[83]:


### Trimming outliers ###

dataset = dataset[dataset['resting_bp'] < upper_limit]
dataset.shape


# In[84]:


### Checking the statistics using the “Describe” function ###

dataset['resting_bp'].describe()


# #### cholesterol column

# In[85]:


# Outliers have been detected.
# The distribution is highly positively skewed distribution,
# hence will use IQR to remove outliers


# ##### IQR

# In[86]:


### Finding the IQR ###

q1 = dataset['cholesterol'].quantile(0.25)
q3 = dataset['cholesterol'].quantile(0.75)

iqr = q3 - q1


# In[87]:


### Finding the upper and lower limits ###

upper_limit = q3 + 1.5 * iqr
lower_limit = q1 - 1.5 * iqr


# In[88]:


### Finding outliers ###

dataset[dataset['cholesterol'] > upper_limit]
dataset[dataset['cholesterol'] < lower_limit]


# In[89]:


### Trimming outliers ###

dataset = dataset[dataset['cholesterol'] < upper_limit]
dataset.shape


# #### max_heart_rate column

# In[90]:


# Outliers have been detected
# The distribution is approximately normal (symmetrical), hence will use z-score to remove outliers.


# ##### Z-score

# In[91]:


### Applying Z-score to remove outliers ###

# Finding the boundary values
print('Highest allowed', dataset['max_heart_rate'].mean() + 3*dataset['max_heart_rate'].std())
print('Lowest allowed', dataset['max_heart_rate'].mean() - 3*dataset['max_heart_rate'].std())


# In[92]:


### Finding the outliers ###

dataset[(dataset['max_heart_rate'] > 213) | (dataset['max_heart_rate'] < 62)]


# In[93]:


### Trimming outliers ###

new_df = dataset[(dataset['max_heart_rate'] < 213) & (dataset['max_heart_rate'] > 62)]
new_df


# In[94]:


### Capping on outliers ###

upper_limit = dataset['max_heart_rate'].mean() + 3*dataset['max_heart_rate'].std()
lower_limit = dataset['max_heart_rate'].mean() - 3*dataset['max_heart_rate'].std()


# In[95]:


### Applying the capping ###

dataset['max_heart_rate'] = np.where(dataset['max_heart_rate']>upper_limit,
                      upper_limit,
                      np.where(dataset['max_heart_rate']<lower_limit,
                               lower_limit,
                               dataset['max_heart_rate']))


# In[96]:


### Returning dimensionality of the Dataset ###

dataset.shape


# In[97]:


### Checking the statistics using the “Describe” function ###

dataset['max_heart_rate'].describe()


# #### st_depression column

# In[98]:


# Outliers have been detected.
# The distribution is highly positively skewed distribution,
# hence will use IQR to remove outliers


# ##### IQR

# In[99]:


### Finding the IQR ###

q1 = dataset['st_depression'].quantile(0.25)
q3 = dataset['st_depression'].quantile(0.75)

iqr = q3 - q1


# In[100]:


### Finding the upper and lower limits ###

upper_limit = q3 + 1.5 * iqr
lower_limit = q1 - 1.5 * iqr


# In[101]:


### Finding outliers ###

dataset[dataset['st_depression'] > upper_limit]
dataset[dataset['st_depression'] < lower_limit]


# In[102]:


### Trimming outliers ###

dataset = dataset[dataset['st_depression'] < upper_limit]
dataset.shape


# In[103]:


### Checking outliers of cleaned dataset ###

checking_outliers(dataset)


# All extreme values are now removed

# ### f) Detect and Remove Duplicates

# In[104]:


# Dropping duplicate rows
dataset.drop_duplicates(inplace = True)

# Resetting index after dropping duplicate rows
dataset.reset_index(drop = True, inplace = True)

# Returning dimensionality of the Dataset
dataset.shape


# In[105]:


# 2 duplicate records have been identified and removed


# In[106]:


### Displaying the final cleaned dataset ###

dataset.head()


# In[107]:


### Correlation Heat map of the final cleaned dataset ###

plot_corrheatmap(dataset)


# ### g) Saving the cleaned dataset into system

# In[108]:


# To save the final cleaned dataset to your preferred location, please change the file location
# This dataset will be used in the next notebook: "heart_disease_classification.ipynb"

dataset.to_csv(r'cleaned_heart_disease_dataset.csv', index = False)

