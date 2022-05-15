#!/usr/bin/env python
# coding: utf-8

# # Reading dataset

# In[1]:


import pandas as pd


# In[2]:


# reading from csv file using pandas
data = pd.read_csv('dataset/set.csv')

# display first 5 rows from dataset
data.head()


# # Checking for missing/null values

# In[3]:


# Print information about a dataset including the index dtype and columns, non-null values and memory usage.
data.info()


# In[4]:


# display number of null values in every column
data.isnull().sum(axis=0)


# <strong> There is not any null or missing values </strong>

# # Encoding of non-numeric values (look at: section 4)
# We need to convert non-numeric values to numeric values. But first, we need to understand the type of each non-numeric column. There are mainly 3 types:
# 1. Binary: The column contains only 2 types of values (example: married: yes/no).
# 2. Nominal: The column contains more than 2 types of values; The values can't have a specific order (example: country: Egypt/France/UK).
# 3. Ordinal: The column contains more than 2 types of values; The values have a specific order (example: size: small/medium/large).
# 
# To determine the type of each non-numeric column, we need to know what unique values does each column contain

# In[5]:


# display each of these columns with both number of unique values of them and these uniques values
# better to be at the form of Dataframe
for col in ['Race','Marital Status','T Stage', 'N Stage','6th Stage','Grade','A Stage',
           'Estrogen Status','Progesterone Status','Status']:
    print("column name: '"+ col +"'\n", "number of values: " +  str(len(data[col].unique())) + "'\n", "The values are", 
          data[col].unique())
    print("----------------------------------")


# 1. The columns <code>A Stage</code>, <code>Estrogen Status</code>, <code>Progesterone Status</code> and <code>Status</code> are binary properties <br/><br/>
# 2. The columns <code>T Stage</code>, <code>N Stage</code>, <code>6th Stage</code> and <code>Grade</code>  are ordinal (categorical) properties <br/><br/>
# 3. The columns <code>Race</code> and <code>Marital Status</code> are nominal (categorical) properties <br/><br/>
# 
# 

# ## Binary-Encoded 

# In[6]:


data_binary_encoded = data.replace({
    'A Stage': {'Regional': 1, 'Distant': 0},
    'Estrogen Status': {'Positive': 1, 'Negative': 0},
    'Progesterone Status' : {'Positive': 1 ,'Negative':0},
    'Status' : {'Alive': 1,'Dead':0}
})


# ## Ordinal-Encoding

# In[7]:


data_ordinal_binary_encoded = data_binary_encoded.replace({
    'T Stage': {'T1': 1,'T2': 2,'T3':3,'T4':4},
    'N Stage': {'N1': 1,'N2': 2,'N3':3},
    '6th Stage': {'IIA': 1,'IIB': 2,'IIIA':3,'IIIB':4,'IIIC':5},
    'Grade' : {'Well differentiated; Grade I': 1,'Moderately differentiated; Grade II': 2,
              'Poorly differentiated; Grade III':3,'Undifferentiated; anaplastic; Grade IV':4}
})


# ## nominal encoding (one hot encoding)
# Then we use <code>pd.get_dummies()</code> to convert the other non-numeric columns to one-hot encoding

# In[8]:


data_encoded_final = pd.get_dummies(data_ordinal_binary_encoded)

data_encoded_final.head()


# In[9]:


# data types of final-encoded-data
data_encoded_final.dtypes


# # Splitting data into input and output

# In[10]:


data_input = data_encoded_final.drop(columns=['Status'])
data_output = data_encoded_final['Status']


# # Splitting data into train, validation, and test

# In[11]:


from sklearn.model_selection import train_test_split

X, X_test, y, y_test = train_test_split(
    data_input, data_output, test_size=0.30, random_state=0
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.30, random_state=0
)

print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('-------------------------')
print('X_val:', X_val.shape)
print('y_val:', y_val.shape)
print('-------------------------')
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)


# # Solving the problem of imbalanced data
# Displaying output value counts for our training set

# In[12]:


y_train.value_counts()


# We use `imbalanced-learn` package to make our training set balanced. We use two methods:
# 1. Undersampling: Removing samples from the majority class (class 0)
# 2. Oversampling: Repeating samples from the minority class
# 

# **Undersamping:**
# 
# Using undersampling to reduce the samples of class 0 so that Class 1 : Class 0 = 0.5

# In[13]:


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=0.5, random_state=0)

X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

y_train_rus.value_counts()


# **Oversamping:**
# 
# Using oversampling to increase the samples of class 1 so that Class 1 : Class 0 = 1

# In[14]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy=1.0, random_state=0)

X_train_balanced, y_train_balanced = ros.fit_resample(X_train_rus, y_train_rus)

# Uncomment the following line if you want to see the difference when the data is not balanced
#X_train_balanced, y_train_balanced = X_train, y_train

y_train_balanced.value_counts()


# Now we use `(X_train_balanced, y_train_balanced)` to train our model

# In[ ]:




