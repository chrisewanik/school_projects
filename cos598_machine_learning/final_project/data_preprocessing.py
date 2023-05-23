# %%
# Final Project Import and Data Cleaning

# %%
# Import libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
# from janitor import clean_names # pip install pyjanitor==0.23.1

# %%
# Load in the Adult dataset
adult_df = pd.read_csv("adult.data", header=None)

# %%
# Set the column names
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 
              'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

# %%
# Get the shape of the data
print(adult_df.shape)

# %%
adult_df.head()

# %%
# Note this dataset has a missing value represented by a '?' so we need to replace it with NaN
adult_df = adult_df.replace(' ?', np.nan)

# Check for missing values
adult_df.isnull().sum()

# %%
# Drop rows with missing values
adult_df = adult_df.dropna()

# %%
# Drop the fnlwgt and education_num columns
adult_df = adult_df.drop(['fnlwgt', 'education_num'], axis=1)

# Description of fnlwgt column from kaggle:
# Its a weight assigned by the US census bureau to each row. The literal meaning is that you will need to replicate each row, final weight times to 
# get the full data. And it would be somewhat 6.1 billion rows in it. Dont be shocked by the size, its an accumulated data over decades.

# We drop education_num because it is a duplicate of education and we want to keep the categorical variable and do one-hot encoding on it

# %%
# We will apply standardization to the age, capital_gain, capital_loss, and hours_per_week columns aka the continuous variables

# Create the Scaler object and fit it to the adult_df continuous variables
scaler = preprocessing.StandardScaler().fit(adult_df[['age', 'capital_gain', 'capital_loss', 'hours_per_week']])

# Transform the adult_df continuous variables
adult_scaled = scaler.transform(adult_df[['age', 'capital_gain', 'capital_loss', 'hours_per_week']])

# %%
# Verify the mean of the scaled data
adult_scaled.mean(axis=0)

# %%
# Convert the scaled data to a dataframe and set the column names
adult_scaled_cont_df = pd.DataFrame(adult_scaled, columns=['age','capital_gain', 'capital_loss', 'hours_per_week'])

# %%
# One hot encoding

# Create the OneHotEncoder object and fit it to the adult_df categorical variables
enc = preprocessing.OneHotEncoder()

# Creat the fit object
encoder = enc.fit(adult_df[['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']])

# Transform the adult_df categorical variables
adult_encoded = encoder.transform(adult_df[['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']]).toarray()


# %%
# Create a list from encoder.categories_ to use as column names

# Create an empty list
cat_cols = []

# Loop through the encoder.categories_ list
for i in range(len(encoder.categories_)):
    # Loop through the sublists
    for j in range(len(encoder.categories_[i])):
        # Append the values to the cat_cols list
        cat_cols.append(encoder.categories_[i][j])


# %%
# Preview the cat_cols list
cat_cols

# %%
# Convert the encoded data to a dataframe and set the column names
adult_encoded_df = pd.DataFrame(adult_encoded, columns=cat_cols)

# %%
adult_encoded_df.head()

# %%
incomes_df = adult_df['income']

# %%
incomes_df.reset_index(drop=True, inplace=True)

# %%
# Concat the scaled continuous variables and the encoded categorical variables
adult_cleaned_df = pd.concat([adult_scaled_cont_df, adult_encoded_df, incomes_df], axis=1)

# %%
# Recode the income column to 1 for >50K and 0 for <=50K
adult_cleaned_df['income'] = np.where(adult_cleaned_df['income'] == ' >50K', 1, 0)

# %%
# Check the shape of the cleaned data
adult_cleaned_df.shape

# %%
# Save the cleaned data to a csv
adult_cleaned_df.to_csv('adult_cleaned.csv', index=False)

# %%
adult_cleaned_df.head()

# %% [markdown]
# 2nd Dataset

# %%
# Load in the auto-mpg dataset that is space seperated into a pandas dataframe
auto_df = pd.read_csv("auto-mpg.data", header=None, delim_whitespace=True)

# %%
# Get the shape of the data
print(auto_df.shape)

# %%
# Set the column names
auto_df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

# %%
# Preview the data
auto_df.head()

# %%
# Split the car_name column into car_name and car_model
auto_df[['make', 'model']] = auto_df['car_name'].str.split(' ', n = 1, expand=True)

# Drop the car_name column
auto_df = auto_df.drop('car_name', axis=1)

# %%
auto_df.head()

# %%
# Get the number of unique values for cylinders, model_year, and origin
print(len(auto_df['cylinders'].unique())) # One Hot Encode
print(len(auto_df['model_year'].unique())) # One Hot Encode
print(len(auto_df['origin'].unique())) # One Hot Encode
print(len(auto_df['make'].unique())) # Label Encode

# %%
# Check for missing values
auto_df.isnull().sum()

# %%
# Note this dataset has a missing value represented by a '?' so we need to replace it with NaN
auto_df = auto_df.replace('?', np.nan)

# Drop rows with missing values
auto_df = auto_df.dropna()

# Get the number of rows
len(auto_df)

# %%
# Create the Scaler object and fit it to the adult_df continuous variables
scaler = preprocessing.StandardScaler().fit(auto_df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']])

# Transform the adult_df continuous variables
auto_scaled = scaler.transform(auto_df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']])

# %%
# Convert the scaled data to a dataframe and set the column names
auto_scaled_cont_df = pd.DataFrame(auto_scaled, columns=['mpg', 'displacement', 'horsepower', 'weight', 'acceleration'])
# We will concat this later

# %%
# One hot encoding

# Create the OneHotEncoder object and fit it to the adult_df categorical variables
enc = preprocessing.OneHotEncoder()

# Creat the fit object
encoder = enc.fit(auto_df[['cylinders', 'model_year', 'origin']])

# Transform the adult_df categorical variables
auto_encoded = encoder.transform(auto_df[['cylinders', 'model_year', 'origin']]).toarray()


# %%
# Create a list from encoder.categories_ to use as column names

# Create an empty list
cat_cols = []

# Loop through the encoder.categories_ list
for i in range(len(encoder.categories_)):
    # Loop through the sublists
    for j in range(len(encoder.categories_[i])):
        # Append the values to the cat_cols list
        cat_cols.append(encoder.categories_[i][j])


# %%
# Append "cylinders" to the cat_cols list at location 0:5
cat_cols[0:5] = ['cylinders_' + str(i) for i in cat_cols[0:5]]

# %%
# Append "cylinders" to the cat_cols list at location 0:5
cat_cols[5:-3] = ['year_' + str(i) for i in cat_cols[5:-3]]

# %%
# Append "cylinders" to the cat_cols list at location 0:5
cat_cols[-3:] = ['origin_' + str(i) for i in cat_cols[-3:]]

# %%
# Preview the cat_cols list
cat_cols # It may be worth checking the length of this list to make sure it is the same length as the number of columns in adult_encoded and that nothing dumb happened

# %%
# Convert the encoded data to a dataframe and set the column names
auto_encoded_df = pd.DataFrame(auto_encoded, columns=cat_cols)

# %%
# Concat the scaled continuous variables and the encoded categorical variables
auto_cleaned_df = pd.concat([auto_scaled_cont_df, auto_encoded_df], axis=1)

# %%
auto_cleaned_df.head()

# %%
# Get the shape of the cleaned data
auto_cleaned_df.shape

# %%
# Save the cleaned data to a csv
auto_cleaned_df.to_csv('auto_cleaned.csv', index=False)


