# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb

# Reading data and macking DataFrame
df = pd.read_csv('Data_Science_Internship - Dump.csv')

# Dropping 1'st Column as it is not important
df.drop(df.columns[0], axis=1, inplace=True)

# Columns
cols = list(df.columns)

# Dropping leads with STATUS other than ‘WON’ or ‘LOST’
df = df[(df['status'] == 'WON') | (df['status'] == 'LOST')]

# Replacing '9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0' with NaN
df.replace(to_replace='9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0', value=np.nan, inplace=True)

# Filling '0' in place of NaN in lost_reason
df['lost_reason'] = df['lost_reason'].fillna(0)

# Filling '1' for every row which don't have 0 in lost_reason
df['lost_reason'] = df['lost_reason'].apply(lambda x: 1 if x!=0 else 0)

# Our dataset have many NaN values so I am finding most frequent values from each column
most_freq_values_dict = {}
for i in cols[3:]:
    value = df[i].value_counts().idxmax()
    most_freq_values_dict[i] = value

# Filling NaN of these columns with most frequent values
for i in cols[3:]:
    df[i].fillna(value=most_freq_values_dict[i], inplace=True)

# Splitting data to label and features
label = df['status']
features = df.drop(['status'], axis=1)

# Using OneHotEncoder and ColumnTransformer for transforming Categorical columns to numeric form
ct = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [i for i in range(14)])], remainder='passthrough')
features = ct.fit_transform(features)

# Transforming labels using LabelEncoder
le = LabelEncoder()
label = le.fit_transform(label)

# Splitting data into training and testing data
features_train, features_test, label_train, label_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Creating model object
xgb_model = xgb.XGBClassifier()

# Training model
xgb_model.fit(features_train, label_train)

# Making predictions
predictions = xgb_model.predict(features_test)

# Finding accuracy
accuracy = accuracy_score(label_test, predictions)

# Testing score
score_test = xgb_model.score(features_test, label_test)

# Training score
score_train = xgb_model.score(features_train, label_train)

# Finding precision_score
precision = precision_score(label_test, predictions)

# Finding recall_score
recall = recall_score(label_test, predictions)

# Finding f1_score
f1 = f1_score(label_test, predictions)