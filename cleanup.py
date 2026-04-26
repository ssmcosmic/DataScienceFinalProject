# First read in the data, see how it looks, decide if cleaning etc. needed
# Python version: 3.14

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = "student_productivity_distraction_dataset_20000.csv"

df = pd.read_csv(filename)






# Clean up


# Check the columns in the dataset
print(df.info())




# Student ID is irrelevant - should be removed / not considered
df = df.drop(columns=['student_id'])

# Check for any null - none
print(df.isnull)


# Check for duplicates - none
print(df.duplicated().sum())


# Check for outliers / invalid ranges
print(df.describe())

# Check for data types
print(df.dtypes)



# Encoding


# Use one-hot encoding for non-numerical columns
# The reason for one-hot encoding is so that nominal data eg. gender is not mistaken by the model for having an order
df = pd.get_dummies(df, columns=['gender'], drop_first=True)






# EDA

# Get an idea for which features correlate the most with focus
print(df.corr(numeric_only=True)['focus_score'].sort_values(ascending=False))


# Check for skewness / if our target variable (focus scores) is not skewed towards any particular variable
df['focus_score'].hist(bins=30)
plt.show()



# Check correlation heat map

# Observation: very weak correlation between individual factors and focus
# Indicates: Maybe not linear relation - need to try Random Forest
# OR: it's a combination of specific factors that determine it? Keep examining

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()