import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os
import shutil

# Download dataset
try:
    source_path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")

    print(f"Dataset is located here: {source_path}")

    path = '/Users/nolanleavitt/.cache/kagglehub/datasets/sobhanmoosavi/us-accidents/versions/13/US_Accidents_March23.csv'
    destination_path = "/Users/nolanleavitt/Desktop/python/us_accidents/data/us_accidents.csv"

        # try to move downloaded data
    try:
        shutil.move(path, destination_path)
        print(f"File successfully moved to: {destination_path}")

    except:
        print('Operation failed')

except:
    pass

#Specify columns to keep
columns_to_keep = ['ID', 'Source', 'Severity', 'Start_Time', 'End_Time',
                   'City', 'County', 'State', 'Zipcode', 'Country',
                   'Temperature(F)', 'Weather_Condition', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                     'No_Exit', 'Railway', 'Roundabout', 'Stop', 'Traffic_Signal']

#chunk_size = 10000
# load dataset into dataframe
df = pd.read_csv("data/us_accidents.csv", nrows=100000, usecols=columns_to_keep)

print(df.info())
#print(df.describe(include='all'))
print(df.dtypes)

# Handle missing data
missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 

# drop missing values from city, & weather condition
df_dropped_na = ['City', 'Zipcode', 'Weather_Condition', 'City']
df.dropna(subset=df_dropped_na, inplace = True)

# check distribution of temperature and weather condition
#sns.histplot(df['Temperature(F)'].dropna(), kde=True) 
#plt.show() # temperature seems to be normally distributed, replace with mean

# replace missing data with means for temperature
mean_temp = df['Temperature(F)'].mean()
df['Temperature(F)'] = df['Temperature(F)'].fillna(mean_temp)

# Create year/month
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])

df['Year'] = df['Start_Time'].dt.year
df['Month'] = df['Start_Time'].dt.month

# Create full time of incident & day of incident
df['full_incident_time'] = df['End_Time'] - df['Start_Time'] 
df['Day_of_Incident'] = df['Start_Time'].dt.dayofweek
print(df['Day_of_Incident'].value_counts())


# Create visuals for analysis
print(df['Weather_Condition'].value_counts())

# weather over severity
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Weather_Condition', hue='Severity', palette='viridis')
plt.xticks(rotation=45)
plt.title("Accident Severity by Weather Condition")
plt.show()

# incidents by month
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Month', hue='Severity', palette='viridis')
plt.title("Accidents by month")

plt.show()

# TODO - bin weather, create region variable, 
