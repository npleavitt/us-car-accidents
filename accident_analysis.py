import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os
import shutil
from docx import Document
from docx.shared import Inches

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
        print('File already exists')

except:
    pass

#Specify columns to keep
columns_to_keep = ['ID', 'Source', 'Severity', 'Start_Time', 'End_Time',
                   'City', 'County', 'State', 'Zipcode', 'Country',
                   'Temperature(F)', 'Weather_Condition', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                     'No_Exit', 'Railway', 'Roundabout', 'Stop', 'Traffic_Signal']

#chunk_size = 10000
# load dataset into dataframe
df = pd.read_csv("us_accidents.csv", nrows=100000, usecols=columns_to_keep)

#######################################   Data Cleaning   ############################################

# Handle missing data
missing_data = df.isnull()

def get_missing_counts(df): # Call function on df to get counts of missing by variable
    for column in df.tolist():
        print(column)
        print (missing_data[column].value_counts())
        print("") 

# drop missing values from city, & weather condition
df_dropped_na = ['City', 'Zipcode', 'Weather_Condition', 'City']
df.dropna(subset=df_dropped_na, inplace = True)

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


#############################   Create visuals for analysis   #############################

# weather over severity
def weather_condition_chart(df):
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x='Weather_Condition', hue='Severity', palette='viridis')
    plt.xticks(rotation=15)
    plt.title("Accident Severity by Weather Condition")
    plt.show()

# incidents by month
def incidents_by_month(df):
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x='Month', hue='Severity', palette='viridis')
    plt.title("Accidents by month")
    plt.show()

# incidents by temperature
def incidents_by_temperature(df):
    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x='Temperature(F)', hue='Severity', palette='husl')
    plt.xticks(rotation=15)
    plt.title("Accident Severity by Weather Condition")
    plt.show()

#############################   Create Bins   #############################

# Bins for weather
weather_condition = {
'Clear': 'Clear/Fair',    
'Overcast'
'Mostly Cloudy': 'Cloudy',    
'Partly Cloudy': 'Cloudy',    
'Scattered Clouds': 'Cloudy',    
'Light Rain': 'Rain/Storm',    
'Haze': 'Smoke/Haze',  
'Fair': 'Clear/Fair',   
'Rain': 'Rain/Storm',    
'Heavy Rain ': 'Rain/Storm',    
'Fog': 'Fog/Haze/Mist',       
'Light Snow'    
'Smoke': 'Fog/Haze/Mist',   
'Light Drizzle'    
'Mist': 'Fog/Haze/Mist',   
'Drizzle': 'Rain/Storm',  
'Patches of Fog': 'Fog/Haze/Mist',    
'Light Thunderstorms and Rain': 'Rain/Storm',    
'Fair / Windy': 'Clear/Fair',
'Rain Showers': 'Rain/Storm',
'Widespread Dust': 'Fog/Haze/Mist',
'Light Freezing Drizzle': 'Rain/Storm',    
'Thunderstorms and Rain': 'Rain/Storm',    
'Shallow Fog': 'Fog/Haze/Mist',   
'Volcanic Ash': 'Other',    
'Light Rain / Windy': 'Rain/Storm',
'Showers in the Vicinity': 'Rain/Storm',
'Light Rain Showers': 'Rain/Storm',
'Light Freezing Fog': 'Fog/Haze/Mist',
'Partly Cloudy / Windy': 'Cloudy',   
'Mostly Cloudy / Windy': 'Cloudy',    
'Heavy Drizzle': 'Rain/Storm',      
'Light Freezing Rain': 'Rain/Storm',        
'Light Haze': 'Fog/Haze/Mist',       
'Blowing Dust / Windy': 'Other',        
'Blowing Sand': 'Other',        
'Hail': 'Other',        
'Thunderstorm': 'Rain/Storm',        
'Light Rain Shower': 'Rain/Storm',        
'Light Rain with Thunder': 'Rain/Storm',        
'Thunder in the Vicinity': 'Rain/Storm'
}

# region dictionary for mapping 
state_to_region = {
    'AL': 'Southeast',
    'AK': 'West',
    'AZ': 'West',
    'AR': 'Southeast',
    'CA': 'West',
    'CO': 'West',
    'CT': 'Northeast',
    'DE': 'Northeast',
    'FL': 'Southeast',
    'GA': 'Southeast',
    'HI': 'West',
    'ID': 'West',
    'IL': 'Midwest',
    'IN': 'Midwest',
    'IA': 'Midwest',
    'KS': 'Midwest',
    'KY': 'Southeast',
    'LA': 'Southwest',
    'ME': 'Northeast',
    'MD': 'Northeast',
    'MA': 'Northeast',
    'MI': 'Midwest',
    'MN': 'Midwest',
    'MS': 'Southeast',
    'MO': 'Midwest',
    'MT': 'West',
    'NE': 'Midwest',
    'NV': 'West',
    'NH': 'Northeast',
    'NJ': 'Northeast',
    'NM': 'Southwest',
    'NY': 'Northeast',
    'NC': 'Southeast',
    'ND': 'Midwest',
    'OH': 'Midwest',
    'OK': 'Southwest',
    'OR': 'West',
    'PA': 'Northeast',
    'RI': 'Northeast',
    'SC': 'Southeast',
    'SD': 'Midwest',
    'TN': 'Southeast',
    'TX': 'Southwest',
    'UT': 'West',
    'VT': 'Northeast',
    'VA': 'Southeast',
    'WA': 'West',
    'WV': 'Southeast',
    'WI': 'Midwest',
    'WY': 'West'
}

# Weather condition & region
df['Weather_Condition'] = df['Weather_Condition'].map(weather_condition)

df['Region'] = df['State'].map(state_to_region)
print(df['Region'].value_counts())


#################################   Generate Descriptive Stats    ######################################






