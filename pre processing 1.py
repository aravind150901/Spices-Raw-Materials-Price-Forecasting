from sqlalchemy import create_engine
import pandas as pd
from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np
from sqlalchemy import create_engine 
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

user = 'aravind'
pw = 'datascience1092'
db = 'spicesdb'
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

Spices = pd.read_excel(r"D:\360DigiTMG\2 Project - Spices Price Forecasting\Dataset\Spices_Final_Data(Mine_loc_without_grade).xlsx")

# Dumping data into the database
Spices.to_sql('spices', con=engine, if_exists='replace', chunksize=500, index=False)

# Loading data from the database
sql = 'SELECT * FROM spices'
df = pd.read_sql_query(sql, con=engine)

# Removing unwanted columns
df.drop(["Grade", "Location"], axis=1, inplace=True)

# Segregate unique values of spice names
unique_spices = df['Spices'].unique()

# Box plot for each spice separately
for Spices in unique_spices:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Spices', y='Price', data=df[df['Spices'] == Spices])
    plt.title(f'Box Plot for {Spices} Prices')
    plt.show()

# List of spices to perform winsorization on
spices_with_outliers = ['FENUGREEK', 'AJWAN SEED','MUSTARD', 'SAFFRON',
                       'CLOVE', 'CASSIA', 'CARDAMOM(SMALL)']    

from scipy.stats.mstats import winsorize

# Outliers treatment using winsorization
for spice in spices_with_outliers:
    spice_data = df[df['Spices'] == spice]
    spice_data['Price'] = winsorize(spice_data['Price'], limits=[0.05, 0.05])


# Box plot for each spice after outliers treatment
for spice in spices_with_outliers:
    spice_data = df[df['Spices'] == spice]['Price']
    
    # Check if there are enough data points for the spice
    if len(spice_data) > 0:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Spices', y='Price', data=df[df['Spices'] == spice])
        plt.title(f'Box Plot for {spice} Prices (After Outliers Treatment)')
        plt.show()
    else:
        print(f'Insufficient data points for {spice} after outliers treatment.')


# Create an empty list to store preprocessed data for each spice
spices_data_list = []
# Create an empty list to store preprocessed data for each spice
for Spices in unique_spices:
    
    # Get the data for the current spice
    spice_data = df[df["Spices"] == Spices]  

    # Append the preprocessed data to the list
    spices_data_list.append(spice_data)

# Imputation for each spice DataFrame in spices_data_list
for spice_data in spices_data_list:
    spice_name = spice_data['Spices'].iloc[0]  # Get the spice name from the DataFrame

    # Impute missing values using mean
    spice_data['Price'].fillna(spice_data['Price'].mean(), inplace=True)

    # You can use other imputation methods based on your requirement.
    # For example, median imputation: spice_data['Price'].fillna(spice_data['Price'].median(), inplace=True)

    # Alternatively, forward fill (ffill): spice_data['Price'].fillna(method='ffill', inplace=True)

    # Display the result or save it as needed
    print(f"Imputed data for {spice_name}:\n{spice_data}\n")

# Iterate through each preprocessed DataFrame and save it to SQL
for spice_data in spices_data_list:
    spice_name = spice_data['Spices'].iloc[0]  # Assuming 'Spices' is a column in your DataFrame
    table_name = f'{spice_name.lower()}'  # Adjust as needed
    
    # Round the 'Price' column to two decimal places
    spice_data['Price'] = spice_data['Price'].round(2)
    
    # Save the DataFrame to SQL
    spice_data.to_sql(table_name, con=engine, if_exists='replace', chunksize=500, index=False)
    
    print(f"Table {table_name} saved to SQL.")
    

        
     
for spice_data in spices_data_list:
    spice_name = spice_data['Spices'].iloc[0]  # Assuming 'Spices' is a column in your DataFrame
    file_name = f'{spice_name.lower()}_preprocessed.csv'  # Adjust as needed
    
    # Round the 'Price' column to two decimal places
    spice_data['Price'] = spice_data['Price'].round(2)
    
    # Save the DataFrame to a CSV file
    spice_data.to_csv(file_name, index=False)
    
    print(f"File {file_name} saved.")

    
