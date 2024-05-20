import pandas as pd
from sqlalchemy import create_engine
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import pmdarima as pm


user = 'aravind'
pw = 'datascience1092'
db = 'spicesdb'
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# List to store loaded DataFrames
loaded_data_list = []

# List of table names
table_names = ['`ajwan seed`', '`black pepper(mg-_x000d_1)`', '`cardamom(large)`', '`cardamom(small)`',
               'cassia', 'chillies', 'clove', 'coriander', 'cumin',
               'fennel', 'fenugreek', 'garlic', 'mace', '`mentha oil`',
               'mustard', '`nutmeg (with shell)`', '`nutmeg (without shell}`', '`poppy seed`',
               'saffron', 'tamarind', 'turmeric']

for table_name in table_names:
    # SQL query to select all columns from the specified table
    sql_query = f'SELECT * FROM {table_name}'
    
    # Load the DataFrame from SQL
    loaded_data = pd.read_sql_query(sql_query, con=engine)
    
    # Append the loaded DataFrame to the list
    loaded_data_list.append(loaded_data)


###### TREND AND SEASONALITY PLOTS ######
for df in loaded_data_list:
    # Set 'Month&Year' as the index
    df.set_index('Month&Year', inplace=True)
    # Sort the DataFrame by 'Month&Year'
    df.sort_index(inplace=True)

    # Plotting trend
    plt.figure(figsize=(12, 6))
    plt.plot(df['Price'], label='Original Series')
    plt.title(f'Trend Plot for {df["Spices"].iloc[0]}')
    plt.xlabel('Month&Year')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Seasonality plot (line chart)
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df.index.month, y=df['Price'], ci=None)
    plt.title(f'Seasonality Plot for {df["Spices"].iloc[0]}')
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.show()

##### RANDOM WALK TEST #####

for df, table_name in zip(loaded_data_list, table_names):

    # Fit AutoReg model with lag order 1
    model = sm.tsa.AutoReg(df['Price'], lags=1)
    results = model.fit()

    # Print summary of the model
    print(f"Summary for {table_name}")
    print(results.summary())

    # Plot the actual vs. predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'], label='Actual Series')
    plt.plot(df.index, results.predict(), label='Predicted Series', color='red')
    plt.title(f'AutoReg Model for {table_name}')
    plt.xlabel('Month&Year')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


##### Stationary Test ADF #####
for df, table_name in zip(loaded_data_list, table_names):

    # Perform Augmented Dickey-Fuller test
    result = adfuller(df['Price'])

    # Print the test result
    print(f"Augmented Dickey-Fuller Test for {table_name}")
    print(f'Test Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values: {result[4]}')

    # Check if the series is stationary
    if result[1] <= 0.05:
        print(f"The series for {table_name} is likely stationary.")
    else:
        print(f"The series for {table_name} is likely non-stationary.")

    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(df['Price'], label='Original Series')
    plt.title(f'Time Series Plot for {table_name}')
    plt.xlabel('Month&Year')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

