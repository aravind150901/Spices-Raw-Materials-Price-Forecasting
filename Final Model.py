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

for df in loaded_data_list:
    # Set 'Month&Year' as the index
    df.set_index('Month&Year', inplace=True)
    # Sort the DataFrame by 'Month&Year'
    df.sort_index(inplace=True)

# Create an empty dictionary to store MAPE values for respective spices using Auto ARIMA
mape_dict_auto_arima = {}

for df, table_name in zip(loaded_data_list, table_names):
    # Data preparation
    
    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(df['Price'], lags=10, ax=ax1)
    plot_pacf(df['Price'], lags=10, ax=ax2)
    plt.suptitle(f'ACF and PACF Plots for {table_name}')
    plt.show()

    # Assuming 'Month&Year' is the index and 'Price' is the target variable
    train = df.iloc[:-12]  # Use the first 24 months for training
    test = df.iloc[-12:]   # Use the most recent 12 months for testing

    # Auto ARIMA model
    model = pm.auto_arima(train['Price'], seasonal=False, m=12, stepwise=True)
    
    # Forecast
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
    
    # Calculate MAPE
    actual = test['Price'].values
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    # Append the MAPE value to the dictionary with the spice name as the key
    mape_dict_auto_arima[table_name] = mape
    
    print(f'MAPE for {table_name} (Auto ARIMA): {mape:.2f}%')

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Price'], label='Train')
    plt.plot(test.index, actual, label='Actual', color='blue')
    plt.plot(test.index, forecast, label='Forecast', color='red')
    plt.title(f'Forecasting for {table_name} (Auto ARIMA)')
    plt.xlabel('Month&Year')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Function to train Holt's method (Double Exponential Smoothing) model and make predictions
def train_holts_method_model(df, table_name):
    # Assuming 'Month&Year' is the index and 'Price' is the target variable
    train = df.iloc[:-12]  # Use the first 24 months for training
    test = df.iloc[-12:]   # Use the most recent 12 months for testing

    # Train Holt's method model
    model = ExponentialSmoothing(train['Price'], trend='add').fit()

    # Forecast for the test set
    forecast = model.forecast(len(test))

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Price'], label='Train')
    plt.plot(test.index, test['Price'], label='Actual', color='blue')
    plt.plot(test.index, forecast, label='Forecast', color='red')
    plt.title(f'Forecasting for {table_name} (Holt\'s Method)')
    plt.xlabel('Month&Year')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Calculate MAPE for evaluation
    actual = test['Price']
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    return mape

# Create an empty dictionary to store MAPE values for respective spices using Holt's Method
mape_dict_holts = {}

# Train Holt's method model for each spice
for df, table_name in zip(loaded_data_list, table_names):
    mape_holts = train_holts_method_model(df, table_name)
    mape_dict_holts[table_name] = mape_holts
    print(f'MAPE for {table_name} (Holt\'s Method): {mape_holts:.2f}%')

# Print the MAPE values for respective spices using Holt's Method
for spice, mape_value in mape_dict_holts.items():
    print(f'MAPE for {spice} (Holt\'s Method): {mape_value:.2f}%')


####### EXP MOVING AVG ########
def forecast_exponential_moving_average(df, table_name, alpha=0.5):
    y = df['Price']

    train = y.iloc[:-12]
    test = y.iloc[-12:]

    # Calculate the exponential moving average over the training period
    exp_moving_avg = train.ewm(alpha=alpha, adjust=False).mean().iloc[-12:]

    # Calculate MAPE using only the test data
    actual_test = test.values
    predictions_test = exp_moving_avg.values
    mape = np.mean(np.abs((actual_test - predictions_test) / actual_test)) * 100
    print(f'MAPE for {table_name} (Exponential Moving Average): {mape:.2f}%')

    # Plot the results for the test data only
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label='Test Data', color='blue')
    plt.plot(test.index, exp_moving_avg, label=f'Exponential Moving Average (Alpha: {alpha})', color='green')
    plt.title(f'Exponential Moving Average Forecast for {table_name} (MAPE: {mape:.2f}%) - Test Data Only')
    plt.legend()
    plt.show()

    return mape

# Create an empty dictionary to store MAPE values for respective spices using Exponential Moving Average
mape_dict_ema = {}

# Loop through the loaded data list and apply forecasting using Exponential Moving Average
for i, df in enumerate(loaded_data_list):
    table_name = table_names[i]
    
    # Call the forecasting function for each spice table
    mape_ema = forecast_exponential_moving_average(df, table_name)
    
    # Store the MAPE value in the dictionary with the spice name as the key
    mape_dict_ema[table_name] = mape_ema
    
    print(f'MAPE for {table_name} (Exponential Moving Average): {mape_ema:.2f}%')

# Print the MAPE values for respective spices using Exponential Moving Average
for spice, mape_value in mape_dict_ema.items():
    print(f'MAPE for {spice} (Exponential Moving Average): {mape_value:.2f}%')    


###### HOLT WINTER MULTI SEASONALITY ######
def forecast_hw(df, table_name):
    try:
        # Extract the target variable
        y = df['Price'] 

        # Split data into training and testing sets
        train = y.iloc[:-12]  # Use the first N months for training (adjust N to be a multiple of 12)
        test = y.iloc[-12:]   # Use the most recent 12 months for testing

        # Fit the Holt-Winters' model
        model = ExponentialSmoothing(train, seasonal='multiplicative', seasonal_periods=12, initialization_method='estimated')
        model_fit = model.fit()

        # Make predictions for the test set
        predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)

        # Calculate MAPE
        actual = test.values
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        print(f'MAPE for {table_name} (Holt-Winters): {mape:.2f}%')

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(train.index, train, label='Training Data')
        plt.plot(test.index, test, label='Test Data')
        plt.plot(test.index, predictions, label='Holt-Winters Forecast')
        plt.title(f'Holt-Winters Forecast for {table_name} (MAPE: {mape:.2f}%)')
        plt.legend()
        plt.show()

        return mape

    except ValueError as e:
        # Handle the specific error related to initial seasonals
        if "Cannot compute initial seasonals using heuristic method" in str(e):
            print(f"Skipping {table_name} due to insufficient data for heuristic method.")
            return None
        else:
            # Raise the exception if it's a different error
            raise

# Create an empty dictionary to store MAPE values for respective spices using Holt-Winters
mape_dict_hw = {}

# Loop through the loaded data list and apply forecasting
for i, df in enumerate(loaded_data_list):
    table_name = table_names[i]
    
    # Call the forecasting function for each spice table
    mape_hw = forecast_hw(df, table_name)
    
    # Store the MAPE value in the dictionary with the spice name as the key
    if mape_hw is not None:
        mape_dict_hw[table_name] = mape_hw

# Print the MAPE values for respective spices using Holt-Winters
for spice, mape_value in mape_dict_hw.items():
    print(f'MAPE for {spice} (Holt-Winters): {mape_value:.2f}%')
    
    
    ######### MOVING AVERAGE #########
    
def forecast_moving_average(df, table_name, window_size=5):
    # Extract the target variable
    y = df['Price'] 
    
    # Split data into training and testing sets
    train = y.iloc[:-12]  # Use the first N months for training (adjust N based on your preference)
    test = y.iloc[-12:]   # Use the most recent 12 months for testing

    # Calculate the moving average over the training period
    moving_avg = train.rolling(window=window_size).mean().iloc[-12:]

    # Calculate MAPE using only the test data
    actual_test = test.values
    predictions_test = moving_avg.values
    mape = np.mean(np.abs((actual_test - predictions_test) / actual_test)) * 100
    print(f'MAPE for {table_name} (Moving Average): {mape:.2f}%')

    # Plot the results for the test data only
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label='Test Data', color='blue')
    plt.plot(test.index, moving_avg, label=f'Moving Average (Window Size: {window_size})', color='red')
    plt.title(f'Moving Average Forecast for {table_name} (MAPE: {mape:.2f}%) - Test Data Only')
    plt.legend()
    plt.show()

    return mape

# Create an empty dictionary to store MAPE values for respective spices using Moving Average
mape_dict_ma = {}

# Loop through the loaded data list and apply forecasting using Moving Average
for i, df in enumerate(loaded_data_list):
    table_name = table_names[i]
    
    # Call the forecasting function for each spice table
    mape_ma = forecast_moving_average(df, table_name)
    
    # Store the MAPE value in the dictionary with the spice name as the key
    mape_dict_ma[table_name] = mape_ma

# Print the MAPE values for respective spices using Moving Average
for spice, mape_value in mape_dict_ma.items():
    print(f'MAPE for {spice} (Moving Average): {mape_value:.2f}%')


####### SARIMA #######
# Create an empty dictionary to store MAPE values for respective spices
mape_dict_sarima = {}

for df, table_name in zip(loaded_data_list, table_names):
    try:
        # Data preparation

        # Plot ACF and PACF
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(df['Price'], lags=10, ax=ax1)
        plot_pacf(df['Price'], lags=10, ax=ax2)
        plt.suptitle(f'ACF and PACF Plots for {table_name}')
        plt.show()

        # Assuming 'Month&Year' is the index and 'Price' is the target variable
        train = df.iloc[:-12]  # Use the first 24 months for training
        test = df.iloc[-12:]   # Use the most recent 12 months for testing

        # SARIMA model (Example order, you may need to tune this)
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        model = sm.tsa.SARIMAX(train['Price'], order=order, seasonal_order=seasonal_order)
        results = model.fit()

        # Forecast
        forecast = results.get_forecast(steps=len(test))
        forecast_mean = forecast.predicted_mean

        # Calculate MAPE
        actual = test['Price']
        mape = np.mean(np.abs((actual - forecast_mean) / actual)) * 100

        # Append the MAPE value to the dictionary with the spice name as the key
        mape_dict_sarima[table_name] = mape

        print(f'MAPE for {table_name}: {mape:.2f}%')

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train['Price'], label='Train')
        plt.plot(test.index, actual, label='Actual', color='blue')
        plt.plot(test.index, forecast_mean, label='Forecast', color='red')
        plt.title(f'Forecasting for {table_name}')
        plt.xlabel('Month&Year')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Skipping {table_name} due to an error: {e}")
        continue

# Print the MAPE values for respective spices
for spice, mape_value in mape_dict_sarima.items():
    print(f'MAPE for {spice}: {mape_value:.2f}%')
    

####### SIMPLE EXP SMOOTHING ########
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Create an empty dictionary to store MAPE values for respective spices using SES
mape_dict_ses = {}

# Function to train Simple Exponential Smoothing model and make predictions
def train_ses_model(df, table_name, alpha=0.2):
    try:
        # Assuming 'Month&Year' is the index and 'Price' is the target variable
        train = df.iloc[:-12]  # Use the first 24 months for training
        test = df.iloc[-12:]   # Use the most recent 12 months for testing

        # Train Simple Exponential Smoothing model with specified alpha
        model = SimpleExpSmoothing(train['Price']).fit(smoothing_level=alpha)

        # Forecast for the test set
        forecast = model.forecast(len(test))

        # Visualization and evaluation
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train['Price'], label='Train')
        plt.plot(test.index, test['Price'], label='Actual', color='blue')
        plt.plot(test.index, forecast, label=f'Forecast (Alpha={alpha})', color='red')
        plt.title(f'Forecasting for {table_name} (Simple Exponential Smoothing)')
        plt.xlabel('Month&Year')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        # Calculate MAPE for evaluation
        actual = test['Price']
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        print(f'MAPE for {table_name} (Alpha={alpha}): {mape:.2f}%')

        # Store the MAPE value in the dictionary with the spice name as the key
        mape_dict_ses[table_name] = mape

        return mape

    except Exception as e:
        print(f"Skipping {table_name} due to an error: {e}")
        return None

# Loop through the loaded data list and apply forecasting using SES
for df, table_name in zip(loaded_data_list, table_names):
    train_ses_model(df, table_name, alpha=0.2)

# Print the MAPE values for respective spices using SES
for spice, mape_value in mape_dict_ses.items():
    print(f'MAPE for {spice} (Simple Exponential Smoothing): {mape_value:.2f}%')


# Create a dictionary to store MAPE values for all models
all_mape_dicts = {
    'Auto ARIMA': mape_dict_auto_arima,
    'Holt\'s Method': mape_dict_holts,
    'Exponential Moving Average': mape_dict_ema,
    'Holt-Winters': mape_dict_hw,
    'Moving Average': mape_dict_ma,
    'SARIMA': mape_dict_sarima,
    'Simple Exponential Smoothing': mape_dict_ses
}

# Iterate through spices to find the best model for each spice
for spice in table_names:
    spice_mape_dict = {}
    for model_name, mape_dict in all_mape_dicts.items():
        spice_mape_dict[model_name] = mape_dict.get(spice, float('inf'))

    best_model_name = min(spice_mape_dict, key=spice_mape_dict.get)
    best_mape = spice_mape_dict[best_model_name]
    
    # Print the best model, its spice name, and MAPE
    print(f'Best Model for Spice {spice}: {best_model_name} with MAPE: {best_mape:.2f}')

    