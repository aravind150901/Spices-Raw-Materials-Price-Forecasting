import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import pmdarima as pm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


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

# Create an empty dictionary to store forecast values for respective spices using Auto ARIMA
forecast_dict_auto_arima = {}

for table_name in table_names:
    # SQL query to select all columns from the specified table
    sql_query = f'SELECT * FROM {table_name}'
    
    # Load the DataFrame from SQL
    loaded_data = pd.read_sql_query(sql_query, con=engine)
    
    # Set 'Month&Year' as the index
    loaded_data.set_index('Month&Year', inplace=True)
    
    # Sort the DataFrame by 'Month&Year'
    loaded_data.sort_index(inplace=True)
    
    # Assuming 'Month&Year' is the index and 'Price' is the target variable
    # Use the entire dataset for training
    train = loaded_data
    
    # Auto ARIMA model
    model = pm.auto_arima(train['Price'], seasonal=False, m=12, stepwise=True)
    
    # Forecast future values
    forecast_periods = 6  # Forecast for the next 6 months
    forecast_index = pd.date_range(start=train.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
    forecast = model.predict(n_periods=forecast_periods)
    
    # Store forecast values in the dictionary with the spice name as the key
    forecast_dict_auto_arima[table_name] = forecast
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Price'], label='Historical Data', color='blue')
    plt.plot(forecast_index, forecast, label='Forecast', color='red', linestyle='dashed', marker='o')
    plt.title(f'Forecasting for {table_name} (Auto ARIMA)')
    plt.xlabel('Month&Year')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    print(f'Forecast for {table_name} (Auto ARIMA): {forecast}')


# Create an empty dictionary to store MAPE values for respective spices using Holt's Method
mape_dict_holts = {}

# Function to train Holt's method (Double Exponential Smoothing) model and make predictions
def train_holts_method_model(df, table_name):
    # Assuming 'Month&Year' is the index and 'Price' is the target variable
    train = df
    
    # Train Holt's method model
    model = ExponentialSmoothing(train['Price'], trend='add').fit()

    # Forecast for the next 6 months
    forecast_periods = 6
    forecast_index = pd.date_range(start=train.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
    forecast = model.forecast(forecast_periods)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Price'], label='Historical Data', color='blue')
    plt.plot(forecast_index, forecast, label='Forecast', color='red', linestyle='dashed', marker='o')
    plt.title(f'Forecasting for {table_name} (Holt\'s Method)')
    plt.xlabel('Month&Year')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return forecast

# Train Holt's method model for each spice
for table_name in table_names:
    # SQL query to select all columns from the specified table
    sql_query = f'SELECT * FROM {table_name}'
    
    # Load the DataFrame from SQL
    loaded_data = pd.read_sql_query(sql_query, con=engine)
    
    # Set 'Month&Year' as the index
    loaded_data.set_index('Month&Year', inplace=True)
    
    # Sort the DataFrame by 'Month&Year'
    loaded_data.sort_index(inplace=True)
    
    forecast_holts = train_holts_method_model(loaded_data, table_name)
    
    # Store forecast values in the dictionary with the spice name as the key
    mape_dict_holts[table_name] = forecast_holts
    
    print(f'Forecast for {table_name} (Holt\'s Method): {forecast_holts}')

# Print the forecasts for respective spices using Holt's Method
for spice, forecast_values in mape_dict_holts.items():
    print(f'Forecast for {spice} (Holt\'s Method): {forecast_values}')


##### EXP MOVING AVG #####

# Create an empty dictionary to store forecast DataFrames for respective spices using Exponential Moving Average
forecast_dict_ema = {}

# Function to train Exponential Moving Average (EMA) model and make predictions
def train_exponential_moving_average_model(df, table_name, alpha=0.1, forecast_periods=6):
    y = df['Price']

    # Calculate the exponential moving average over the training period
    exp_moving_avg = y.ewm(alpha=alpha, adjust=False).mean()

    # Forecast the next 6 months
    forecast_index = pd.date_range(start=y.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
    forecast = exp_moving_avg.iloc[-1] * (1 + alpha) ** np.arange(1, forecast_periods + 1)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(y.index, y, label='Historical Data', color='blue')
    plt.plot(forecast_index, forecast, label=f'Exponential Moving Average (Alpha: {alpha})', color='green', linestyle='dashed', marker='o')
    plt.title(f'Forecasting for {table_name} (Exponential Moving Average)')
    plt.xlabel('Month&Year')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Combine forecasted values with their respective months
    forecast_df_ema = pd.DataFrame({'Month&Year': forecast_index, 'Forecast': forecast})

    return forecast_df_ema

# Train Exponential Moving Average model for each spice
for table_name in table_names:
    # SQL query to select all columns from the specified table
    sql_query = f'SELECT * FROM {table_name}'

    # Load the DataFrame from SQL
    loaded_data = pd.read_sql_query(sql_query, con=engine)

    # Set 'Month&Year' as the index
    loaded_data.set_index('Month&Year', inplace=True)

    # Sort the DataFrame by 'Month&Year'
    loaded_data.sort_index(inplace=True)

    forecast_df_ema = train_exponential_moving_average_model(loaded_data, table_name)

    # Store forecast DataFrame in the dictionary with the spice name as the key
    forecast_dict_ema[table_name] = forecast_df_ema

    print(f'Forecast for {table_name} (Exponential Moving Average):\n{forecast_df_ema}')

# Print the forecast DataFrames for respective spices using Exponential Moving Average
for spice, forecast_df in forecast_dict_ema.items():
    print(f'Forecast for {spice} (Exponential Moving Average):\n{forecast_df}')


####### HOLT WINTER'S MULTI SEASONALITY #######
# Create an empty dictionary to store forecast DataFrames for respective spices using Holt-Winters
forecast_dict_hw = {}

# Function to train Holt-Winters' method model and make predictions
def train_forecast_hw(df, table_name, forecast_periods=6):
    try:
        # Extract the target variable
        y = df['Price'] 

        # Fit the Holt-Winters' model on the entire dataset
        model = ExponentialSmoothing(y, seasonal='multiplicative', seasonal_periods=12, initialization_method='estimated')
        model_fit = model.fit()

        # Make predictions for the next 6 months
        predictions = model_fit.predict(start=len(y), end=len(y) + forecast_periods - 1)
        
        
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(y.index, y, label='Historical Data')
        plt.plot(predictions.index, predictions, label='Holt-Winters Forecast')
        plt.title(f'Holt-Winters Forecast for {table_name}')
        plt.xlabel('Month&Year')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        # Combine forecasted values with their respective months
        forecast_index = pd.date_range(start=y.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
        forecast_df_hw = pd.DataFrame({'Month&Year': forecast_index, 'Forecast': predictions})

        return forecast_df_hw

    except Exception as e:
        print(f"Error in forecasting for {table_name}: {str(e)}")
        return None

# Train and forecast for each spice
for table_name in table_names:
    # SQL query to select all columns from the specified table
    sql_query = f'SELECT * FROM {table_name}'
    
    # Load the DataFrame from SQL
    loaded_data = pd.read_sql_query(sql_query, con=engine)
    
    # Set 'Month&Year' as the index
    loaded_data.set_index('Month&Year', inplace=True)
    
    # Sort the DataFrame by 'Month&Year'
    loaded_data.sort_index(inplace=True)
    
    forecast_df_hw = train_forecast_hw(loaded_data, table_name)

    # Store forecast DataFrame in the dictionary with the spice name as the key
    forecast_dict_hw[table_name] = forecast_df_hw

    print(f'Forecast for {table_name} (Holt-Winters):\n{forecast_df_hw}')

# Print the forecast DataFrames for respective spices using Holt-Winters
for spice, forecast_df_hw in forecast_dict_hw.items():
    print(f'Forecast for {spice} (Holt-Winters):\n{forecast_df_hw}')
    
    
    
# Create an empty dictionary to store forecasts for respective spices using SES
forecast_dict_ses = {}

# Function to train Simple Exponential Smoothing model and make predictions
def train_ses_model(df, table_name, alpha=0.2, forecast_periods=6):
    try:
        # Assuming 'Month&Year' is the index and 'Price' is the target variable
        # Train Simple Exponential Smoothing model on the entire dataset
        model = SimpleExpSmoothing(df['Price']).fit(smoothing_level=alpha)

        # Forecast for the next 6 months after the last month in training data
        forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
        forecast = model.forecast(forecast_periods)

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Price'], label='Actual', color='blue')
        plt.plot(forecast_index, forecast, label=f'Forecast (Alpha={alpha})', color='red')
        plt.title(f'Forecasting for {table_name} (Simple Exponential Smoothing)')
        plt.xlabel('Month&Year')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        # Combine forecasted values with their respective months
        forecast_df_ses = pd.DataFrame({'Month&Year': forecast_index, 'Forecast': forecast})

        # Store forecast DataFrame in the dictionary with the spice name as the key
        forecast_dict_ses[table_name] = forecast_df_ses

        return forecast_df_ses

    except Exception as e:
        print(f"Error in forecasting for {table_name}: {str(e)}")
        return None

# Train and forecast for each spice
for table_name in table_names:
    # SQL query to select all columns from the specified table
    sql_query = f'SELECT * FROM {table_name}'
    
    # Load the DataFrame from SQL
    loaded_data = pd.read_sql_query(sql_query, con=engine)
    
    # Set 'Month&Year' as the index
    loaded_data.set_index('Month&Year', inplace=True)
    
    # Sort the DataFrame by 'Month&Year'
    loaded_data.sort_index(inplace=True)
    
    forecast_ses = train_ses_model(loaded_data, table_name)

# Print the forecast DataFrames for respective spices using SES
for spice, forecast_ses in forecast_dict_ses.items():
    print(f'Forecast for {spice} (SES):\n{forecast_ses}')


###### SARIMA  #######


# Create an empty dictionary to store forecasts for respective spices using SARIMA
forecast_dict_sarima = {}

# Function to train SARIMA model and make predictions
def train_sarima_model(df, table_name, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), forecast_periods=6):
    try:
        # Assuming 'Month&Year' is the index and 'Price' is the target variable
        # Train SARIMA model on the entire dataset
        model = sm.tsa.SARIMAX(df['Price'], order=order, seasonal_order=seasonal_order)
        results = model.fit()

        # Forecast for the next 6 months after the last month in the dataset
        forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
        forecast = results.get_forecast(steps=forecast_periods).predicted_mean

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Price'], label='Actual', color='blue')
        plt.plot(forecast_index, forecast, label=f'Forecast (Order={order}, Seasonal Order={seasonal_order})', color='green')
        plt.title(f'Forecasting for {table_name} (SARIMA)')
        plt.xlabel('Month&Year')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        # Combine forecasted values with their respective months
        forecast_df_sarima = pd.DataFrame({'Month&Year': forecast_index, 'Forecast': forecast})

        # Store forecast DataFrame in the dictionary with the spice name as the key
        forecast_dict_sarima[table_name] = forecast_df_sarima

        return forecast_df_sarima

    except Exception as e:
        print(f"Error in forecasting for {table_name} (SARIMA): {str(e)}")
        return None

# Train and forecast for each spice
for table_name in table_names:
    # SQL query to select all columns from the specified table
    sql_query = f'SELECT * FROM {table_name}'
    
    # Load the DataFrame from SQL
    loaded_data = pd.read_sql_query(sql_query, con=engine)
    
    # Set 'Month&Year' as the index
    loaded_data.set_index('Month&Year', inplace=True)
    
    # Sort the DataFrame by 'Month&Year'
    loaded_data.sort_index(inplace=True)
    
    # SARIMA order and seasonal order (you may need to tune these)
    sarima_order = (1, 1, 1)
    sarima_seasonal_order = (1, 1, 1, 12)
    
    forecast_sarima = train_sarima_model(loaded_data, table_name, order=sarima_order, seasonal_order=sarima_seasonal_order)

# Print the forecast DataFrames for respective spices using SARIMA
for spice, forecast_sarima in forecast_dict_sarima.items():
    print(f'Forecast for {spice} (SARIMA):\n{forecast_sarima}')

# Define the forecasting function based on the selected model
def forecast_selected_model(model_name, loaded_data, table_name):
    forecast_index = pd.Index([])
    forecast = []
    try:
        if model_name == 'Auto ARIMA':
            # Auto ARIMA forecasting code
            model = pm.auto_arima(loaded_data['Price'], seasonal=False, m=12, stepwise=True)
            forecast_periods = 6
            forecast_index = pd.date_range(start=loaded_data.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
            forecast = model.predict(n_periods=forecast_periods)

        elif model_name == 'Holt\'s Method':
            # Holt's Method forecasting code
            model = ExponentialSmoothing(loaded_data['Price'], trend='add').fit()
            forecast_periods = 6
            forecast_index = pd.date_range(start=loaded_data.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
            forecast = model.forecast(forecast_periods)

        elif model_name == 'Exponential Moving Average':
            # Exponential Moving Average forecasting code
            forecast_df_ema = train_exponential_moving_average_model(loaded_data, table_name)
            forecast = forecast_df_ema['Forecast'].values
            forecast_index = forecast_df_ema['Month&Year']

        elif model_name == 'Holt-Winters':
            # Holt-Winters forecasting code
            forecast_df_hw = train_forecast_hw(loaded_data, table_name)
            forecast = forecast_df_hw['Forecast'].values
            forecast_index = forecast_df_hw['Month&Year']

        elif model_name == 'Simple Exponential Smoothing':
            # Simple Exponential Smoothing forecasting code
            forecast_ses = train_ses_model(loaded_data, table_name)
            forecast = forecast_ses['Forecast'].values
            forecast_index = forecast_ses['Month&Year']

        elif model_name == 'SARIMA':
            # SARIMA forecasting code
            sarima_order = (1, 1, 1)
            sarima_seasonal_order = (1, 1, 1, 12)
            forecast_df_sarima = train_sarima_model(loaded_data, table_name, order=sarima_order, seasonal_order=sarima_seasonal_order)
            forecast = forecast_df_sarima['Forecast'].values
            forecast_index = forecast_df_sarima['Month&Year']

        else:
            st.warning("Invalid model selection")

    except Exception as e:
        print(f"Error in forecasting for {table_name} ({model_name}): {str(e)}")

    print("Length of forecast_index:", len(forecast_index))
    print("Length of forecast:", len(forecast))
    return forecast_index, forecast

# Streamlit app
st.title("Spice Price Forecasting App")

# Dropdown for selecting the spice
selected_spice = st.selectbox("Select Spice", table_names)

# Dropdown for selecting the forecasting model
selected_model = st.selectbox("Select Forecasting Model", ['Auto ARIMA', 'Holt\'s Method', 'Exponential Moving Average', 'Holt-Winters', 'Simple Exponential Smoothing', 'SARIMA'])

# Load the data for the selected spice
sql_query = f'SELECT * FROM {selected_spice}'
loaded_data = pd.read_sql_query(sql_query, con=engine)
loaded_data.set_index('Month&Year', inplace=True)
loaded_data.sort_index(inplace=True)

# Perform forecasting based on the selected model
forecast_index, forecast_values = forecast_selected_model(selected_model, loaded_data, selected_spice)


plt.figure(figsize=(12, 6))
plt.plot(loaded_data.index, loaded_data['Price'], label='Historical Data', color='blue')
plt.plot(forecast_index, forecast_values, label='Forecast', color='red', linestyle='dashed', marker='o')
plt.title(f'Forecasting for {selected_spice} ({selected_model})')
plt.xlabel('Month&Year')
plt.ylabel('Price')
plt.legend()
st.pyplot()


# Display the forecast values
st.write(f'Forecast for {selected_spice} ({selected_model}): {forecast_values}')
