import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

def main():
    # Step 1: Load and preprocess the data
    data = pd.read_csv('sales_data.csv')  # Assuming you have a CSV file with historical sales data
    # Perform any necessary data cleaning and preprocessing steps here
    data['Date'] = pd.to_datetime(data['Date'])
    data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')

    # Step 2: Split the data into training and validation sets
    train_data = data[:-12]  # Assuming you want to use the last 12 months as a validation set
    valid_data = data[-12:]

    # Step 3: Train the SARIMA model
    model = SARIMAX(train_data['Quantity'], order=(2, 0, 1), seasonal_order=(1, 0, 1, 12))  # SARIMA(p, d, q)(P, D, Q, s) order parameters
    model_fit = model.fit()

    # Step 4: Generate forecasts for the validation set
    forecast_values = model_fit.predict(start=valid_data.index[0], end=valid_data.index[-1])

    # Step 5: Evaluate the model's performance
    mse = np.mean((forecast_values - valid_data['Quantity']) ** 2)
    mae = np.mean(np.abs(forecast_values - valid_data['Quantity']))
    rmse = np.sqrt(mse)

    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Step 6: Generate forecasts for future inventory
    last_date = data['Date'].iloc[-1]
    future_index = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=12, freq='D')
    future_forecast = model_fit.get_forecast(steps=12)
    future_values = future_forecast.predicted_mean

    print("Future forecasts:", future_values)

    # Step 7: Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Quantity'], label='Historical Sales')
    plt.plot(valid_data['Date'], valid_data['Quantity'], label='Validation Set')
    plt.plot(future_index, future_values, label='Future Inventory Forecast')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.title('Inventory Forecast')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
