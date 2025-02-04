from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

def train_model(df_monthly):
    """
    Навчання моделі для прогнозування.
    :param df_monthly: DataFrame з даними про продажі
    :return: Навчена модель, середня помилка
    """
    X = df_monthly[['Month_Number']]
    y = df_monthly[['Total_Sales']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return model, mae

def forecast_future_sales(model, df_monthly, months=6):
    """
    Прогнозує продажі на майбутні місяці.
    :param model: Навчена модель
    :param df_monthly: DataFrame із даними про продажі
    :param months: Кількість місяців для прогнозу
    :return: DataFrame з майбутніми прогнозами
    """
    last_date = df_monthly.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months, freq='M')
    future_month_numbers = np.arange(len(df_monthly) + 1, len(df_monthly) + months + 1)
    future_sales = model.predict(pd.DataFrame({'Month_Number': future_month_numbers}))
    return pd.DataFrame({'Date': future_dates, 'Predicted_Sales': future_sales.flatten()})
