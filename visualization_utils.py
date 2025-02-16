import matplotlib.pyplot as plt

def plot_monthly_sales(df_monthly):
    """
     Construction of sales dynamics by months.
    : param df_monthly: Dataframe with data on the month of sales
    """
    plt.figure(figsize=(12, 8))
    plt.plot(df_monthly.index, df_monthly['Total_Sales'], marker='o', linestyle='-', label='Total Sales')
    plt.title('Sales Dynamics by Month')
    plt.xlabel('Date')
    plt.ylabel('Sales Amount')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sales_forecast(df_monthly, df_future):
    """
     Building a graph of actual sales and forecasting.
    : param df_monthly: Dataframe with actual sales
    : param df_future: Dataframe with sales forecast
    """
    plt.figure(figsize=(12, 8))
    plt.plot(df_monthly.index, df_monthly['Total_Sales'], marker='o', linestyle='-', label='Actual Sales')
    plt.plot(df_future['Date'], df_future['Predicted_Sales'], marker='o', linestyle='--', label='Forecast')
    plt.title('Actual Sales and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales Amount')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_category_distribution(df):
    """
     Construction of a circular chart for sales distribution by categories.
    : param df: dataframe with sales data
    """
    category_data = df.groupby('Category')['Total_Sales'].sum()
    plt.figure(figsize=(8, 8))
    plt.pie(category_data, labels=category_data.index, autopct='%1.1f%%')
    plt.title('Sales Distribution by Category')
    plt.show()
