import matplotlib.pyplot as plt

def plot_monthly_sales(df_monthly):
    """
    Побудова графіка динаміки продажів по місяцях.
    :param df_monthly: DataFrame з даними про продажі по місяцях
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
    Побудова графіка фактичних продажів і прогнозу.
    :param df_monthly: DataFrame з фактичними продажами
    :param df_future: DataFrame з прогнозом продажів
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
    Побудова кругової діаграми для розподілу продажів за категоріями.
    :param df: DataFrame з даними про продажі
    """
    category_data = df.groupby('Category')['Total_Sales'].sum()
    plt.figure(figsize=(8, 8))
    plt.pie(category_data, labels=category_data.index, autopct='%1.1f%%')
    plt.title('Sales Distribution by Category')
    plt.show()
