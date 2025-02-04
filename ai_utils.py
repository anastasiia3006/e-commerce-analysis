import openai

def analyze_sales_data(df_monthly, api_key, model="gpt-3.5-turbo"):
    """ Analyzes sales data and calls Openai GPT to create a text report.
        : param api_Key: API-key for access to Openai.
        : Param Model: GPT model for data analysis (default "GPT-3.5-Turbo").
        : Return: Sales Analysis Text Report"""

    openai.api_key = api_key

    max_sales = df_monthly['Total_Sales'].max()
    min_sales = df_monthly['Total_Sales'].min()
    max_date = df_monthly[df_monthly['Total_Sales'] == max_sales].index[0]
    min_date = df_monthly[df_monthly['Total_Sales'] == min_sales].index[0]
    start_sales = df_monthly['Total_Sales'].iloc[0]
    end_sales = df_monthly['Total_Sales'].iloc[-1]
    trend = 'growth' if end_sales > start_sales else 'fall'

    description = f"""
    In the period under consideration, the total sales trend demonstrates {trend}.
    Maximum sales volume: {max_sales} on {max_date.strftime('%Y-%m')}
    Minimum sales volume: {min_sales} on {min_date.strftime('%Y-%m')}
    """

    prompt = f"""
    Analyze sales trends:
    Data: {df_monthly['Total_Sales'].tolist()}
    Date: {df_monthly.index.strftime('%Y-%m').tolist()}
    Provide insights and trends.
    """

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        gpt_analysis = response['choices'][0]['message']['content']
    except Exception as e:
        gpt_analysis = f"Error during OpenAI API call: {e}"

    return description, gpt_analysis
