import pandas as pd
#import os

from ai_utils import analyze_sales_data
from ml_utils import train_model, forecast_future_sales
from flask import Flask, request, jsonify
from visualization_utils import plot_monthly_sales, plot_sales_forecast, plot_category_distribution

app = Flask(__name__)

# Download data
df = pd.read_csv('sales_2024.csv')
df['Date'] = pd.to_datetime(df['Date'])
df_monthly = df.resample('ME', on='Date').sum()
df_monthly['Month_Number'] = range(1, len(df_monthly) + 1)

# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("API key is not set. Please configure it as an environment variable.")

plot_monthly_sales(df_monthly)  # Construction of sales dynamics graphics
plot_category_distribution(df)  # Constructing a circular chart by category

# Навчання моделі та прогноз
model, mae = train_model(df_monthly)
df_future = forecast_future_sales(model, df_monthly)

plot_sales_forecast(df_monthly, df_future)  # Construction of a prognosis schedule



#  API route to analyze
@app.route('/analyze', methods=['POST'])
def analyze(api_key):
    try:
        api_key = request.json.get('api_key')
        if not api_key:
            return jsonify({"error": "API key is required"}), 400

        description, gpt_analysis = analyze_sales_data(df_monthly, api_key)
        return jsonify({"description": description, "gpt_analysis": gpt_analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#  API route for forecasting
@app.route('/forecast', methods=['GET'])
def forecast():
    try:
        model, mae = train_model(df_monthly)
        future_forecast = forecast_future_sales(model, df_monthly)
        return jsonify({
            "mean_absolute_error": mae,
            "future_forecast": future_forecast.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)