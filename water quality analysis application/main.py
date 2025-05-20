import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Load and prepare data
def load_and_prepare_data():
    df = pd.read_excel('Water level.xlsx', sheet_name='2022')
    df.columns = ['DATE'] + [f'M{i}' for i in range(1, 6)]
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month
    df['QUARTER'] = df['DATE'].dt.quarter
    df['MONTH_NAME'] = df['DATE'].dt.month_name()
    melted_df = df.melt(id_vars=['DATE', 'YEAR', 'MONTH', 'QUARTER', 'MONTH_NAME'],
                       value_vars=[f'M{i}' for i in range(1, 6)],
                       var_name='METER', value_name='USAGE')
    melted_df['USAGE'] = melted_df['USAGE'].replace(0, np.nan)
    return df, melted_df

df, melted_df = load_and_prepare_data()

# 1. Enhanced Yearly Analysis with Alerts
def yearly_analysis_with_alerts(melted_df):
    print("\n" + "="*50)
    print("YEARLY ANALYSIS & ALERTS")
    print("="*50)

    for year in melted_df['YEAR'].unique():
        year_data = melted_df[melted_df['YEAR'] == year]
        yearly_usage = year_data.groupby('METER')['USAGE'].sum().sort_values(ascending=False)

        print(f"\nYear {year} Summary:")
        print("-"*30)
        print(yearly_usage.to_string())

        # Yearly alerts
        most_used = yearly_usage.idxmax()
        least_used = yearly_usage.idxmin()
        print(f"\nALERT: In {year}, {most_used} was the most used meter ({yearly_usage.max():,.0f} Klts)")
        print(f"ALERT: In {year}, {least_used} was the least used meter ({yearly_usage.min():,.0f} Klts)")
        print(f"RECOMMENDATION: Consider redistributing some load from {most_used} to {least_used}")

        # Monthly analysis within year
        monthly_usage = year_data.groupby(['MONTH_NAME', 'METER'])['USAGE'].sum().unstack()
        peak_month = monthly_usage.sum(axis=1).idxmax()
        low_month = monthly_usage.sum(axis=1).idxmin()

        print(f"\nMonthly Patterns in {year}:")
        print(f"Peak usage month: {peak_month}")
        print(f"Lowest usage month: {low_month}")

        # Visualize yearly data
        plt.figure(figsize=(12, 6))
        sns.barplot(data=year_data, x='MONTH_NAME', y='USAGE', hue='METER',
                   order=['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December'])
        plt.title(f'Monthly Water Usage by Meter in {year}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

yearly_analysis_with_alerts(melted_df)

# 2. Seasonal Prediction with Alerts
def seasonal_prediction_with_alerts(df):
    print("\n" + "="*50)
    print("SEASONAL PREDICTION & ALERTS")
    print("="*50)

    for meter in [f'M{i}' for i in range(1, 6)]:
        print(f"\nAnalyzing {meter}...")
        meter_data = df[['DATE', meter]].set_index('DATE')
        meter_data = meter_data.interpolate()

        # Time series decomposition
        decomposition = seasonal_decompose(meter_data, model='additive', period=12)
        seasonal_component = decomposition.seasonal

        # Get seasonal peaks
        monthly_seasonal = seasonal_component.groupby(seasonal_component.index.month).mean()
        peak_month_num = monthly_seasonal.idxmax()
        peak_month_name = datetime(2020, peak_month_num, 1).strftime('%B')
        peak_strength = monthly_seasonal.max()

        print(f"\nSeasonal Pattern for {meter}:")
        print(f"Peak usage typically occurs in {peak_month_name}")
        print(f"Seasonal strength: {peak_strength:.2f}")

        # Train prediction model
        def prepare_features(data):
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            data['year'] = data.index.year
            data['lag1'] = data[meter].shift(1)
            data['lag12'] = data[meter].shift(12)
            return data.dropna()

        model_data = prepare_features(meter_data.copy())
        X = model_data.drop(columns=[meter])
        y = model_data[meter]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Predict next year
        last_date = model_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')

        future_data = pd.DataFrame(index=future_dates)
        future_data['month'] = future_data.index.month
        future_data['quarter'] = future_data.index.quarter
        future_data['year'] = future_data.index.year

        # Recursive forecasting
        predictions = []
        last_known = model_data[meter].iloc[-1]
        last_known_12 = model_data[meter].iloc[-12] if len(model_data) >= 12 else last_known

        for date in future_dates:
            future_data.loc[date, 'lag1'] = last_known
            future_data.loc[date, 'lag12'] = last_known_12

            pred = model.predict(future_data.loc[[date]])[0]
            predictions.append(pred)

            last_known_12 = last_known if len(predictions) == 1 else predictions[-2]
            last_known = pred

        # Create prediction dataframe
        forecast = pd.DataFrame({
            'DATE': future_dates,
            'PREDICTION': predictions,
            'MONTH': future_dates.month,
            'MONTH_NAME': future_dates.strftime('%B')
        })

        # Identify peak prediction month
        peak_pred_month = forecast.loc[forecast['PREDICTION'].idxmax(), 'MONTH_NAME']
        peak_pred_value = forecast['PREDICTION'].max()

        print(f"\nPREDICTION ALERT for {meter}:")
        print(f"Highest predicted usage in {peak_pred_month} ({peak_pred_value:.0f} Klts)")
        print("Recommended actions:")
        print("- Check for potential leaks before this period")
        print("- New Admissions")

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(model_data.index, model_data[meter], label='Historical')
        plt.plot(forecast['DATE'], forecast['PREDICTION'], label='Forecast', color='red')
        plt.scatter(forecast['DATE'], forecast['PREDICTION'], color='red')
        plt.title(f'12-Month Usage Forecast for {meter}\nPeak predicted in {peak_pred_month}')
        plt.legend()
        plt.show()

seasonal_prediction_with_alerts(df)

# 3. Comprehensive Meter Efficiency Analysis
def meter_efficiency_analysis(melted_df):
    print("\n" + "="*50)
    print("METER EFFICIENCY ANALYSIS")
    print("="*50)

    # Calculate usage statistics
    meter_stats = melted_df.groupby('METER')['USAGE'].agg(['mean', 'sum', 'std', 'count'])
    meter_stats['utilization'] = meter_stats['sum'] / meter_stats['sum'].sum()
    meter_stats['efficiency'] = 1 - (meter_stats['std'] / meter_stats['mean'])
    meter_stats['usage_rank'] = meter_stats['sum'].rank(ascending=False)

    print("\nMeter Efficiency Statistics:")
    print(meter_stats.sort_values('usage_rank').to_string())

    # Identify under/over utilized meters
    avg_utilization = meter_stats['utilization'].mean()
    over_utilized = meter_stats[meter_stats['utilization'] > avg_utilization * 1.5]
    under_utilized = meter_stats[meter_stats['utilization'] < avg_utilization * 0.5]

    print("\nALERT: Over-utilized Meters (consider load balancing):")
    print(over_utilized[['sum', 'utilization']].to_string())

    print("\nALERT: Under-utilized Meters (potential for load sharing):")
    print(under_utilized[['sum', 'utilization']].to_string())


    # Visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(x=meter_stats.index, y='utilization', data=meter_stats)
    plt.axhline(avg_utilization, color='red', linestyle='--', label='Average Utilization')
    plt.title('Water Meter Utilization Analysis')
    plt.ylabel('Utilization Percentage')
    plt.legend()
    plt.show()

meter_efficiency_analysis(melted_df)