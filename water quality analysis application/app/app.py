from flask import Flask, render_template, request, redirect, url_for
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
import os
app = Flask(__name__)

@app.route('/' , methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/utility', methods=['GET','POST'])
def utility():
    return render_template('utility.html')

@app.route('/softening', methods=['GET','POST'])
def softening():
    return render_template('softening.html')

@app.route('/drinking', methods=['GET','POST'])
def drinking():
    return render_template('drinking.html')

@app.route('/dialysis', methods=['GET','POST'])
def dialysis():
    return render_template('dialysis.html')

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

@app.route('/utility/quantity', methods=['GET', 'POST'])
def util_quantity():
    df, melted_df = load_and_prepare_data()
    years = sorted(melted_df['YEAR'].unique())
    selected_year = None
    alerts = []
    plot_path = None

    if request.method == 'POST':
        selected_year = request.form.get('year')
        if selected_year:
            selected_year = int(selected_year)
            filtered_df = melted_df[melted_df['YEAR'] == selected_year]

            yearly_usage = filtered_df.groupby('METER')['USAGE'].sum().sort_values(ascending=False)
            most_used = yearly_usage.idxmax()
            least_used = yearly_usage.idxmin()

            alerts.append(f"In {selected_year}, {most_used} was the most used meter ({yearly_usage.max():,.0f} Klts)")
            alerts.append(f"In {selected_year}, {least_used} was the least used meter ({yearly_usage.min():,.0f} Klts)")
            alerts.append(f"Recommendation: Consider redistributing some load from {most_used} to {least_used}")

            monthly_usage = filtered_df.groupby(['MONTH_NAME', 'METER'])['USAGE'].sum().unstack()
            peak_month = monthly_usage.sum(axis=1).idxmax()
            low_month = monthly_usage.sum(axis=1).idxmin()

            alerts.append(f"Peak usage month: {peak_month}")
            alerts.append(f"Lowest usage month: {low_month}")

            # Plot
            plt.figure(figsize=(12, 6))
            sns.barplot(data=filtered_df, x='MONTH_NAME', y='USAGE', hue='METER',
                        order=['January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December'])
            plt.title(f'Monthly Water Usage by Meter in {selected_year}')
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_path = 'static/usage_plot.png'
            plt.savefig(plot_path)
            plt.close()

    return render_template('utility-quantity.html', years=years, selected_year=selected_year, alerts=alerts, plot_path=plot_path)

@app.route('/utility/quality', methods=['GET','POST'])
def util_quality():
    return render_template('utility-quality.html')




if __name__ == '__main__':
    app.run(debug=True)