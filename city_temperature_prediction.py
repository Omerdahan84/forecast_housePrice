import pandas as pd
import plotly.express as px
import numpy as np

import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import sklearn
from polynomial_fitting import PolynomialFitting


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    threshold = 2.5
    X = pd.read_csv(filename, parse_dates=['Date']).dropna()
    X = X[X.Temp > 0]
    # X = remove_outliers(X, threshold)
    X['DayOfYear'] = X['Date'].dt.dayofyear
    return X


def remove_outliers(X:pd.DataFrame, threshold:float) ->pd.DataFrame:
    # remove data points suspected as Fahrenheit
    # Handle invalid data (e.g., drop rows with missing 'Date' or 'Temperature')
    mean = X['Temp'].mean()
    std_dev = X['Temp'].std()
    X['z_score'] = (X['Temp'] - mean) / std_dev
    X = X[np.abs(X['z_score']) < threshold].copy()
    X.drop(columns='z_score', inplace=True)  # Drop the z_score column
    return X


def filter_country_data(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Filter the dataset to contain only samples from Israel.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The filtered dataframe containing only samples from Israel.
    """
    return df[df['Country'] == country]


def plot_temperature_trend_by_day(df: pd.DataFrame):
    """
    Plot a scatter plot of average daily temperature as a function of the 'DayOfYear',
    color-coded by year with a discrete color scale.

    Args:
        df (pd.DataFrame): The input dataframe.
    """
    # Extract year from the 'Date' column
    df['Year'] = df['Date'].dt.year - 0

    # Create a color map for each year
    unique_years = df['Year'].unique()
    color_mapping = {
        year: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for
        i, year in enumerate(unique_years)}

    # Plot using Plotly
    fig = go.Figure()

    for year in unique_years:
        year_data = df[df['Year'] == year]
        fig.add_trace(
            go.Scatter(x=year_data['DayOfYear'], y=year_data['Temp'], mode='markers',
                       marker_color=color_mapping[year], name=str(year)))

    # Update layout for better visualization
    fig.update_layout(title='Average Daily Temperature in Israel by Day of Year',
                      xaxis_title='Day of Year',
                      yaxis_title='Temperature (°C)',
                      legend_title='Year',
                      legend=dict(title='Year', yanchor="top", y=0.99, xanchor="left",
                                  x=0.01,
                                  itemsizing='constant', traceorder='reversed',
                                  tracegroupgap=10))

    # Save the figure as an image
    fig.show()


def plot_temperature_trend_by_month(df):
    # Calculate the standard deviation of temperature for each month
    std_temp_by_month = df.groupby('Month')['Temp'].std().reset_index()

    # Plot using Plotly
    fig = px.bar(std_temp_by_month, x='Month', y='Temp',
                 title='STD of Temperature in Each Month',
                 labels={'Month': 'Month', 'Temp': 'STD of temperature'})

    fig.show()

#TODO: chhck what needed,calculated meand but in the form written avg
def plot_avg_per_month_country(df):
    # Group by 'Country' and 'Month', calculate average and standard deviation of temperature
    grouped_data = df.groupby(['Country', 'Month'])['Temp'].agg(
        ['mean', 'std']).reset_index()
    # Plot using Plotly Express
    fig = px.line(grouped_data, x='Month', y='mean', color='Country',
                  error_y='std', title='Average Monthly Temperature by Country',
                  labels={'mean': 'Average Temperature (°C)', 'Month': 'Month'},
                  color_discrete_sequence=px.colors.qualitative.Bold)

    # Show plot
    # fig.show()


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    X = load_data("city_temperature.csv")
    # Question 3 - Exploring data for specific country
    df_israel = filter_country_data(X, "Israel")
    plot_temperature_trend_by_day(df_israel)
    plot_temperature_trend_by_month(df_israel)
    # Question 4 - Exploring differences between countries
    plot_avg_per_month_country(X)
    # Question 5 - Fitting model for different values of `k`
    X_sample, y_sample = df_israel['DayOfYear'], df_israel.Temp

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_sample,
                                                                                y_sample,
                                                                                test_size=0.25,
                                                                                random_state=9)
    losses = []
    for k in range(1, 11):
        pf = PolynomialFitting(k)
        pf.fit(X_train.values, y_train.values)
        loss = pf.loss(X_test.values, y_test.values)
        losses.append(loss)
        print(f"For polynomial with {k} degree, the MSE val is :{loss}")
    # Plotting the bar plot using Plotly
    fig = px.bar(x=list(range(1, 11)), y=losses)
    fig.update_layout(
        title='Test Error for Different Values of k',
        xaxis_title='k',
        yaxis_title='Test Error'
    )
    fig.show()
    # Question 6 - Evaluating fitted model on different countries
    other_countries = X['Country'].unique()
    other_countries = other_countries[other_countries != 'Israel']
    CHOSEN_K = 5
    country_errors = []
    pf = PolynomialFitting(CHOSEN_K)
    pf.fit(X_train.to_numpy(), y_train.to_numpy())
    for country in other_countries:
        df_country = filter_country_data(X, country)
        X_country = df_country['DayOfYear']
        y_country = df_country['Temp']
        error = pf.loss(X_country.to_numpy(), y_country.to_numpy())
        country_errors.append({'Country': country, 'Error': error})

    # Create a DataFrame for the errors
    errors_df = pd.DataFrame(country_errors)
    # Plot a bar plot showing the model's error for each country using Plotly
    unique_countries = errors_df['Country'].unique()
    color_palette = px.colors.qualitative.Plotly  # Using Plotly's qualitative color palette
    color_map = {country: color_palette[i % len(color_palette)] for i, country in
                 enumerate(unique_countries)}

    # Plot a bar plot showing the model's error for each country using Plotly
    fig = px.bar(errors_df, x='Country', y='Error',
                 title=f'Model Error for Each Country Using k={CHOSEN_K} '
                       f'fitted over Israel',
                 labels={'Country': 'Country', 'Error': 'loss'},
                 color='Country',
                 color_discrete_map=color_map,
                 text_auto=True)
    # Update layout for better visualization
    fig.update_layout(
        xaxis=dict(tickmode='linear'),
        yaxis=dict(title='Model Error'),
        title=dict(x=0.5)
    )

    # Show plot
    fig.show()

