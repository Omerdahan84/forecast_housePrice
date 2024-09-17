import sklearn
import pandas as pd
import numpy as np
from typing import NoReturn
from sklearn.model_selection import train_test_split
from linear_regression import *
import plotly.express as px
import plotly.graph_objects as go

import os


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """

    # Drop duplicate rows from the DataFrame
    X.drop_duplicates(inplace=True)
    # fill values where missing
    X = fill_missing_data(X)
    # clean the data accoeding to logic
    X, y = clean_train_data(X, y)
    # calc how old was the house when sold
    X['yr_old_sale'] = X['date'].dt.year - X['yr_built']
    # delete the id and date column
    drop_features = ['id', 'date']
    X.drop(drop_features, axis=1, inplace=True)
    # change features
    # make yr_renovated to binary
    X['yr_renovated'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    X.rename(columns={'yr_renovated': 'did_renovated'}, inplace=True)
    return X, y


def fill_missing_data(X):
    # where there is no date add the frequent date
    X['date'] = X['date'].fillna(X['date'].mode()[0])
    X['date'] = pd.to_datetime(X['date'])
    for col in X.columns:
        if X[col].dtype != 'object':
            X[col] = X[col].fillna(X[col].mean())
    return X


def clean_train_data(X: pd.DataFrame, y: pd.Series):
    # Check that prices are valid
    valid_index = y > 0
    y = y[valid_index]
    X = X.loc[valid_index]

    # Define the logical checks
    conditions = [
        X['bedrooms'] >= 0,
        X['bathrooms'] >= 0,
        X['sqft_living'] > 0,
        X['sqft_lot'] > 0,
        X['floors'] > 0,
        X['waterfront'].isin([0, 1]),  # 'waterfront' 1 or 0
        X['view'].isin(range(5)),  # view 0-4
        X['condition'].isin(range(1, 6)),  # condition 1-5
        X['grade'].isin(range(1, 14)),  # grade 1-13
        X['yr_built'].isin(range(2025)),  # Assuming current year is 2024
        X['lat'].between(-90, 90),
        X['long'].between(-180, 180),
        X['sqft_living15'] > 0,
        X['sqft_lot15'] > 0,
        (X['yr_renovated'] == 0) | (X['yr_renovated'] >= X['yr_built']),
        X['sqft_living'] == X['sqft_above'] + X['sqft_basement']

    ]

    # Combine all conditions with logical AND
    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition &= condition

    # Filter the DataFrame
    X = X[combined_condition]
    y = y[X.index]

    return X, y


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    X = fill_missing_data(X)
    # deal with invalid data
    X = fill_invalid_data(X)
    # create new cols
    X['yr_old_sale'] = X['date'].dt.year - X['yr_built']
    X['yr_renovated'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    X.rename(columns={'yr_renovated': 'did_renovated'}, inplace=True)
    # drop features
    drop_features = ['id', 'date']
    X.drop(drop_features, axis=1, inplace=True)
    return X


def fill_invalid_data(X: pd.DataFrame):
    # Define conditions
    conditions = [
        X['bedrooms'] >= 0,
        X['bathrooms'] >= 0,
        X['sqft_living'] > 0,
        X['sqft_lot'] > 0,
        X['floors'] > 0,
        X['waterfront'].isin([0, 1]),  # 'waterfront' 1 or 0
        X['view'].isin(range(5)),  # view 0-4
        X['condition'].isin(range(1, 6)),  # condition 1-5
        X['grade'].isin(range(1, 14)),  # grade 1-13
        X['yr_built'].isin(range(2025)),  # Assuming current year is 2024
        X['lat'].between(-90, 90),
        X['long'].between(-180, 180),
        X['sqft_living15'] > 0,
        X['sqft_lot15'] > 0,
        (X['yr_renovated'] == 0) | (X['yr_renovated'] >= X['yr_built'])
    ]

    # Check conditions and fill invalid values with median
    check_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                  'waterfront', 'view', 'condition', 'grade', 'yr_built', 'lat', 'long',
                  'sqft_living15', 'sqft_lot15', 'yr_renovated']
    for col, condition in zip(check_cols, conditions):
        median_value = X[col].median()
        X[col] = np.where(condition, X[col], median_value)
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    os.makedirs(output_path, exist_ok=True)
    # Calculate the standard deviation of the response
    y_std = y.std()
    for col in X.columns:
        # Calculate the standard deviation of the feature
        x_std = X[col].std()
        # Calculate the covariance between the feature and the response
        covariance = np.cov(X[col], y)[0, 1]
        # Calculate Pearson correlation coefficient
        correlation = covariance / (x_std * y_std)
        # Create scatter plot
        # plt.figure(figsize=(16, 12))
        # plt.scatter(X[col], y, alpha=0.5)
        # plt.title(f'{col} vs Price\nPearson Correlation: {correlation:.2f}')
        # plt.xlabel(col)
        # plt.ylabel('Price')
        # # Save plot to file
        # plt_path = os.path.join(output_path, f'{col}_vs_price.png')
        # plt.savefig(plt_path)
        # plt.close()
        # Create scatter plot using Plotly
        fig = px.scatter(
            x=X[col],
            y=y,
            title=f'{col} vs Price<br>Pearson Correlation: {correlation:.2f}',
            labels={col: col, 'y': 'Price'},
            opacity=0.5
        )

        # Update layout for better visualization
        fig.update_layout(
            xaxis_title=col,
            yaxis_title='Price',
            title_x=0.5,
            width=800,
            height=600
        )
        #png_path = os.path.join(output_path, f'{col}_vs_price.png')
        #fig.write_image(png_path)


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=4)
    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)
    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train, output_path='plots')
    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)
    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # Initialize lists to store results
    percentages = list(range(10, 101))
    p_values = np.arange(10, 101)
    mean_losses = []
    std_losses = []

    # Loop through increasing percentages
    for p in percentages:
        losses = []
        # Repeat the process 10 times
        for _ in range(10):
            # Sample p% of the overall training data

            # taking  a sample of p  precent from the
            sampled_X_train = X_train.sample(frac=p / 100)
            sampled_y_train = y_train.loc[sampled_X_train.index]
            # Fit linear model
            LR = LinearRegression(include_intercept=False)
            LR.fit(sampled_X_train.values, sampled_y_train.values)
            # Calculate loss
            loss = LR.loss(X_test.values, y_test.values)
            losses.append(loss)

        # Store average and variance of loss
        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)


    # Create the plotly figure
    # Create the trace for mean loss
    # Create the trace for mean loss
    # Create figure
    fig = go.Figure()

    # Add mean loss trace
    fig.add_trace(
        go.Scatter(x=percentages, y=mean_losses, mode='lines', name='Mean Loss'))

    # Add confidence interval
    upper_bound = np.array(mean_losses) + 2 * np.array(std_losses)
    lower_bound = np.array(mean_losses) - 2 * np.array(std_losses)
    fig.add_trace(go.Scatter(x=list(percentages) + list(percentages)[::-1],
                             y=list(upper_bound) + list(lower_bound)[::-1],
                             fill='toself', fillcolor='rgba(0,100,80,0.2)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name='Confidence Interval'))

    # Update layout
    fig.update_layout(title='Mean Loss as a Function of Training Size Percentage',
                      xaxis_title='Percentage of Training Data',
                      yaxis_title='Mean Loss')

    # Show figure
    fig.show()

