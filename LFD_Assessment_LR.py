import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt


def read_csv_files(file_name):
    dir_name = 'Assets//'
    return pd.read_csv(dir_name + file_name)


pd.set_option('display.max_columns', None)

# Read CSV files and create a dataframe
prices_df = read_csv_files('prices-split-adjusted.csv')
securities_df = read_csv_files('securities.csv')
fundamentals_df = read_csv_files('fundamentals.csv')

# Rename the column in securities_df and fundamentals_df so that symbol columns match for merging
securities_df = securities_df.rename(columns={'Ticker symbol': 'symbol'})
fundamentals_df = fundamentals_df.rename(columns={'Ticker Symbol': 'symbol'})

# Merge the two dataframes so securities information and fundamentals information is available in prices df
securities_prices_df = pd.merge(prices_df, securities_df, on='symbol', how='left')
stock_exchange_df = pd.merge(securities_prices_df, fundamentals_df, on='symbol', how='left')

filtered_se_df = stock_exchange_df[
    ['date', 'symbol', 'open', 'close', 'high', 'low', 'volume', 'Security', 'GICS Sector', 'GICS Sub Industry',
     'Date first added', 'Period Ending', 'After Tax ROE', 'Gross Profit', 'Gross Margin', 'Profit Margin',
     'Total Revenue', 'For Year', 'Earnings Per Share']]

# Data Preprocessing
# Convert date column to datetime column
filtered_se_df['date'] = pd.to_datetime(filtered_se_df['date'])
filtered_se_df['Date first added'] = pd.to_datetime(filtered_se_df['Date first added'])
filtered_se_df['Period Ending'] = pd.to_datetime(filtered_se_df['Period Ending'])

# Identify duplicate values if any
has_duplicate_rows = filtered_se_df.duplicated().any()

# Identify Null or NAN values if any and drop those rows
has_null_values = filtered_se_df.isnull().values.any()
null_column_list = filtered_se_df.columns[filtered_se_df.isnull().any()].tolist()
filtered_se_df = filtered_se_df.sort_values(by=['symbol', 'date'])
filtered_se_df['Date first added'] = filtered_se_df.groupby('symbol')['date'].transform('min')
filtered_se_df['Period Ending'] = filtered_se_df.groupby('symbol')['date'].transform('max')
filtered_se_df['For Year'] = filtered_se_df.groupby('symbol')['Period Ending'].transform(lambda row: row.dt.year)

filtered_se_df_cleaned = filtered_se_df.dropna(
    subset=['After Tax ROE', 'Gross Profit', 'Gross Margin', 'Profit Margin', 'Total Revenue', 'Earnings Per Share'],
    how='all')

grouped_se_df = filtered_se_df_cleaned.groupby(['GICS Sector', 'GICS Sub Industry'])
mean_values = grouped_se_df['Earnings Per Share'].mean()
filtered_se_df_cleaned = filtered_se_df_cleaned.merge(mean_values, on=['GICS Sector', 'GICS Sub Industry'],
                                                      suffixes=('', '_mean'))
filtered_se_df_cleaned['Earnings Per Share'].fillna(filtered_se_df_cleaned['Earnings Per Share_mean'], inplace=True)
filtered_se_df_cleaned.drop(columns=['Earnings Per Share_mean'], inplace=True)

new_grouped_se_df = filtered_se_df_cleaned.groupby(['GICS Sector'])
mean_values = new_grouped_se_df['Earnings Per Share'].mean()
filtered_se_df_cleaned = filtered_se_df_cleaned.merge(mean_values, on=['GICS Sector'], suffixes=('', '_mean'))
filtered_se_df_cleaned['Earnings Per Share'].fillna(filtered_se_df_cleaned['Earnings Per Share_mean'], inplace=True)
filtered_se_df_cleaned.drop(columns=['Earnings Per Share_mean'], inplace=True)

# Start Linear Regression
# Decide which sectors we want to study
symbols = ['MSFT', 'ATVI', 'EA']

# Filter DataFrame by sectors
available_features = ['open', 'low', 'high', 'volume', 'After Tax ROE', 'Gross Profit', 'Gross Margin', 'Profit Margin',
                      'Total Revenue', 'Earnings Per Share']
target_feature = ['close']

for symbol in symbols:
    security_wise_df = filtered_se_df_cleaned[filtered_se_df_cleaned['symbol'].eq(symbol)]
    security_wise_df.sort_values(by='date', ascending=True, inplace=True)
    x = security_wise_df[available_features]
    y = security_wise_df[target_feature]

    # Split the data into train/test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)

    # Create a model for regression model
    LinReg = LinearRegression()

    # Perform Recursive Feature Elimination on the available features
    rfe = RFE(LinReg, n_features_to_select=3)
    rfe.fit(x_train, y_train)

    # Get selected features from RFE model
    selected_features = x.columns[rfe.support_]

    # Train new model with selected features
    LinReg.fit(x_train[selected_features], y_train)

    # For each selected security predict the target variable
    train_predictions = LinReg.predict(x_train[selected_features])
    test_predictions = LinReg.predict(x_test[selected_features])
    test_mae = metrics.mean_absolute_error(y_test, test_predictions)
    train_mae = metrics.mean_absolute_error(y_train, train_predictions)
    test_rmse = metrics.mean_squared_error(y_test, test_predictions)
    train_rmse = metrics.mean_squared_error(y_train, train_predictions)
    R2_train = LinReg.score(x_train[selected_features], y_train)
    R2_test = LinReg.score(x_test[selected_features], y_test)

    print('Slope:', LinReg.coef_)
    print('Intercept:', LinReg.intercept_)

    # Evaluate performance using Mean Absolute Error and/or Root Mean Squared Error for each selected sector
    print('Mean Absolute Error For Test Set - ', test_mae)
    print('Mean Absolute Error For Training Set - ', train_mae)

    print('Root Mean Squared Error For Test Set - ', test_rmse)
    print('Root Mean Squared Error For Training Set - ', train_rmse)

    print('R2_train:', R2_train)
    print('R2_test:', R2_test)

    prices_test = x_test.copy()
    prices_test['Actual_Close'] = y_test
    prices_test['Predicted_Close'] = test_predictions
    prices_test['Actual_Predicted_Diff'] = prices_test['Predicted_Close'] - prices_test['Actual_Close'].shift(1)
    prices_test.at[0, 'Actual_Predicted_Diff'] = None

    # Plot using scatterplot with regression line
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=prices_test, x='Actual_Close', y='Actual_Predicted_Diff')
    sns.regplot(data=prices_test, x='Actual_Close', y='Actual_Predicted_Diff', scatter=False, color='red',
                line_kws={'label': 'Regression Line'})
    plt.title(f'Actual vs Predicted Closing Values with Regression Line for {symbol} Security')
    plt.xlabel('Actual Close Price ($)')
    plt.ylabel('Actual Predicted Difference ')
    plt.grid()
    plt.legend(labels=['Actual Close Values', 'Regression Line'])
    plt.savefig(f'Linear Regression plot for {symbol} security.jpeg', bbox_inches='tight')
    plt.clf()
    plt.close()

    # Create a bar plot for MAE
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Train', 'Test'], y=[train_mae, test_mae])
    plt.title(f'Mean Absolute Error Comparison between Train and Test for {symbol} Security')
    plt.ylabel('Mean Absolute Error')
    plt.grid()
    plt.legend()
    plt.annotate('Mean Absolute Error For Test Set - ', test_mae)
    plt.annotate('Mean Absolute Error For Training Set - ', train_mae)
    plt.savefig(f'Mean Absolute Error Bar Plot for {symbol} security.jpeg', bbox_inches='tight')
    plt.clf()
    plt.close()

    # Create a bar plot for MAE
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Train', 'Test'], y=[train_rmse, test_rmse])
    plt.title(f'Root Mean Squared Error Comparison between Train and Test for {symbol} Security')
    plt.ylabel('Root Mean Squared Error')
    plt.grid()
    plt.legend()
    print()
    plt.savefig(f'Root Mean Squared Error Bar Plot for {symbol} security.jpeg', bbox_inches='tight')
    plt.clf()
    plt.close()

    # Create a bar plot for R-squared
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Train', 'Test'], y=[R2_train, R2_test])
    plt.title(f'R-squared Comparison between Train and Test for {symbol} Security')
    plt.ylabel('R-squared')
    plt.ylim(0, 1)  # Adjust the y-axis limits for R-squared
    plt.grid()
    plt.legend()
    plt.savefig(f'R-squared Bar Plot for {symbol} security.jpeg', bbox_inches='tight')
    plt.clf()
    plt.close()

