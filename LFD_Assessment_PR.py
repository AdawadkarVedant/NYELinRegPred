import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
available_features = ['open']
target_feature = ['close']
for available_feature in ['open', 'low', 'high']:
    for symbol in symbols:
        print('For Symbol - ', symbol)
        security_wise_df = filtered_se_df_cleaned[filtered_se_df_cleaned['symbol'].eq(symbol)]
        security_wise_df.sort_values(by='date', ascending=True, inplace=True)
        x = security_wise_df[available_feature]
        y = security_wise_df[target_feature]

        # Split the data into train/test data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)

        poly = PolynomialFeatures(degree=20)
        X_train_poly = poly.fit_transform(x_train)

        poly.fit(X_train_poly, y_train)
        poly_reg = LinearRegression()
        poly_reg.fit(X_train_poly, y_train)

        X_test_poly = poly.fit_transform(x_test)
        predictions = poly_reg.predict(X_test_poly)

        R2_train = poly_reg.score(X_train_poly, y_train)
        R2_test = poly_reg.score(X_test_poly, y_test)

        print('R2_train:', R2_train)
        print('R2_test:', R2_test)

        print('Slope:', poly_reg.coef_)
        print('Intercept:', poly_reg.intercept_)

        # Evaluate performance using Mean Absolute Error and/or Root Mean Squared Error for each selected sector
        print('Mean Absolute Error - ', metrics.mean_absolute_error(y_test, predictions))
        print('Root Mean Squared Error - ', metrics.mean_squared_error(y_test, predictions))

        R2_train = poly_reg.score(X_train_poly, y_train)
        R2_test = poly_reg.score(X_test_poly, y_test)

        print('R2_train:', R2_train)
        print('R2_test:', R2_test)

        prices_test = x_test.copy()
        prices_test['Actual_Close'] = y_test
        prices_test['Predicted_Close'] = predictions
        prices_test['Actual_Predicted_Diff'] = prices_test['Predicted_Close'] - prices_test['Actual_Close'].shift(1)
        prices_test.at[0, 'Actual_Predicted_Diff'] = None
        print(prices_test.head())

        # Plot using scatterplot with regression line
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=prices_test, x='Actual_Close', y='Actual_Predicted_Diff')
        sns.regplot(data=prices_test, x='Actual_Close', y='Actual_Predicted_Diff', scatter=False, color='red',
                    line_kws={'label': 'Regression Line'})
        plt.title(f'Actual vs Predicted Closing Values with Regression Line for {symbol} Security for {available_feature}')
        plt.xlabel('Actual Close Price ($)')
        plt.ylabel('Actual Predicted Difference ')
        plt.grid()
        plt.legend()
        plt.show()
        # plt.savefig(f'Linear Regression plot for {symbol} security', bbox_inches='tight')
        plt.close()

