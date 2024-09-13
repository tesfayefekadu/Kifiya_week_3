import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calculate_descriptive_statistics(df):
    numerical_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured']  # add more numerical columns if needed
    return df[numerical_cols].describe()

# We will inspect the data types of each column to ensure categorical variables, dates, and numerical fields are appropriately represented.
def inspect_data_types(df):
    return df.dtypes

# We'll check for any missing values in the dataset.
def check_missing_values(df):
    return df.isnull().sum()


#####  Univariate Analysis

# Function to identify numerical and categorical columns
def identify_column_types(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'datetime']).columns.tolist()
    return numerical_cols, categorical_cols

# Univariate Analysis: Histograms for Numerical Columns
def identify_column_types(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numerical_cols, categorical_cols

# Univariate Analysis: Histograms for Numerical Columns
def plot_numerical_distributions(df):
    numerical_cols, _ = identify_column_types(df)
    df[numerical_cols].hist(figsize=(10, 8))
    plt.tight_layout()
    plt.show()

# Univariate Analysis: Bar charts for Categorical Columns
def plot_categorical_distributions(df):
    _, categorical_cols = identify_column_types(df)
    for col in categorical_cols:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, data=df)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()

# Bivariate and Multivariate Analysis
def plot_correlation(df):
    correlation_cols = ['TotalPremium', 'TotalClaims', 'PostalCode']
    sns.pairplot(df[correlation_cols], diag_kind='kde')
    plt.show()

    # Compute correlation matrix
    corr_matrix = df[correlation_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()    


    #   Data Comparison Trends Over Geography  

def compare_trends_by_geography(df):
    # Aggregating by ZipCode and CoverType
    geography_trends = df.groupby(['PostalCode', 'CoverType']).agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum',
        'make': 'count'  # Example trend for car make
    }).reset_index()

    sns.barplot(x='PostalCode', y='TotalPremium', hue='CoverType', data=geography_trends)
    plt.title('Total Premium by Postal Code and Cover Type')
    plt.xticks(rotation=45)
    plt.show()  

######    Outlier Detection    
def detect_outliers(df):
    numerical_cols = ['TotalPremium', 'TotalClaims', 'SumInsured']  # Add more if necessary
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, y=col)
        plt.title(f'Outliers in {col}')
        plt.show()

####    Visualization

# Creative visualization 1: Heatmap of correlations
def plot_heatmap(df):
    correlation_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm']
    corr_matrix = df[correlation_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.show()

# Creative visualization 2: Premium Distribution by CoverType
def premium_distribution_by_covertype(df):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='CoverType', y='TotalPremium', data=df, inner='quartile')
    plt.title('Premium Distribution by Cover Type')
    plt.show()

# Creative visualization 3: Total Claims over time by CoverType
def claims_trend_over_time(df):
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    monthly_claims = df.groupby(['TransactionMonth', 'CoverType']).agg({'TotalClaims': 'sum'}).reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='TransactionMonth', y='TotalClaims', hue='CoverType', data=monthly_claims)
    plt.title('Claims Trend Over Time by Cover Type')
    plt.xticks(rotation=45)
    plt.show()


