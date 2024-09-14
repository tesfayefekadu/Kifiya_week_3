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

# 1. Handling Missing Values
def handle_missing_values(df):
    # Drop columns with more than 50% missing values (adjust threshold as needed)
    df_cleaned = df.dropna(thresh=len(df) * 0.5, axis=1)
    
    # Fill missing values in numerical columns with the median
    numerical_cols, categorical_cols = identify_column_types(df_cleaned)
    df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(df_cleaned[numerical_cols].median())
    
    # Fill missing values in categorical columns with mode
    df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna(df_cleaned[categorical_cols].mode().iloc[0])
    
    return df_cleaned

# 2. Correcting Data Types
def correct_data_types(df):
    # Convert TransactionMonth to datetime
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    
    # Convert numerical fields that might be incorrectly treated as objects
    numeric_fields = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
    for col in numeric_fields:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce invalid parsing to NaN
    
    return df

# 3. Removing Duplicates
def remove_duplicates(df):
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    return df_cleaned

# 4. Outlier Treatment (optional, using IQR method)
def handle_outliers(df):
    numerical_cols, _ = identify_column_types(df)
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        # Define outliers as 1.5 times the IQR
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    
    return df
def clean_data(df):
    df = handle_missing_values(df)
    df = correct_data_types(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    
    return df
# Save the cleaned data to a CSV file
def save_cleaned_data_csv(df, file_name):
    df.to_csv(file_name, index=False)


