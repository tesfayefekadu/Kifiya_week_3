import pandas as pd
import scipy.stats as stats
from statsmodels.stats.weightstats import ztest

# T-test to compare mean TotalClaims (risk) across two provinces
def t_test_risk_provinces(df):
    group_a = df[df['Province'] == 'Gauteng']['TotalClaims']
    group_b = df[df['Province'] == 'Western Cape']['TotalClaims']
    
    t_stat, p_value = stats.ttest_ind(group_a, group_b, nan_policy='omit')
    return p_value
# T-test to compare mean TotalClaims (risk) between zip codes
def t_test_risk_zipcodes(df):
    group_a = df[df['PostalCode'] == 2410]['TotalClaims']
    group_b = df[df['PostalCode'] == 1709]['TotalClaims']
    
    t_stat, p_value = stats.ttest_ind(group_a, group_b, nan_policy='omit')
    return p_value
def t_test_margin_zipcodes(df):
    group_a = df[df['PostalCode'] == 2410]['Margin']
    group_b = df[df['PostalCode'] == 1709]['Margin']
    
    t_stat, p_value = stats.ttest_ind(group_a, group_b, nan_policy='omit')
    return p_value
def t_test_risk_gender(df):
    group_a = df[df['Gender'] == 'Male']['TotalClaims']
    group_b = df[df['Gender'] == 'Female']['TotalClaims']
    
    t_stat, p_value = stats.ttest_ind(group_a, group_b, nan_policy='omit')
    return p_value