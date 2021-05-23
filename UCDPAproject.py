# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import charity regulators data in xl

file = "C:\\Users\\omaho\\downloads\\public-register-03052021.xlsx"
charity_df = pd.ExcelFile(file)

# get the sheet names
print(charity_df.sheet_names)

# split into 2 dfs
pr_df = charity_df.parse('Public Register', skiprows=1)
ar_df = charity_df.parse('Annual Reports', skiprows=1)
print(pr_df.head())
print(ar_df.head()) 

# Set indexes to Registered Charity Number in both
prind_df=pr_df.set_index('Registered Charity Name')
arind_df=ar_df.set_index('Period End Date')

# check number of nulls
prind_df.isnull().sum()
arind_df.isnull().sum()


# Clean prind_df
dictpr = {'Primary Address': 'Not given', 'CRO Number':'Not Registered', 'CHY Number': 'Not Registered', 'Charitable Purpose': 'Not given', 'Charitable Objects': 'Not given'} 
prind_df=prind_df.fillna(dictpr)

# Clean arind df
# AR get rid of the empty columns
arind_df=arind_df.dropna(axis =1, thresh = 10000)
print(arind_df.shape)

# Drop rows with no Financial Data
arind_df=arind_df[arind_df['Financial: Gross Income'].notna()]
arind_df=arind_df[arind_df['Financial: Gross Expenditure'].notna()]

# select report dates in 2019 only

ar19ind_df=arind_df.loc['2019-01-01':'2019-12-31']
print(ar19ind_df.head())
print(ar19ind_df.shape)

# Change index of Annual Report df to Registered Charity Name
ar19ind_df=ar19ind_df.set_index('Registered Charity Name')

# Merge 2 dataframes

creg_df=prind_df.merge(ar19ind_df, how = 'right', left_index=True, right_index=True)


# take county data from Benefatcts csv

filebenefacts = "C:\\Users\\omaho\\downloads\\CHARITIES20210507130928csv.csv"
bf_df = pd.read_csv(filebenefacts)


# Create new df with required columns
bfnew_df=bf_df[['Subsector Name', 'County','CRA']].copy()

# Clean this df
bfnew_df.isnull().sum()
# bfnew_df=bf_df[['Subsector Name', 'County','CRA']].copy()
bfdict = {'Subsector Name': 'Not available', 'County':'Not known'}
bfnew_df=bfnew_df.fillna(bfdict)
# Delete columns with no CRA 
bfnew_df=bfnew_df.dropna()

# CRA type needs to be float
bfnew_df=bfnew_df[bfnew_df['CRA'].str.contains('Deregistered') == False]
bfnew_df['CRA']=bfnew_df['CRA'].astype(float)
creg_df=creg_df.rename(columns={'Registered Charity Number_x' : 'RCN'})

#Merge benefacts data into charity register

ch_df=creg_df.merge(bfnew_df, how="inner", left_on="RCN",right_on="CRA")


ch_df
