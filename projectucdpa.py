# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:19:50 2021

@author: omaho
"""
# UCD PA Project Cert Introductory Data Analytics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print('Hello')

# Import Benefacts csv file as dataframe
filebenefacts = "C:\\Users\\omaho\\downloads\\CHARITIES20210507130928csv.csv"
bf_df = pd.read_csv(filebenefacts)
# print(bf_df.head())
# set benefacts id as index
bf_dfind = bf_df.set_index('Benefacts Id')
bf_dfind.head()

# check how many nas in each column
# print(bf_dfind.isnull().sum())

# Drop duplicates
bf_dfind.drop_duplicates(inplace=True)

# Large number of almost empty columns - delete these
bf_dfind.dropna(axis=1, thresh=1000, inplace=True)
print(bf_dfind.shape)

# Remove all na rows.
bfacts_df = bf_dfind.dropna(how='all')
print(bfacts_df.shape)
# determine where nulls are now
print(bfacts_df.isnull().sum()/bfacts_df.shape[0]*100)
# Dictionary of Replacements for na
dictreplace = {'County':'Not given', 'Registered Address': 'Not given', ' CRO':'Not Registered', 'CHY':0, 'DES':'Not Registered', 'otherName1':'N/A'}#replace nulls with NaN 
# Replace nas
final_df = bfacts_df.fillna(dictreplace)
# Make sure CRA column has no text
final2_df=final_df[final_df['CRA'].str.contains('Deregistered') == False]
# pd.to_numeric(final2_df['CRA'])
print(final2_df.loc[:, ['CRA']])
# final2_df['CHY']=pd.to_numeric(final2_df['CHY'], errors='coerce')
final2_df['CRA']=final2_df['CRA'].astype(float)
print(final2_df.isnull().sum())
# Import Charity regulator excel sheets as 2 dataframes
file = "C:\\Users\\omaho\\downloads\\public-register-03052021.xlsx"
char_xls = pd.ExcelFile(file)
print(char_xls.sheet_names)
char_df1 = char_xls.parse('Public Register', skiprows=1)
char_df2 = char_xls.parse('Annual Reports', skiprows=1)
# print(char_df1.head())
# print(char_df2.head()) 
# Clean char_df2
char_df2.drop_duplicates(inplace=True)
# set index for char_df2
char_df2ind = char_df2.set_index('Period End Date')
# drop blank columns
char_df2ind.dropna(axis=1, thresh=10000, inplace=True)
print(char_df2ind.columns)
# Get the latest Period for income form Char_dfind
finalchar_df=char_df2ind.loc['2019-01-01':'2019-12-31']
print(finalchar_df.head())
#change Registered Charity Number to int32
# finalchar_df = finalchar_df.astype({col: 'int32' for col in finalchar_df.select_dtypes('int64').columns})
 #merge 2 dataframes
merged_df = final2_df.merge(finalchar_df, how='right', left_on='CRA', right_on='Registered Charity Number', suffixes=('_ben', '_cr'))

print(merged_df.head())



merged_df.groupby('County') ['Subsector Name'].value_counts()
 