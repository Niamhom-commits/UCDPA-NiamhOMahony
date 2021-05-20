# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
prind_df=pr_df.set_index('Registered Charity Number')
arind_df=ar_df.set_index('Registered Charity Number')

# check number of nulls
prind_df.isnull().sum()
arind_df.isnull().sum()
# AR get rid of the empty columns
arind_df=arind_df.dropna(axis =1, thresh = 10000)

