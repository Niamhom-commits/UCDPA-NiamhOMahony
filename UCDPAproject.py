# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import charity regulators data as an Excel 2 sheet spreadsheet

file = "public-register-03052021.xlsx"
charity_df = pd.ExcelFile(file)

# get the sheet names
print(charity_df.sheet_names)

# split into 2 dfs
pr_df = charity_df.parse('Public Register', skiprows=1)
ar_df = charity_df.parse('Annual Reports', skiprows=1)
print(pr_df.head())
print(ar_df.head()) 

# look at columns and incices in both dataframes
print(pr_df.columns)
print(ar_df.columns)
print(pr_df.index)
print(ar_df.index)

# AR(Annual Reports) Check median and mean for Gross Income w Numpy allowing for Nan values
income = ar_df['Financial: Gross Income']
np_income= np.array(income)
income_mean = np.nanmean(np_income)
income_median = np.nanmedian(np_income)

print(income_mean) 
print(income_median)                    


# Set indexes to Registered Charity Number for pr_df
# and Period End date for ar_df
prind_df=pr_df.set_index('Registered Charity Name')
arind_df=ar_df.set_index('Period End Date')


# custom function to print percent of nulls in each column
def percent_null (df1):
    percent=df1.isnull().sum()*100/len(df1)
    return percent

print(percent_null(prind_df))
print(percent_null(arind_df))
    
# Drop duplicates    
prind_df.drop_duplicates(inplace=True)  
arind_df.drop_duplicates(inplace=True)  
 
# Clean prind_df:
   
# replace nulls with Dictionary replacements
dictpr = {'Primary Address': 'Not given', 'CRO Number':'Not Registered', 'CHY Number': 'Not Registered', 'Charitable Purpose': 'Not given', 'Charitable Objects': 'Not given'} 
prind_df=prind_df.fillna(dictpr)

# Clean arind_df:
# AR get rid of the empty columns where threshold is breached
arind_df=arind_df.dropna(axis =1, thresh = 10000)
print(arind_df.shape)

# Where financial numeric data is missing replace with the median value.
arind_df['Financial: Gross Income'].fillna(arind_df['Financial: Gross Income'].median(), inplace=True)
arind_df['Financial: Gross Expenditure'].fillna(arind_df['Financial: Gross Expenditure'].median(), inplace=True)

# In arind_df select report dates in 2019 only

ar19ind_df=arind_df.loc['2019-01-01':'2019-12-31']
print(ar19ind_df.head())
print(ar19ind_df.shape)

# Change index of ar19ind_df to 'Registered Charity Name' to allow merge

ar19ind_df=ar19ind_df.set_index('Registered Charity Name')

# Tidy up column names and delete unneccessary columns

prind_df.drop(['Also Known As', 'Primary Address', 'Country Established', 'Charitable Purpose', 'Charitable Objects'], axis=1, inplace=True)
ar19ind_df.drop(['Report Size', 'Period Start Date', 'Report Activity', 'Activity Description', 'Beneficiaries'], axis=1, inplace=True)

# Merge these 2 dataframes: right join on index

creg_df=prind_df.merge(ar19ind_df, how = 'right', left_index=True, right_index=True)


# Collect extra data from Benefatcts csv

filebenefacts = "CHARITIES20210507130928.csv"
bf_df = pd.read_csv(filebenefacts)


# Create new df from csv with useful columns
bfnew_df=bf_df[['Subsector Name', 'County','CRA']].copy()

# Clean this df
# check for nulls using custom function again
print(percent_null(bfnew_df))

# Drop duplicates
bfnew_df.drop_duplicates(inplace=True)

# replace nulls with appropriate values using Dictionary
bfdict = {'Subsector Name': 'Not available', 'County':'Not known'}
bfnew_df=bfnew_df.fillna(bfdict)

# Delete columns with no CRA (Charity Regulator No)
bfnew_df=bfnew_df.dropna(axis=0, subset=['CRA'])

#Check datatypes for bfnew_df
print(bfnew_df.dtypes)

# CRA needs to be converted to a float
# Remove data with str as part of CRA which would cause a problem
 
bfnew_df=bfnew_df[bfnew_df['CRA'].str.contains('Deregistered') == False]

# Convert CRA to float
bfnew_df['CRA']=bfnew_df['CRA'].astype(float)

# set bf index to CRA
bfind_df=bfnew_df.set_index('CRA')


#Rename creg columns for easier manipulation
creg_df=creg_df.rename(columns={'Registered Charity Number_x' : 'RCN'})



#Merge cleaned Benefacts data into charity register =inner join

ch_df=creg_df.merge(bfind_df, how="inner", left_on="RCN",right_index=True)


# Insert a new column in ch_df which confirms if the charity has a CHY number or not

for lab, row in ch_df.iterrows():
       if "registered" in str(row['CHY Number']).lower():
           ch_df.loc[lab, 'CHY'] = "No"
       else:
           ch_df.loc[lab, 'CHY'] = "Yes"
 

# start investigating data

sns.set_style('whitegrid')
sns.set_palette('hls',8)


# first visualistaion - County by Number of registered charities
#g = sns.countplot(x="County", data=ch_df, hue='CHY', hue_order=['Yes','No'])

ax = sns.countplot(x="County", data=ch_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="right")
ax.set_title('Number of Registered Charities Per County')
ax.set_ylabel('Number of Charities')

plt.tight_layout()
plt.show()
plt.clf()
plt.close()

# Governing Forms numbers - second visualisation
ax= sns.countplot(x='Governing Form', data=ch_df, hue='CHY', hue_order=['Yes','No'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_title('Charity Governing Forms - Number in each category')
plt.tight_layout()
ax.set_ylabel('No of charities')
plt.show()
plt.clf()
plt.close()

# governing form mean income
sns.barplot(data=ch_df, x='Governing Form', y='Financial: Gross Income')
plt.title('Governing Form - Average Gross Income')
plt.ylabel('Gross Income €10m')
plt.xticks(rotation=90)
plt.show()
plt.clf()
plt.close()



# Scatter plot to show relationship between Gross Income and Expenditure
g=sns.relplot(kind='scatter', data=ch_df, x='Financial: Gross Income', y='Financial: Gross Expenditure', style='CHY', hue='Number of Volunteers')
g.fig.suptitle('Gross Income vs Gross Expenditure grouped by CHY + Num Volunteers')
g.set(xlabel='Gross Income €10m', ylabel='Gross Expenditure €10m')    
plt.xticks(rotation=90)
plt.show()
plt.clf()
plt.close()

# stripplot to look at distribution of gross income by county
sns.stripplot(data=ch_df, x='County', y='Financial: Gross Income')
plt.title('Gross Income per charity by County')
plt.xticks(rotation=90)
plt.ylabel('Gross Income €10m')
plt.show()
plt.clf()
plt.close()



# histogram to show distribution of Income
plt.hist(ch_df['Financial: Gross Income'])
plt.title('Gross Income Distribution € - All Charities')
plt.show()
plt.clf()
plt.close()

# Not a clear view - skewed by large numbers. Look at smaller sample
view_df=ch_df[ch_df['Financial: Gross Income'] < 1000000]
plt.hist(view_df['Financial: Gross Income'])
plt.title('Gross Income Distribution - Charities earning under €1m')
plt.show()
plt.clf()
plt.close()




# Top 500 charities sorted by subsector
top500_df=ch_df.sort_values('Financial: Gross Income', ascending=False).iloc[0:500,:]
sns.countplot(data=top500_df, x='Subsector Name')
plt.xlabel('Subsector')
plt.ylabel('No of charities')
plt.xticks(rotation=90)
plt.title('Top 500 charities by income sorted by Subsector')
plt.show()
plt.clf()
plt.close()




# Lineplot of Gross Income against Gross Expenditure showing confidence
sns.lmplot(data=ch_df, x='Financial: Gross Income', y='Financial: Gross Expenditure')
plt.title('Lineplot of Gross Income against Gross Expenditure showing confidence')
plt.xlabel('Gross Income €10m')
plt.ylabel('Gross Income €10m')
plt.show()
plt.clf()
plt.close()







