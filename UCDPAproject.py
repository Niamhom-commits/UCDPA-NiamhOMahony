# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# look at columns and incices in both dataframes
print(pr_df.columns)
print(ar_df.columns)
print(pr_df.index)
print(ar_df.index)

# AR(Annual Reports) Check median and mean for Gross Income w Numpy
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

# check number of nulls
prind_df.isnull().sum()
arind_df.isnull().sum()


# Clean prind_df:
# replace nulls with Dictionary replacements
dictpr = {'Primary Address': 'Not given', 'CRO Number':'Not Registered', 'CHY Number': 'Not Registered', 'Charitable Purpose': 'Not given', 'Charitable Objects': 'Not given'} 
prind_df=prind_df.fillna(dictpr)

# Clean arind_df:
# AR get rid of the empty columns where threshold is breached
arind_df=arind_df.dropna(axis =1, thresh = 10000)
print(arind_df.shape)

# Drop rows with no Financial Data
arind_df=arind_df[arind_df['Financial: Gross Income'].notna()]
arind_df=arind_df[arind_df['Financial: Gross Expenditure'].notna()]

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

filebenefacts = "C:\\Users\\omaho\\downloads\\CHARITIES20210507130928csv.csv"
bf_df = pd.read_csv(filebenefacts)


# Create new df from csv with useful columns
bfnew_df=bf_df[['Subsector Name', 'County','CRA']].copy()

# Clean this df
# check for nulls
bfnew_df.isnull().sum()

# replace nulls with appropriate values using Dictionary
bfdict = {'Subsector Name': 'Not available', 'County':'Not known'}
bfnew_df=bfnew_df.fillna(bfdict)

# Delete columns with no CRA (Charity Regulator No)
bfnew_df=bfnew_df.dropna()

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

#Create new column - net surplus or deficit
# ch_df['SurpDef'] = ch_df['Financial: Gross Income'] - ch_df['Financial: Gross Expenditure']

# Use for and if loop to determine if deficit or surplus in 2019 add new column

# result = []
# for value in ch_df['SurpDef']:
#    if value >= 0:
#       result.append("Surplus")
#   else:
#      result.append("Deficit")
       
# ch_df["Result"]=result
#

# Insert a new column in ch_df which confirms if the charity has a CHY number or not

for lab, row in ch_df.iterrows():
       if "registered" in str(row['CHY Number']).lower():
           ch_df.loc[lab, 'CHY'] = "No"
       else:
           ch_df.loc[lab, 'CHY'] = "Yes"
 

# start investigating data

# How many charities have a CHY number?
# print(ch_df[ch_df['CHY'] == 'YES'].value_counts())

# How many charities in each subsector
# ch_df.groupby('County') ['Subsector Name'].value_counts()

# How many charities have CHY number and are registered as company
#ch2_df=ch_df[(ch_df['CHY Number'] != 'Not Registered') & (ch_df['CRO Number'] != 'Not Registered')]

#ch_df.groupby('County')['Financial: Gross Income'].mean().plot.bar()



# ar19ind_df.drop(['Report Size', 'Period Start Date', 'Report Activity', 'Activity Description', 'Beneficiaries'], axis=1, inplace=True)


# ch_df['Governing Form'].value_counts().plot.bar()
        

         
         
# ch_df['County'].value_counts().plot.bar()
# first visualistaion - County by Number of registered charities
ax = sns.countplot(x="County", data=ch_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_title('Number of Registered Charities Per County')
ax.set_ylabel('Number of Charities')
plt.tight_layout()
plt.show()
plt.clf()
plt.close()
# Govrening Forms  numbers
ax= sns.countplot(x='Governing Form', data=ch_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_title('Charity Governing Forms Ireland')
plt.tight_layout()
plt.show()
plt.clf()
plt.close()

# Scatter plt finances
sns.scatterplot(ch_df['Financial: Gross Income'], ch_df['Financial: Gross Expenditure'], hue=ch_df['CHY']
, s =15, alpha=0.5)
plt.show()
plt.clf()
plt.close()

# For closer analysis divide charities into groups below and above median


chmed=ch_df['Financial: Gross Income'].median()
belowmedian_df=ch_df[ch_df['Financial: Gross Income'] <= chmed ]
abovemedian_df=ch_df[ch_df['Financial: Gross Income'] > chmed]

sns.scatterplot(data=abovemedian_df, x='Financial: Gross Income', y='Financial: Gross Expenditure', hue='CHY')
plt.show()  
plt.clf()
plt.close()                    
sns.scatterplot(data=belowmedian_df, x='Financial: Gross Income', y='Financial: Gross Expenditure', hue='CHY')
plt.show()
plt.clf()
plt.close()



#plt.clf()
#plt.close()


top50_df=ch_df.sort_values('Financial: Gross Income', ascending=False).iloc[0:50,:]
top50_df['Subsector Name'].value_counts().plot.bar()






plt.clf()
plt.close()
# governing form mean income
sns.barplot(data=ch_df, x='Governing Form', y='Financial: Gross Income')



frequency_df=ch_df['Subsector Name'].value_counts()
frequency_list=frequency_df.index.tolist()
top10_list=frequency_list[0:10]
btm10_list=frequency_list[-10:]
top10_bool=ch_df['Subsector Name'].isin(top10_list)
top10_df=ch_df[top10_bool]
top10_df['Subsector Name'].value_counts().plot.bar()
plt.show()
plt.clf()
plt.close()


# sns.scatterplot(ch_df['Financial: Gross Income'], ch_df['Financial: Gross Expenditure'], hue=ch_df['CHY'], s=30)

# plt.clf()
#ch_df['Subsector Name'].value_counts().plot.bar()
#plt.clf()
#plt.scatter(ch_df['Financial: Gross Income'], ch_df['Financial: Gross Expenditure'], s=5)
#plt.clf()

#ch_df.groupby('Subsector Name')['Financial: Gross Income'].max().plot(kind='barh')

plt.clf()
plt.close()
#top 10 subsector analysis

frequency_df=ch_df['Subsector Name'].value_counts()
frequency_list=frequency_df.index.tolist()
top10_list=frequency_list[0:10]
btm10_list=frequency_list[-10:]
top10_bool=ch_df['Subsector Name'].isin(top10_list)
top10_df=ch_df[top10_bool]
#top10_df['Subsector Name'].value_counts().plot.bar()
sns.countplot(data=top10_df, x='Subsector Name')
plt.show()
plt.clf()
plt.close()
btm10_bool=ch_df['Subsector Name'].isin(btm10_list)
btm10_df=ch_df[btm10_bool]
sns.countplot(data=btm10_df, x='Subsector Name')

top50_df=ch_df.sort_values('Financial: Gross Income', ascending=False).iloc[0:50,:]
top50_df['Subsector Name'].value_counts().plot.bar()

count_df=ch_df[ch_df['Financial: Gross Income'] < 2111977]

# dub_df['Financial: Gross Income'].cumsum().iloc[-1]

# ch_df['Financial: Gross Profit']=ch_df['Financial: Gross Income'] - ch_df['Financial: Gross Expenditure']

#sns.scatterplot(data=ch_df, x='Financial: Gross Income', y='Financial: Gross Expenditure', hue='SurpDef')
#p#lt.clf()
#t.close()
# sns.countplot(data=ch_df, x='County')

sns.barplot(data=top50_df, x= 'County', y='Financial: Gross Income')
sns.barplot(data=top50_df, x= 'Governing Form', y='Financial: Gross Income', rotation =90)


ax = sns.countplot(x="County", data=ch_df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
plt.clf()
plt.close()
plt.figure(figsize=(15,10)) #adjust the size of plot



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")  #it will rotate text on x axis

plt.tight_layout()
plt.show()

sns.lmplot(data=ch_df, x='Financial: Gross Income', y='Financial: Gross Expenditure')

plt.hist(top50_df['Financial: Gross Income'])


sns.barplot(data=ch_df, y='Financial: Gross Income', x='Subsector Name', hue='County')
