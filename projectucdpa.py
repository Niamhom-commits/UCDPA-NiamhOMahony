# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:19:50 2021

@author: omaho
"""
# UCD PA Project Cert Introductory Data Analytics
import pandas as pd
file = "C:\\Users\\omaho\\downloads\\public-register-03052021.xlsx"
charxls=pd.ExcelFile(file)
print(charxls.sheet_names)
chardf1=charxls.parse('Public Register', skiprows=1)
chardf2=charxls.parse('Annual Reports', skiprows=1)
print(chardf1.head())
print(chardf2.head()) 
                     