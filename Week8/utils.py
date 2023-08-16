import pandas as pd

def info_data(data):
    
    print('Show dataset:')
    print(data.head(2).T)
    print(50*'+')
    
    print('Informations:')
    print(data.info())
    print(50*'-')
    
    print('Column names:')
    print(data.columns)
    print(50*'-')
    
    print('Statistical reports:')
    print(data.describe().T)
    print(50*'-')
    
    print('Check Null values:')
    print(data.isnull().sum())
    print(50*'-')
    
    print('Check unique values:')
    print(data.nunique())
    

