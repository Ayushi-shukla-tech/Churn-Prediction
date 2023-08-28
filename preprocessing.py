#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """
    #Defining the map function
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0})

    # Encode binary categorical features
    binary_list = ['Gender','Senior_Citizen','Location_Houston','Location_Los_Angeles','Location_Miami','Location_New_York']
    df[binary_list] = df[binary_list].apply(binary_map)

    
    #Drop values based on operational options
    if (option == "Online"):
        columns = ['Age','Gender','Subscription_Length_Months', 'Monthly_Bill','Total_Usage_GB','Senior_Citizen', 'Total_Charges','Location_Houston','Location_Los_Angeles','Location_Miami','Location_New_York']
        #Encoding the other categorical categoric features with more than two categories
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    elif (option == "Batch"):
        pass
        df = df[['Age','Gender','Subscription_Length_Months', 'Monthly_Bill','Total_Usage_GB','Senior_Citizen', 'Total_Charges','Location_Houston','Location_Los_Angeles','Location_Miami','Location_New_York']]
        columns = ['Age','Gender','Subscription_Length_Months', 'Monthly_Bill','Total_Usage_GB','Senior_Citizen', 'Total_Charges','Location_Houston','Location_Los_Angeles','Location_Miami','Location_New_York']
        #Encoding the other categorical categoric features with more than two categories
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    else:
        print("Incorrect operational options")


    #feature scaling
    sc = MinMaxScaler()
    df['Subscription_Length_Months'] = sc.fit_transform(df[['Subscription_Length_Months']])
    df['Monthly_Bill'] = sc.fit_transform(df[['Monthly_Bill']])
    df['Total_Charges'] = sc.fit_transform(df[['Total_Charges']])
    return df

