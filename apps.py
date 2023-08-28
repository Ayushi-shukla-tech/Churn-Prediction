#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pickle
import pandas as pd
import os
#from flasgger import Swagger
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image


# In[5]:


#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

#load the model from disk
import joblib
model = joblib.load(r"model.sav")

#Import python scripts
from preprocessing import preprocess

def main():
    #Setting Application title
    st.title('Customer Churn Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn based on historical customer data..
    The application is functional for both online prediction and batch data prediction. n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('App.jpeg')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        #Based on our optimal features selection
        st.subheader("Demographic data")
        seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
        Location_Houston = st.selectbox('Location_Houston:', ('Yes', 'No'))
        Location_Los_Angeles = st.selectbox('Location_Los_Angeles:', ('Yes', 'No'))
        Location_Miami = st.selectbox('Location_Miami:', ('Yes', 'No'))
        Location_New_York = st.selectbox('Location_New_York:', ('Yes', 'No'))   
        gender = st.selectbox('Gender:', ('Male', 'Female'))
        st.subheader("Payment data")
        tenure = st.slider('Number of months the customer taken subscription with the company', min_value=0, max_value=72, value=0)
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
        totalcharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)
        Total_Usage_GB = st.number_input('The total amount GB used by customer',min_value=0, max_value=500, value=0)
        Age = st.number_input('Age of customer',min_value=0, max_value=100, value=0)
        

        data = {
                'Age': Age,
                'Gender': gender,
                'Subscription_Length_Months':tenure,
                'Monthly_Bill': monthlycharges,
                'Total_Usage_GB' : Total_Usage_GB,
                'Senior_Citizen': seniorcitizen,
                'Total_Charges': totalcharges,
                'Location_Houston': Location_Houston,
                'Location_New_York': Location_New_York,
                'Location_Miami': Location_Miami,
                'Location_Los_Angeles': Location_Los_Angeles,
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)
        #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with SunbaseData Services.')


    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the customer will terminate the service.',
                                                    0:'No, the customer is happy with SunbaseData Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
        main()

