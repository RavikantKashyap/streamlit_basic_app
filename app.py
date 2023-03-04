import pandas as pd
import numpy as np
import streamlit as st
import pickle

# loading the saved model
loaded_model = pickle.load(open('dob_model.sav', 'rb'))

def dob_check(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'cluster 0: both Lp and prep present'
    elif (prediction[0] == 1):
        return 'cluster 1:  Lp missing and prep present'
    else:
        return 'cluster 2:  Lp present and prep missing'

# giving a title
st.title('Date of Birth web App')
    
    
# getting the input data from the user
dob = st.text_input('Enter dob for LP')
edqdob = st.text_input('Enter dob for prep')
    
# code for Prediction
result = ''
    
# creating a button for Prediction
    
if st.button('DOB Test Result'):
    result = dob_check([dob, edqdob])
        
        
st.success(result)
