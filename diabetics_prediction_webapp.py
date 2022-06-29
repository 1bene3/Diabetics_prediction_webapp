# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 19:24:10 2022

@author: Rukevwe Ovuowo
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:/Users/Rukevwe Ovuowo/Videos/Data Science/python/Deploying ML models/Diabetics/trained_diabetic_model.sav', 'rb'))


def diabetics_prediction(input_data):

   #change the input data to a numpy array
   to_numpy = np.asarray(input_data)

   #reshape for one instance
   input_reshape = to_numpy.reshape(1, -1)

   #building a predictive model
   prediction = loaded_model.predict(input_reshape)
   print(prediction)

   if(prediction[0] == 0):
     return 'Diabetics'
   else:
     return 'Non diabetics'
   


def main():
    #giving a title
    st.title('Diabetics prediction web app')
    
    # getting the input data from the user
  
    Pregnancies = st.text_input('Enter the value for Pregnancies')

    Glucose = st.text_input('Enter the value for Glucose')
  
    BloodPressure = st.text_input('Enter the value for BloodPressure ')
     
    SkinThickness = st.text_input('Enter the value for SkinThickness')
 
    Insulin = st.text_input('Enter the value for Insulin')
   
    BMI = st.text_input('Enter the value for BMI')
     
    DiabetesPedigreeFunction = st.text_input('Enter the value for DiabetesPedigreeFunction')
    
    Age = st.text_input('Enter the value for Age')
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Diabetics test result'):
        diagnosis = diabetics_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    
    st.success(diagnosis)
        
if __name__ == '__main__':
    main()        
    
    
    
    
    
    
    
    
    
    
    
    
    
     
    
    
    
    