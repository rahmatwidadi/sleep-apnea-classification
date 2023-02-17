import pandas as pd
import numpy as np
import os 
import streamlit as st
import tensorflow as tf

model = tf.keras.models.load_model('Model_Terbaik.h5')

st.set_page_config(page_title='Sleep Apnea Classifier', 
                   layout="centered", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv')

with st.sidebar:
    st.title("Sleep Apnea Classifier")
    choice = st.radio("Process the data here!", ["Home","About"], label_visibility='hidden')
    st.info("This project application helps you predict sleep apnea.")
    
if choice == "Home":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset", type=['csv'], label_visibility="collapsed")
        if file: 
            df = pd.read_csv(file, header=None)
            df.to_csv('dataset.csv')
            data = df
            st.caption('This is your 5 last column sample of your data')
            st.dataframe(data.iloc[:, -5:])
            
            predict_button = st.button('Predict')

            if predict_button:
                
                list_predict_data = []

                test_data = df.values
                predict_data = model.predict(test_data)
                
                for i in predict_data:
                    class_name = np.argmax(i)
                    list_predict_data.append(class_name)
                
                st.caption('This is your 6 last sample column predicted file')
    
                df['Predict'] = list_predict_data
                df['Predict'] = df['Predict'].replace([0, 1], ['Negative', 'Positive'])
                df.to_csv('predicted_data.csv')
                data['Predict'] = list_predict_data
                data['Predict'] = data['Predict'].replace([0, 1], ['Negative', 'Positive'])
                st.dataframe(data.iloc[:, -6:])

                with open('predicted_data.csv', 'rb') as f:
                    st.download_button('Download file', f, file_name='predicted_data.csv')

if choice == "About":
    st.title("Sleep Apnea Classifier")
    st.markdown('---')
    st.caption('This project is used to predict data into sleep apnea or negative. Model using Deep Learning Conv1D layer, resulting 91% of accuracy.')