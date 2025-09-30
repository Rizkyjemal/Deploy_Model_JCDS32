
import streamlit as st
import pandas as pd
import numpy as np
import pickle

#Memberikan judul
st.title("Survive Predictor")
st.title("This website can beused to predict survival rate Titanic Customer")

#Menambahkan sidebar
st.sidebar.header("Please Input Customer's features!")

#Create user input
def create_user_input():
    #numerical : "pclass", "age", "sibsp","parch","fare"
    pclass = st.sidebar.slider("pclass",min_value=1, max_value=3,value=1)
    age = st.sidebar.slider("age", min_value=1,max_value=80,value=22)
    sibsp = st.sidebar.slider("sibsp",min_value=0,max_value=8,value=1)
    parch = st.sidebar.slider("parch",min_value=0,max_value=6,value=1)
    fare = st.sidebar.slider("fare",min_value=0,max_value=513,value=30)

    #Categorical : "sex", "embarked"
    sex = st.sidebar.radio("sex",["male","female"])
    embarked = st.sidebar.radio("embarked",["S","C","Q"])

    #Createdictionary from data input
    user_data ={
        "pclass":pclass,
        "sex":sex,
        "age":age,
        "sibsp":sibsp,
        "parch":parch,
        "fare":fare,
        "embarked":embarked
    }

    #Convert to dataframe
    user_data_df=pd.DataFrame([user_data])
    return user_data_df

#Define customer data
data_customer = create_user_input()

#Create 2 containers
col1, col2 = st.columns(2)

#Kiri
with col1:
    st.subheader("Customer Features")
    st.write(data_customer.transpose())

#Load Model
with open("best_model.sav","rb") as f:
    model_loaded = pickle.load(f)

#Predict to customers data
target = model_loaded.predict(data_customer) #Target 1 atau 0 dari model
probability = model_loaded.predict_proba(data_customer) [0] 

#Menampilkan hasil prediksi
# Kanan
with col2:
    st.subheader("Prediction Result")
    if target == 1:
        st.write("This Customer will SURVIVE!")
    else:
        st.write("This Customer will NOT SURVIVE!")
    
    #Display probability
    st.write(f"Probability of survive : {probability[1]:.2f}")
