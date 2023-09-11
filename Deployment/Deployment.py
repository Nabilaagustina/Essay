import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
import joblib
import streamlit as st
from PIL import Image
#  load model
knn_from_joblib = joblib.load('../Saved Model/modelKKN.bin')
st.markdown("<h2 style='text-align:center; color:Orange;'>Credit Score Classification</h2>",
            unsafe_allow_html=True)

image_1 = Image.open(r"./Images/dataset-cover.jpg")
st.image(image_1, caption='Pengertian hash', width=700)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        Outstanding_Debt = st.number_input(label='Insert a Outstanding Debt', min_value=0)
        st.write('The current number is ', Outstanding_Debt)
    with col2:
        Annual_Income = st.number_input(label='Insert a Annual Income', min_value=0)
        st.write('The current number is ', Annual_Income)

with st.container():
    col3, col4 = st.columns(2)
    with col3:
        Monthly_Inhand_Salary = st.number_input(label='Insert a Monthly Inhand Salary', min_value=0)
        st.write('The current number is ', Monthly_Inhand_Salary)
    with col4:
        Credit_History_Age = st.number_input(label='Insert a Credit History Age', min_value=0)
        st.write('The current number is ', Credit_History_Age)

with st.container():
    col5, col6 = st.columns(2)
    with col5:
        Interest_Rate = st.number_input(label='Insert a Monthly Interest Rate', min_value=0)
        st.write('The current number is ', Interest_Rate)
    with col6:
        Monthly_Balance = st.number_input(label='Insert a Monthly Balance', min_value=0)
        st.write('The current number is ', Monthly_Balance)

with st.container():
    col7, col8 = st.columns(2)
    with col7:
        Delay_from_due_date = st.number_input(label='Insert a Delay from due date', min_value=0)
        st.write('The current number is ', Delay_from_due_date)
    with col8:
        Total_EMI_per_month = st.number_input(label='Insert a Total EMI per month', min_value=0)
        st.write('The current number is ', Total_EMI_per_month)

with st.container():
    col9, col10 = st.columns(2)
    with col7:
        Changed_Credit_Limit = st.number_input(label='Insert a Changed Credit Limit', min_value=0)
        st.write('The current number is ', Delay_from_due_date)
    with col8:
        Num_of_Loan = st.number_input(label='Insert a Num of Loan', min_value=0)
        st.write('The current number is ', Num_of_Loan)

def user_input_data(): 
    data = {
        'Outstanding_Debt': Outstanding_Debt,
        'Annual_Income': Annual_Income,
        'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
        'Credit_History_Age': Credit_History_Age,
        'Interest_Rate': Interest_Rate,
        'Monthly_Balance': Monthly_Balance,
        'Delay_from_due_date': Delay_from_due_date,
        'Total_EMI_per_month': Total_EMI_per_month,
        'Changed_Credit_Limit': Changed_Credit_Limit,
        'Num_of_Loan': Num_of_Loan,
    }
    input_data = pd.DataFrame(data, index=[0])
    return input_data

df = user_input_data() 
with st.container():
    col11, col12= st.columns(2)
    with col11:
        st.dataframe(df.astype(str).T.rename(columns={0:'input_data'}))
    with col12:
        if st.button('Buat Prediksi'):   
            prediction = knn_from_joblib.predict(df)[0]
            st.success(f'Credit score probability is:&emsp;{prediction}')