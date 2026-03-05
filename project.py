
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Study Hours vs Grades Dataset")

uploaded = st.file_uploader("Upload the dataset")
stdHrs = st.number_input("Enter the Number of Study Hours : ")
subButton = st.button("Submit")

df = None
output = None

def modelFun(stdHrs):
    X = df[['study_hours']]
    y = df['grade']

    lr = LinearRegression()
    lr.fit(X,y)
    test = pd.DataFrame([[stdHrs]], columns=['study_hours'])
    output = lr.predict(test)
    return output[0]

if subButton and uploaded:
    df = pd.read_csv(uploaded)
    st.success("File is Scanned")

    output = modelFun(stdHrs)
else:
    st.warning("Submit the Form for predictions ")



st.write(output)
