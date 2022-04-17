import streamlit as st
from clf1 import predict

# streamlit run app.py
st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("LSTM recommendation system")
st.write("")

UserId = st.text_input("please input your User Id",value = None)

if text is not None:

    st.write("")
    st.write("Just a second...")
    recommendation = predict(UserId)


    st.write("top_5_recommendation:", recommendation)