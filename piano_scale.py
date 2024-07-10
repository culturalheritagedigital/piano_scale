import streamlit as st
import numpy as np


st.title('Piano Scale Calculation')

st.write('This tool is supposed to help you calculate the parameters of a piano string scale.')



dataframe = np.random.randn(30, 30)
st.dataframe(dataframe)