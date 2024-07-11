import streamlit as st
import numpy as np


st.title('Piano Scale Calculation')

st.write('This tool is supposed to help you calculate the parameters of a piano string scale.')

st.checkbox('Check me out')

st.write('Here\'s our first attempt at using data to create a table:')

dataframe = {
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}


dataframe = np.random.randn(30, 30)
st.dataframe(dataframe)