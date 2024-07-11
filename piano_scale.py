import streamlit as st
import numpy as np

from scipy.io import wavfile

st.title('Piano Scale Calculation')

st.write('This tool is supposed to help you calculate the parameters of a piano string scale.')

st.checkbox('Check me out')

# st.write('Here\'s our first attempt at using data to create a table:')
# dataframe = np.random.randn(30, 30)
# st.dataframe(dataframe)

kammerton = st.number_input("Insert concert pitch")
st.write("The current concert pitch is ", kammerton, " Hz.")