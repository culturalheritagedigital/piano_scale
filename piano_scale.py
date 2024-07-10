import streamlit as st
import numpy as np


st.title('Piano Scale')

#st.write('Here is a piano scale:')


dataframe = np.random.randn(10, 20)
st.dataframe(dataframe)