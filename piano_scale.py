import streamlit as st
import numpy as np

from scipy.io import wavfile

note_names = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A♯0', 'A♯1',
       'A♯2', 'A♯3', 'A♯4', 'A♯5', 'A♯6', 'A♯7', 'B0', 'B1', 'B2', 'B3',
       'B4', 'B5', 'B6', 'B7', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
       'C8', 'C♯1', 'C♯2', 'C♯3', 'C♯4', 'C♯5', 'C♯6', 'C♯7', 'D1', 'D2',
       'D3', 'D4', 'D5', 'D6', 'D7', 'D♯1', 'D♯2', 'D♯3', 'D♯4', 'D♯5',
       'D♯6', 'D♯7', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'F1', 'F2',
       'F3', 'F4', 'F5', 'F6', 'F7', 'F♯1', 'F♯2', 'F♯3', 'F♯4', 'F♯5',
       'F♯6', 'F♯7', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G♯1',
       'G♯2', 'G♯3', 'G♯4', 'G♯5', 'G♯6', 'G♯7']

st.title('Piano Scale Calculation')

#st.write('This tool is supposed to help you calculate the parameters of a piano string scale.')

#st.checkbox('Check me out')

# st.write('Here\'s our first attempt at using data to create a table:')
# dataframe = np.random.randn(30, 30)
# st.dataframe(dataframe)

kammerton = st.number_input("Insert concert pitch", value=440.0)
st.write("The current concert pitch is ", kammerton, " Hz.")

key = st.selectbox(
    "Select a key:",
    note_names)

st.write("You selected:", key)