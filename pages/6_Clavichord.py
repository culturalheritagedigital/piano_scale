import streamlit as st
import numpy as np
import pandas as pd

from scipy.io import wavfile

note_names = ('A0', 'A♯0', 'B0', 'C1', 'C♯1', 'D1', 'D♯1', 'E1', 'F1', 'F♯1',
       'G1', 'G♯1', 'A1', 'A♯1', 'B1', 'C2', 'C♯2', 'D2', 'D♯2', 'E2',
       'F2', 'F♯2', 'G2', 'G♯2', 'A2', 'A♯2', 'B2', 'C3', 'C♯3', 'D3',
       'D♯3', 'E3', 'F3', 'F♯3', 'G3', 'G♯3', 'A3', 'A♯3', 'B3', 'C4',
       'C♯4', 'D4', 'D♯4', 'E4', 'F4', 'F♯4', 'G4', 'G♯4', 'A4', 'A♯4',
       'B4', 'C5', 'C♯5', 'D5', 'D♯5', 'E5', 'F5', 'F♯5', 'G5', 'G♯5',
       'A5', 'A♯5', 'B5', 'C6', 'C♯6', 'D6', 'D♯6', 'E6', 'F6', 'F♯6',
       'G6', 'G♯6', 'A6', 'A♯6', 'B6', 'C7', 'C♯7', 'D7', 'D♯7', 'E7',
       'F7', 'F♯7', 'G7', 'G♯7', 'A7', 'A♯7', 'B7', 'C8')

# String diameters of steel strings in mm
string_diameters = [0.700, 0.725, 0.750, 0.775, 0.800, 0.825, 0.850, 0.875, 0.900, 0.925, 0.950, 0.975, 1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150, 1.175, 1.200, 1.225, 1.250, 1.300, 1.350, 1.400, 1.450, 1.500, 1.550, 1.600]

# Tensile strengths of steel in N/mm^2
tensile_strengths = [2480.00, 2470.00, 2440.00, 2420.00, 2400.00, 2380.00, 2360.00, 2350.00, 2340.00, 2320.00, 2310.00, 2290.00, 2280.00, 2260.00, 2240.00, 2220.00, 2220.00, 2200.00, 2200.00, 2180.00, 2180.00, 2160.00, 2160.00, 2110.00, 2110.00, 2060.00, 2060.00, 2000.00, 2000.00, 1980.00,]

# calculate the load capacity in N:
def load_capacity(diameter, tensile_strength, safety_factor=0.8):
    return np.pi * (diameter/2)**2 * tensile_strength * safety_factor

# String load capacity in N (including a safety factor of 0.8)
string_load_capacities = np.round([load_capacity(d, ts) for d, ts in zip(string_diameters, tensile_strengths)],2)

rho = 7850  # Density of steel in kg/m^3

# Youngs modulus of steel in N/mm^2
E = 215000  # N/mm^2



st.title('Clavichord')

st.subheader("Tangentenposition")

with st.expander("Click to read more:"):
    st.markdown(
    """
    - An ideal string has no stiffness and has uniform linear density throughout.
    - The restoring force for transverse vibrations is provided solely by the tension in the string.
    - The partials follow a harmonic series:
    """
    )

    st.latex(r''' f_n = n \cdot f_1 ''')
#st.checkbox('Check me out')

# st.write('Here\'s our first attempt at using data to create a table:')
# dataframe = np.random.randn(30, 30)
# st.dataframe(dataframe)

saitenlaenge = st.number_input("Geben Sie die Länge der Saite an:", value=650, step=0.01)

st.write("Die gewählte Saitenlänge ist: ", saitenlaenge, " mm.")




# n = st.number_input("Insert number of harmonics:", value=20, min_value=1)


# damping_factor = st.slider("Select a damping factor:", min_value=0.0, max_value=3.0, value=.3, step=.1)
# #st.write("I'm ", age, "years old")

# frequencies1 = [f(key_num) * k for k in np.arange(1,n+1,1)]  # Frequencies in Hz
# amplitudes = [0-k for k in np.arange(1,n+1,1)]  # Amplitudes in dB
# damping_factors = damping_factor*np.arange(n+1)  # Damping factors in dB/sec
# signal = generate_wav_file(frequencies1, amplitudes, damping_factors)

# st.audio(signal, format="audio/mpeg", sample_rate=48000)

# four = np.abs(np.fft.fft(signal[0:48000]))
# four = four/np.max(four)
# fourlog = 20*np.log10(four/np.max(four))

# if f(key_num)*(n+2) >20000:
#     st.line_chart(fourlog[0:20000], x_label="Frequency [Hz]", y_label="Amplitude [dB]")
# else:
#     st.line_chart(fourlog[0:int(f(key_num)*(n+2))], x_label="Frequency [Hz]", y_label="Amplitude [dB]")
    

# st.header("Taylor String Parameters")

# st.latex(r''' f_n = \frac{1}{l \cdot d} \cdot \sqrt{\frac{F}{\pi \cdot \rho}}  ''')

# st.write("with l = string length [m], d = string diameter [m], F = string load [N], ρ = density of steel [kg/m^3], n = harmonic number")


# l = st.number_input("Insert string length [mm]:", value=402.00, min_value=40.00, max_value=2500.00, step=0.01)

# d = st.selectbox(
#     "Select a string diameter [mm]:",
#     string_diameters, index=11)

# st.header("Tensile Strengths and Load Capacities")
# #
# def taylor_string_load(f, l, d, rho):
#     return (np.pi * rho * (f * l * d)**2)

# actual_load = np.round(taylor_string_load(f(key_num), l/1000, d/1000, rho),2)
# max_load = string_load_capacities[string_diameters.index(d)]

# percentage_of_max_load = np.round(actual_load/max_load*100,2)

# st.write("The actual load is ", actual_load, "N, which is ", percentage_of_max_load, "% of the maximum load capacity (", max_load, " N, including a safety factor of 0.8).")


# # df = pd.DataFrame({"Diameter (mm)": string_diameters, "Tensile strength (N/mm^2)": tensile_strengths, "Max load capacity (*0.75) (N)": string_load_capacities})

# # st.dataframe(df)

# st.header("String Stretching")

# def string_stretching(load, l, d, E):
#     return (load * l) / (d**2 * (np.pi/4) * E)

# actual_string_stretching = np.round(string_stretching(actual_load, l, d, E),4)

# actual_string_stretching_percent = np.round(actual_string_stretching/l*100,2)

# st.write("The actual string stretching is ", actual_string_stretching, "mm or ", actual_string_stretching_percent,  "%.")