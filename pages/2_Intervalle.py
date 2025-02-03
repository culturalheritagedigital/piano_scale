import streamlit as st
import numpy as np
import pandas as pd

from scipy.io import wavfile

# note_names = ('A0', 'A♯0', 'B0', 'C1', 'C♯1', 'D1', 'D♯1', 'E1', 'F1', 'F♯1',
#        'G1', 'G♯1', 'A1', 'A♯1', 'B1', 'C2', 'C♯2', 'D2', 'D♯2', 'E2',
#        'F2', 'F♯2', 'G2', 'G♯2', 'A2', 'A♯2', 'B2', 'C3', 'C♯3', 'D3',
#        'D♯3', 'E3', 'F3', 'F♯3', 'G3', 'G♯3', 'A3', 'A♯3', 'B3', 'C4',
#        'C♯4', 'D4', 'D♯4', 'E4', 'F4', 'F♯4', 'G4', 'G♯4', 'A4', 'A♯4',
#        'B4', 'C5', 'C♯5', 'D5', 'D♯5', 'E5', 'F5', 'F♯5', 'G5', 'G♯5',
#        'A5', 'A♯5', 'B5', 'C6', 'C♯6', 'D6', 'D♯6', 'E6', 'F6', 'F♯6',
#        'G6', 'G♯6', 'A6', 'A♯6', 'B6', 'C7', 'C♯7', 'D7', 'D♯7', 'E7',
#        'F7', 'F♯7', 'G7', 'G♯7', 'A7', 'A♯7', 'B7', 'C8')

note_names = ('A4', 'A♯4',
       'B4', 'C5', 'C♯5', 'D5', 'D♯5', 'E5', 'F5', 'F♯5', 'G5', 'G♯5',
       'A5')

# # String diameters of steel strings in mm
# string_diameters = [0.700, 0.725, 0.750, 0.775, 0.800, 0.825, 0.850, 0.875, 0.900, 0.925, 0.950, 0.975, 1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150, 1.175, 1.200, 1.225, 1.250, 1.300, 1.350, 1.400, 1.450, 1.500, 1.550, 1.600]

# # Tensile strengths of steel in N/mm^2
# tensile_strengths = [2480.00, 2470.00, 2440.00, 2420.00, 2400.00, 2380.00, 2360.00, 2350.00, 2340.00, 2320.00, 2310.00, 2290.00, 2280.00, 2260.00, 2240.00, 2220.00, 2220.00, 2200.00, 2200.00, 2180.00, 2180.00, 2160.00, 2160.00, 2110.00, 2110.00, 2060.00, 2060.00, 2000.00, 2000.00, 1980.00,]

# # calculate the load capacity in N:
# def load_capacity(diameter, tensile_strength, safety_factor=0.8):
#     return np.pi * (diameter/2)**2 * tensile_strength * safety_factor

# # String load capacity in N (including a safety factor of 0.8)
# string_load_capacities = np.round([load_capacity(d, ts) for d, ts in zip(string_diameters, tensile_strengths)],2)

# rho = 7850  # Density of steel in kg/m^3

# # Youngs modulus of steel in N/mm^2
# E = 215000  # N/mm^2



def generate_wav_file(frequencies, amplitudes_db, damping_factors):
    duration = 3  # Duration of the sound in seconds

    hamm = np.hamming(48000)[24000:48000]
    ones = np.ones(int(48000*2.5))
    fadeout = np.append(ones, hamm)

    sample_rate = 48000  # Sample rate in Hz

    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples, endpoint=False)

    # Initialize the composite sound signal
    signal = np.zeros(num_samples)

    # Find the loudest sine amplitude
    max_amplitude_db = max(amplitudes_db)
    max_amplitude = 10**(max_amplitude_db / 20.0)  # Convert dB to linear scale

    # Generate individual sinusoidal components
    for frequency, amplitude_db, damping_factor in zip(frequencies, amplitudes_db, damping_factors):
        # Calculate the decay factor for the damping
        decay = np.exp(-damping_factor * time)

        # Convert amplitude from dB to linear scale, relative to the loudest sine
        amplitude = 10**((amplitude_db - max_amplitude_db) / 20.0) * max_amplitude

        # Generate the sinusoidal wave with decay
        wave = amplitude * np.sin(2 * np.pi * frequency * time) * decay

        # Add the wave to the composite signal
        signal += wave

    # Normalize the signal
    signal /= np.max(np.abs(signal))

    # Convert the signal to the appropriate data type for WAV files (-32767 to 32767 for int16)
    signal = (32767 * signal).astype(np.int16)
    signal = signal[0:48000*3]
    signal = signal * fadeout

    return signal



st.title('Intervalle')

# with st.expander("Click to read more:"):
#     st.markdown(
#     """
#     - An ideal string has no stiffness and has uniform linear density throughout.
#     - The restoring force for transverse vibrations is provided solely by the tension in the string.
#     - The partials follow a harmonic series:
#     """
#     )

#     st.latex(r''' f_n = n \cdot f_1 ''')
#st.checkbox('Check me out')

# st.write('Here\'s our first attempt at using data to create a table:')
# dataframe = np.random.randn(30, 30)
# st.dataframe(dataframe)

kammerton = st.number_input("Choose a concert pitch:", value=440, step=1)
st.write("Der aktuelle Kammerton hat ", kammerton, " Hz.")

def f(key):
    return np.round(kammerton * 2**((key-49)/12),4)

key1 = st.selectbox(
    "Wählen Sie eine Taste:",
    note_names, index=0)

key_num1 = note_names.index(key1)+1

key2 = st.selectbox(
    "Wählen Sie eine Taste:",
    note_names, index=0)

key_num2 = note_names.index(key2)+1


st.write("Die aktuell gewählten Tasten sind ", key1, " und ", key2," mit den Grundfrequenzen ", f(key_num1), "Hz und ", f(key_num2)," in gleichstufig temperierter Stimmung.")

#st.subheader("Ideal String")



n = st.number_input("Insert number of harmonics:", value=10, min_value=1)


damping_factor = st.slider("Select a damping factor:", min_value=0.0, max_value=3.0, value=.3, step=.1)
#st.write("I'm ", age, "years old")

frequencies1 = [f(key_num1) * k for k in np.arange(1,n+1,1)]  # Frequencies in Hz
frequencies2 = [f(key_num2) * k for k in np.arange(1,n+1,1)]  # Frequencies in Hz

amplitudes = [0-k for k in np.arange(1,n+1,1)]  # Amplitudes in dB
damping_factors = damping_factor*np.arange(n+1)  # Damping factors in dB/sec

signal = generate_wav_file(frequencies1, amplitudes, damping_factors)

st.audio(signal, format="audio/mpeg", sample_rate=48000)

four = np.abs(np.fft.fft(signal[0:48000]))
four = four/np.max(four)
fourlog = 20*np.log10(four/np.max(four))

if f(key_num)*(n+2) >20000:
    st.line_chart(fourlog[0:20000], x_label="Frequency [Hz]", y_label="Amplitude [dB]")
else:
    st.line_chart(fourlog[0:int(f(key_num)*(n+2))], x_label="Frequency [Hz]", y_label="Amplitude [dB]")
    
