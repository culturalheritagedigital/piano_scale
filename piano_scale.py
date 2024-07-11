import streamlit as st
import numpy as np

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

def generate_wav_file(frequencies, amplitudes_db, damping_factors):
    duration = 3  # Duration of the sound in seconds
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

    return signal



st.title('Piano Scale Calculation')

#st.write('This tool is supposed to help you calculate the parameters of a piano string scale.')

#st.checkbox('Check me out')

# st.write('Here\'s our first attempt at using data to create a table:')
# dataframe = np.random.randn(30, 30)
# st.dataframe(dataframe)

kammerton = st.number_input("Insert concert pitch:", value=440.0)
st.write("The current concert pitch is ", kammerton, " Hz.")

def f(key):
    return np.round(kammerton * 2**((key-49)/12),4)

key = st.selectbox(
    "Select a key:",
    note_names, index=48)

key_num = note_names.index(key)+1

st.write("You selected:", key, "with a frequency of", f(key_num), "Hz in Equal temperament.")

st.subheader("Ideal String")

n = st.number_input("Insert number of harmonics:", value=20)
frequencies1 = [f(key_num) * k for k in np.arange(1,n+1,1)]  # Frequencies in Hz
amplitudes = [0-k for k in np.arange(1,n+1,1)]  # Amplitudes in dB
damping_factors = 0.3*np.arange(n+1)**1  # Damping factors in dB/sec
signal = generate_wav_file(frequencies1, amplitudes, damping_factors)

st.audio(signal, format="audio/mpeg", sample_rate=48000)

four = np.abs(np.fft.fft(signal[0:48000]))
four = four/np.max(four)
fourlog = 20*np.log10(four/np.max(four))

if f(key_num)*(n+2) >20000:
    st.line_chart(fourlog[0:20000], x_label="Frequency [Hz]", y_label="Amplitude [dB]")
else:
    st.line_chart(fourlog[0:int(f(key_num)*(n+2))], x_label="Frequency [Hz]", y_label="Amplitude [dB]")
    

st.line_chart(fourlog[0:int(f(key_num)*(n+2))], x_label="Frequency [Hz]", y_label="Amplitude [dB]")