# import streamlit as st
# import numpy as np
# import pandas as pd

# from scipy.io import wavfile


# # note_names = ('A0', 'A♯0', 'B0', 'C1', 'C♯1', 'D1', 'D♯1', 'E1', 'F1', 'F♯1',
# #        'G1', 'G♯1', 'A1', 'A♯1', 'B1', 'C2', 'C♯2', 'D2', 'D♯2', 'E2',
# #        'F2', 'F♯2', 'G2', 'G♯2', 'A2', 'A♯2', 'B2', 'C3', 'C♯3', 'D3',
# #        'D♯3', 'E3', 'F3', 'F♯3', 'G3', 'G♯3', 'A3', 'A♯3', 'B3', 'C4',
# #        'C♯4', 'D4', 'D♯4', 'E4', 'F4', 'F♯4', 'G4', 'G♯4', 'A4', 'A♯4',
# #        'B4', 'C5', 'C♯5', 'D5', 'D♯5', 'E5', 'F5', 'F♯5', 'G5', 'G♯5',
# #        'A5', 'A♯5', 'B5', 'C6', 'C♯6', 'D6', 'D♯6', 'E6', 'F6', 'F♯6',
# #        'G6', 'G♯6', 'A6', 'A♯6', 'B6', 'C7', 'C♯7', 'D7', 'D♯7', 'E7',
# #        'F7', 'F♯7', 'G7', 'G♯7', 'A7', 'A♯7', 'B7', 'C8')

# note_names = ('A4', 'A♯4',
#        'B4', 'C5', 'C♯5', 'D5', 'D♯5', 'E5', 'F5', 'F♯5', 'G5', 'G♯5',
#        'A5')

# def generate_wav_file(frequencies, amplitudes_db, damping_factors):
#     duration = 3  # Duration of the sound in seconds

#     hamm = np.hamming(48000)[24000:48000]
#     ones = np.ones(int(48000*2.5))
#     fadeout = np.append(ones, hamm)

#     sample_rate = 48000  # Sample rate in Hz

#     num_samples = int(duration * sample_rate)
#     time = np.linspace(0, duration, num_samples, endpoint=False)

#     # Initialize the composite sound signal
#     signal = np.zeros(num_samples)

#     # Find the loudest sine amplitude
#     max_amplitude_db = max(amplitudes_db)
#     max_amplitude = 10**(max_amplitude_db / 20.0)  # Convert dB to linear scale

#     # Generate individual sinusoidal components
#     for frequency, amplitude_db, damping_factor in zip(frequencies, amplitudes_db, damping_factors):
#         # Calculate the decay factor for the damping
#         decay = np.exp(-damping_factor * time)

#         # Convert amplitude from dB to linear scale, relative to the loudest sine
#         amplitude = 10**((amplitude_db - max_amplitude_db) / 20.0) * max_amplitude

#         # Generate the sinusoidal wave with decay
#         wave = amplitude * np.sin(2 * np.pi * frequency * time) * decay

#         # Add the wave to the composite signal
#         signal += wave

#     # Normalize the signal
#     signal /= np.max(np.abs(signal))

#     # Convert the signal to the appropriate data type for WAV files (-32767 to 32767 for int16)
#     signal = (32767 * signal).astype(np.int16)
#     signal = signal[0:48000*3]
#     signal = signal * fadeout

#     return signal



# st.title('Intervalle')

# kammerton = 440
# # kammerton = st.number_input("Choose a concert pitch:", value=440, step=1)
# # st.write("Der aktuelle Kammerton hat ", kammerton, " Hz.")

# def f(key):
#     return np.round(kammerton * 2**((key-49)/12),4)

# key1 = st.selectbox(
#     "Wählen Sie eine Taste:",
#     note_names, index=0)

# key_num1 = note_names.index(key1)+48

# key2 = st.selectbox(
#     "Wählen Sie eine zweite Taste:",
#     note_names, index=0)

# key_num2 = note_names.index(key2)+48


# st.write("Die aktuell gewählten Tasten sind ", key1, " und ", key2," mit den Grundfrequenzen ", f(key_num1), "Hz und ", f(key_num2)," in gleichstufig temperierter Stimmung.")

# n = st.number_input("Insert number of harmonics:", value=10, min_value=1)

# damping_factor = st.slider("Select a damping factor:", min_value=0.0, max_value=3.0, value=.3, step=.1)

# frequencies1 = [f(key_num1) * k for k in np.arange(1,n+1,1)]  # Frequencies in Hz
# frequencies2 = [f(key_num2) * k for k in np.arange(1,n+1,1)]  # Frequencies in Hz

# amplitudes = [0-k for k in np.arange(1,n+1,1)]  # Amplitudes in dB
# damping_factors = damping_factor*np.arange(n+1)  # Damping factors in dB/sec

# signal = generate_wav_file(frequencies1, amplitudes, damping_factors)

# st.audio(signal, format="audio/mpeg", sample_rate=48000)

# four = np.abs(np.fft.fft(signal[0:48000]))
# four = four/np.max(four)
# fourlog = 20*np.log10(four/np.max(four))

# if f(key_num1)*(n+2) >20000:
#     st.line_chart(fourlog[0:20000], x_label="Frequency [Hz]", y_label="Amplitude [dB]")
# else:
#     st.line_chart(fourlog[0:int(f(key_num1)*(n+2))], x_label="Frequency [Hz]", y_label="Amplitude [dB]")
    


import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile

note_names = ('A4', 'A♯4',
       'B4', 'C5', 'C♯5', 'D5', 'D♯5', 'E5', 'F5', 'F♯5', 'G5', 'G♯5',
       'A5')

intervall_name = ('Prime', 'kleine Sekunde', 'große Sekunde', 'kleine Terz', 'große Terz', 'reine Quarte', 'Tritonus / verminderte Quinte', 'reine Quinte', 'kleine Sexte', 'große Sexte', 'kleine Septime', 'große Septime', 'Oktave')

st.set_page_config(
    page_title="Sonare",
    page_icon="🎹",
)

def generate_wav_file(frequencies1, frequencies2, amplitudes_db, damping_factors):
    duration = 5
    hamm = np.hamming(48000)[24000:48000]
    ones = np.ones(int(48000*2.5))
    fadeout = np.append(ones, hamm)
    sample_rate = 48000
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples, endpoint=False)
    
    signal1 = np.zeros(num_samples)
    signal2 = np.zeros(num_samples)
    
    max_amplitude_db = max(amplitudes_db)
    max_amplitude = 10**(max_amplitude_db / 20.0)

    # Generate components for first key
    for idx, frequency in enumerate(frequencies1):
        decay = np.exp(-damping_factors[idx] * time)
        amplitude = 10**((amplitudes_db[idx] - max_amplitude_db) / 20.0) * max_amplitude
        wave = amplitude * np.sin(2 * np.pi * frequency * time) * decay
        signal1 += wave

    # Generate components for second key
    for idx, frequency in enumerate(frequencies2):
        decay = np.exp(-damping_factors[idx] * time)
        amplitude = 10**((amplitudes_db[idx] - max_amplitude_db) / 20.0) * max_amplitude
        wave = amplitude * np.sin(2 * np.pi * frequency * time) * decay
        signal2 += wave

    # Combine signals
    signal = signal1 + signal2
    signal /= np.max(np.abs(signal))
    signal = (32767 * signal).astype(np.int16)
    signal = signal[0:48000*3]
    signal = signal * fadeout
    return signal, signal1, signal2

st.title('Schwebungen und Rauhigkeit')

st.write("Wählen Sie zwei Frequenzen aus:")

f1 = st.number_input("f1:", value=440, min_value=1, max_value=4400)

f2 = st.slider("f2:", min_value=f1, max_value=f1*2, value=f1, step=1)



# kammerton = 440

# def f(key):
#     return np.round(kammerton * 2**((key-49)/12),4)

# key1 = st.selectbox(
#     "Wählen Sie eine Taste:",
#     note_names[0], index=0)

# key_num1 = note_names.index(key1)+49

# key2 = st.selectbox(
#     "Wählen Sie eine zweite Taste:",
#     note_names, index=0)

# key_num2 = note_names.index(key2)+49

# st.write("Die aktuell gewählten Tasten sind:")

# st.write(key1, "mit ",  f(key_num1), "Hz")

# st.write(key2, "mit " , f(key_num2), "Hz")

# if key_num1 > key_num2:
#     interv_name = intervall_name[key_num1-key_num2]
# else:
#     interv_name = intervall_name[key_num2-key_num1]

# st.write("mit einem Intervall einer ", interv_name," in gleichstufig temperierter Stimmung.")


# n = st.number_input("Wählen Sie die Anzahl der Teiltöne:", value=20, min_value=1)

#damping_factor = st.slider("Wählen Sie einen Dämpfungsfaktor:", min_value=0.0, max_value=1.0, value=.2, step=.05)

n=1
damping_factor = 0

frequencies1 = [f1 * k for k in np.arange(1,n+1,1)]
frequencies2 = [f2 * k for k in np.arange(1,n+1,1)]

amplitudes = [0-k for k in np.arange(1,n+1,1)]
damping_factors = damping_factor*np.arange(n+1)

signal, signal1, signal2 = generate_wav_file(frequencies1, frequencies2, amplitudes, damping_factors)

st.audio(signal, format="audio/mpeg", sample_rate=48000)

# Calculate FFT for both signals
four1 = np.abs(np.fft.fft(signal1[0:48000]))
four1 = four1/np.max(four1)
fourlog1 = 20*np.log10(four1/np.max(four1))

four2 = np.abs(np.fft.fft(signal2[0:48000]))
four2 = four2/np.max(four2)
fourlog2 = 20*np.log10(four2/np.max(four2))

# Create DataFrame for plotting
# max_freq = max(f1, f2)
# if max_freq*(n+2) >5000:
#     plot_range = 5000
# else:
#     plot_range = int(max_freq*(n+2))

plot_range = f2*2

df = pd.DataFrame({
    f1: fourlog1[0:plot_range],
    f2: fourlog2[0:plot_range]
})

st.write("Das Spektrum der beiden Töne ist in der folgenden Grafik dargestellt. Sie können den Frequenzbereich durch Zoomen anpassen.")
# st.line_chart(signal)
st.line_chart(df)