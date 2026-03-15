import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Sonare",
    page_icon="🎹",
)

def generate_wav_file(frequencies1, frequencies2, amplitudes_db, damping_factors):
    duration = 3
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

#f2 = st.number_input("f2:", value=f1+1, min_value=1, max_value=4400)

f2 = st.slider("f2:", min_value=f1, max_value=f1*2, value=f1+1, step=1)

st.write("Frequenzdifferenz: " + str(f2-f1) + " Hz.")

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

amplitudes = [-k for k in np.arange(1,n+1,1)]
damping_factors = damping_factor*np.arange(n+1)

signal, signal1, signal2 = generate_wav_file(frequencies1, frequencies2, amplitudes, damping_factors)

st.audio(signal, format="audio/mpeg", sample_rate=48000)

# Calculate FFT for both signals
four1 = np.abs(np.fft.fft(signal1[0:48000]))
four1 = four1/np.max(four1)
fourlog1 = 20*np.log10(four1)

four2 = np.abs(np.fft.fft(signal2[0:48000]))
four2 = four2/np.max(four2)
fourlog2 = 20*np.log10(four2)

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