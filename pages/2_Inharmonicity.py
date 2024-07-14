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
def load_capacity(diameter, tensile_strength, safety_factor=0.75):
    return np.pi * (diameter/2)**2 * tensile_strength * safety_factor

# String load capacity in N (including a safety factor of 0.75)
string_load_capacities = np.round([load_capacity(d, ts) for d, ts in zip(string_diameters, tensile_strengths)],2)

rho = 7850  # Density of steel in kg/m^3

# Youngs modulus of steel in N/mm^2
E = 215000  # N/mm^2



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



st.title('Real Stretched String')

with st.expander("Click to read more:"):

    st.markdown(
    """
    - A real string has a bending stiffness.
    - The restoring force for transverse vibrations is composed by the tension in the string and an additional wave length dependend part (dispersion).
    - The partials do not follow a straight harmonic series:
    """
    )

    st.latex(r''' f_n > n \cdot F_0 ''')


kammerton = st.number_input("Choose a concert pitch:", value=440, step=1)
st.write("The current concert pitch is ", kammerton, " Hz.")

def f(key):
    return np.round(kammerton * 2**((key-49)/12),4)

key = st.selectbox(
    "Select a key:",
    note_names, index=48)

key_num = note_names.index(key)+1

st.write("The current key is ", key, " with a fundamental frequency of", f(key_num), "Hz in Equal temperament.")

#st.subheader("Ideal String")

st.header("Inharmonicity Calculation")

l = st.number_input("Insert string length [mm]:", value=402.00, min_value=40.00, max_value=2500.00, step=0.01)

d = st.selectbox(
    "Select a string diameter [mm]:",
    string_diameters, index=11)

st.markdown("__Attention__: For the inharmonicity calculation, the fundamental frequency of the string without stiffness would be needed, which is unknown. Therefore, the fundamental frequency of the real string is used instead. This introduces an error in the inharmonicity calculation.")

def delta_inharmonicity(d, f, l): # d and l in mm
    return 3.4 * 10**15 * d**2 / (f**2 * l**4)

actual_delta_inharmonicity = np.round(delta_inharmonicity(d, f(key_num), l),5)

st.write("The inharmonicity of the string after Young is $\delta = $", actual_delta_inharmonicity , "cent .")

def convert_delta_to_B(delta,n):
    return (2**((2*delta/1200))-1) / n**2

actual_B_inharmonicity = np.round(convert_delta_to_B(actual_delta_inharmonicity, 1),5)

st.write("The corresponding inharmonicity coefficient after Fletcher is $B=$ ", actual_B_inharmonicity, ".")

factor = st.slider("Select a factor for $B$:", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

n = st.number_input("Insert number of harmonics:", value=20, min_value=1)

def calculate_inharmonic_partials(f0, B, n):
    return n* f0 * np.sqrt(1 + B * n**2) 

list_of_inharmonic_partial_frequencies = [calculate_inharmonic_partials(f(key_num), actual_B_inharmonicity*factor, k) for k in np.arange(1,n+1,1)]

damping_factor = st.slider("Select a damping factor:", min_value=0.0, max_value=3.0, value=.3, step=.1)


#st.write("I'm ", age, "years old")

#frequencies1 = [f(key_num) * k for k in np.arange(1,n+1,1)]  # Frequencies in Hz
amplitudes = [0-k for k in np.arange(1,n+1,1)]  # Amplitudes in dB
damping_factors = damping_factor*np.arange(n+1)  # Damping factors in dB/sec
signal = generate_wav_file(list_of_inharmonic_partial_frequencies, amplitudes, damping_factors)

st.audio(signal, format="audio/mpeg", sample_rate=48000)

four = np.abs(np.fft.fft(signal[0:48000]))
four = four/np.max(four)
fourlog = 20*np.log10(four/np.max(four))

if f(key_num)*(n+2) >20000:
    st.line_chart(fourlog[0:20000], x_label="Frequency [Hz]", y_label="Amplitude [dB]")
else:
    st.line_chart(fourlog[0:int(f(key_num)*(n+2))], x_label="Frequency [Hz]", y_label="Amplitude [dB]")
    

def taylor_string_load(f, l, d, rho):
    return (np.pi * rho * (f * l * d)**2)

actual_load = np.round(taylor_string_load(f(key_num), l/1000, d/1000, rho),2)
max_load = string_load_capacities[string_diameters.index(d)]

percentage_of_max_load = np.round(actual_load/max_load*100,2)

def string_stretching(load, l, d, E):
    return (load * l) / (d**2 * (np.pi/4) * E)

actual_string_stretching = np.round(string_stretching(actual_load, l, d, E),4)

actual_string_stretching_percent = np.round(actual_string_stretching/l*100,2)


df = pd.DataFrame({"Concert Pitch [Hz]": kammerton, "f_1": [f(key_num)], "String Length [mm]": l, "String Diameter [mm]": d, "Inharmonicity [cent]": actual_delta_inharmonicity, "Inharmonicity Coefficient B": actual_B_inharmonicity })    

st.dataframe(df)

# st.header("Taylor String Parameters")

# st.latex(r''' f_n = \frac{1}{l \cdot d} \cdot \sqrt{\frac{F}{\pi \cdot \rho}}  ''')

# st.write("with l = string length [m], d = string diameter [m], F = string load [N], ρ = density of steel [kg/m^3], n = harmonic number")


# l = st.number_input("Insert string length [mm]:", value=402.00, min_value=40.00, max_value=2500.00, step=0.01)

# d = st.selectbox(
#     "Select a string diameter [mm]:",
#     string_diameters, index=11)

# st.header("Tensile Strengths and Load Capacities")



# st.write("The actual load is ", actual_load, "N, which is ", percentage_of_max_load, "% of the maximum load capacity (", max_load, " N, including a safety factor of 0.75).")


# # df = pd.DataFrame({"Diameter (mm)": string_diameters, "Tensile strength (N/mm^2)": tensile_strengths, "Max load capacity (*0.75) (N)": string_load_capacities})

# # st.dataframe(df)

# st.header("String Stretching")



# st.write("The actual string stretching is ", actual_string_stretching, "mm or ", actual_string_stretching_percent,  "%.")