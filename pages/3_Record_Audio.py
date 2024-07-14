import streamlit as st
import numpy as np
import pandas as pd
from scipy.io.wavfile import read

import librosa

from math import log,ceil,pi,sin,cos
import operator

from io import BytesIO

from scipy.io import wavfile

from streamlit_mic_recorder import mic_recorder

note_names = ('A0', 'A♯0', 'B0', 'C1', 'C♯1', 'D1', 'D♯1', 'E1', 'F1', 'F♯1',
       'G1', 'G♯1', 'A1', 'A♯1', 'B1', 'C2', 'C♯2', 'D2', 'D♯2', 'E2',
       'F2', 'F♯2', 'G2', 'G♯2', 'A2', 'A♯2', 'B2', 'C3', 'C♯3', 'D3',
       'D♯3', 'E3', 'F3', 'F♯3', 'G3', 'G♯3', 'A3', 'A♯3', 'B3', 'C4',
       'C♯4', 'D4', 'D♯4', 'E4', 'F4', 'F♯4', 'G4', 'G♯4', 'A4', 'A♯4',
       'B4', 'C5', 'C♯5', 'D5', 'D♯5', 'E5', 'F5', 'F♯5', 'G5', 'G♯5',
       'A5', 'A♯5', 'B5', 'C6', 'C♯6', 'D6', 'D♯6', 'E6', 'F6', 'F♯6',
       'G6', 'G♯6', 'A6', 'A♯6', 'B6', 'C7', 'C♯7', 'D7', 'D♯7', 'E7',
       'F7', 'F♯7', 'G7', 'G♯7', 'A7', 'A♯7', 'B7', 'C8')

initial_B = [3.62954232e-04, 3.05756199e-04, 2.55435374e-04, 2.15077819e-04,
       1.72587721e-04, 1.60203287e-04, 1.42836729e-04, 1.40608533e-04,
       1.28491589e-04, 1.22878914e-04, 1.15135953e-04, 1.08297133e-04,
       9.88492612e-05, 1.00403930e-04, 9.83161081e-05, 9.93905933e-05,
       1.01300212e-04, 1.03485934e-04, 1.02608279e-04, 1.05067291e-04,
       1.05345791e-04, 1.05369761e-04, 1.05820424e-04, 1.06774289e-04,
       1.08793063e-04, 1.17409931e-04, 1.27056398e-04, 1.35215837e-04,
       1.41276015e-04, 1.45081652e-04, 1.44895732e-04, 1.49414862e-04,
       1.58612135e-04, 1.74264391e-04, 1.95595229e-04, 2.23147377e-04,
       2.49616526e-04, 2.74763437e-04, 2.99978651e-04, 3.27748270e-04,
       3.56348450e-04, 3.88847241e-04, 4.28850885e-04, 4.73865620e-04,
       5.11840353e-04, 5.73624029e-04, 6.29217907e-04, 6.98766067e-04,
       7.65557831e-04, 8.70885542e-04, 9.56911162e-04, 1.05143435e-03,
       1.15529449e-03, 1.26941388e-03, 1.39480592e-03, 1.53258412e-03,
       1.68397198e-03, 1.85031386e-03, 2.03308690e-03, 2.23391417e-03,
       2.45457905e-03, 2.69704109e-03, 2.96345340e-03, 3.25618179e-03,
       3.57782573e-03, 3.93124150e-03, 4.31956749e-03, 4.74625212e-03,
       5.21508443e-03, 5.73022776e-03, 6.29625668e-03, 6.91819764e-03,
       7.60157360e-03, 8.35245309e-03, 9.17750406e-03, 1.00840531e-02,
       1.10801507e-02, 1.21746422e-02, 1.33772470e-02, 1.46986445e-02,
       1.61505690e-02, 1.77459138e-02, 1.94988459e-02, 2.14249318e-02,
       2.35412754e-02, 2.58666703e-02, 2.84217665e-02, 3.12292539e-02]

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

def inharmonicity(X, gt, dgt, beta, _lambda, _iter, B, f0, N, NL):
    K = len(X)
    n = np.arange(1,N+1)
    f = np.zeros(len(n))

    for i in range(len(n)):
        f[i] = n[i]*f0*(1+B*n[i]**2)**0.5

    N = min(N,len(np.where(f<K-200)[0])) 
    n = np.arange(1,N+1)
    ftemp = np.zeros(len(n))
    a = np.zeros(len(n))
    for i in range(len(n)):
        ftemp[i] = f[i]
        a[i] = 1

    f = ftemp
    h = 1
    fk = np.arange(1,K+1)

    for itTime in range(_iter):

        it = itTime+1
        # choose the window function
        i = min(len(gt), int(ceil(it/5)))
        gtao = gt[i-1]
        dgtao = dgt[i-1]

        # calculate the reconstruction
        g = np.zeros((K,N))
        f = [int(round(freq)) for freq in f]

        for i in range(len(n)):
            g[f[i]-1,i] = 1
            g[:,i] = np.convolve(g[:,i],gtao,'same')

        g = g[:K,:]
        W = g.dot(a)+np.finfo(float).eps
        V = W.dot(h)

        # update a

        HV = V**(beta-1)*h
        HVX = V**(beta-2)*X*h
        P0a = HV.dot(g)
        Q0a = HVX.dot(g)
        a = a*Q0a/P0a

        # update W

        g = np.zeros((K,N))
        dg = np.zeros((K,N))

        for i in range(len(n)):
            g[f[i]-1,i] = 1
            g[:,i] = np.convolve(g[:,i],gtao,'same')
            dg[f[i]-1,i] = 1
            dg[:,i] = np.convolve(dg[:,i],dgtao,'same')

        g = g[:len(X),:]
        dg = dg[:len(X),:]
        W = g.dot(a)+np.finfo(float).eps
        V = W.dot(h)

        # update f

        nN = range(1,N+1)
        P0f = - a*((fk*HV).dot(dg)) - a*f*(HVX.dot(dg))
        Q0f = - a*((fk*HVX).dot(dg)) - a*f*(HV.dot(dg))

        P1f = np.zeros(len(nN))
        Q1f = np.zeros(len(nN))
        for i in range(len(nN)):
            P1f[i] = 2*f[i]
            Q1f[i] = 2*nN[i]*f0*(1+B*nN[i]**2)**0.5

        f = f*(Q0f+_lambda[int(ceil(it/100))-1]*Q1f)/(P0f+_lambda[int(ceil(it/100))-1]*P1f);

        # update B and F0

        ftemp = np.zeros(len(f))
        for i in range(len(f)):
            ftemp[i] = round(f[i])

        NLtemp = np.zeros(len(ftemp))
        for i in range(len(ftemp)):
            NLtemp[i] = NL[int(ftemp[i])]

        nNL = np.where(a > NLtemp)[0]
        if nNL.size == 0:
            nNL = np.asarray([1])
        else:
            nNL = nNL+1

        u = range(1,31)
        for i in range(len(u)):
            ftemp = np.zeros(len(nNL))
            for i in range(len(nNL)):
                ftemp[i] = f[nNL[i]-1]

            f0 = sum(ftemp*nNL*((1+B*nNL**2)**0.5))/sum(nNL**2*(1+B*nNL**2))

            ib = range(1,21)
            for b in range(len(ib)):
                P1B = f0*sum(nNL**4)
                Q1B = sum(nNL**3*ftemp/((1+B*nNL**2)**0.5))
                B = B*Q1B/P1B
    
    ftemp = np.zeros(len(f))
    for i in range(len(f)):
        ftemp[i] = round(f[i])
    f = ftemp
    
    for i in range(len(f)):
        idx, c = max(enumerate(X[int(f[i]-2-1):int(f[i]+2+1)]), key=operator.itemgetter(1))
        f[i]=f[i]+idx+1-3

    return a, f, B, f0, V

def estimate_inharmonicity(wav_file_path, midiNum,sr=48000):
    #wav_file = "IS-v96-m60.wav"
    #midiNum = 60
	# noteFileS = [index for index, value in enumerate(wav_file) if value == '/'][-1]
	# midiNumS = [index for index, value in enumerate(wav_file) if value == 'm'][-1]
	# midiNumE = [index for index, value in enumerate(wav_file) if value == '.'][-1]
	# midiNum = int(wav_file[midiNumS+1:midiNumE])
	# noteFile = wav_file[noteFileS+1:midiNumE]
    # x = MonoLoader(filename=wav_file)()

    fs = sr
    x, sr = librosa.load(wav_file_path, sr=fs)
    B = initial_B
    #B = B['B']

    fR = 1 # frequency resolution
    K = int(round(fs/fR/3)) # K is the number of frequency bins, frequency range 0-fs/3
    N = 30 # N is the number of partials
    # T is the number of time frames

    R = 88 # R is the number of piano notes
    r = range(R) # MIDI index 1:R 
    f0 = np.zeros(88);
    for i in r:
        f0[i] = 440*2**((i+1-49)/12)/fR
    # f0(r) = 440*2.^((r-49)/12) fundamental frequency for piano notes
    # f0 = f0/fR frequency index of f0

    beta = 1
    _lambda = np.asarray([0.125,5*10**(-3)])
    _iter = 150

    tao = np.asarray([1/60, 1/40, 1/30, 1/20, 1/10, 1/8, 1/6, 1/4, 1/2, 1]) # window size

    S = np.zeros((K,R))
    B1 = np.zeros(88)

    r = midiNum-20-1 # Piano note index
    onset = 0.5*fs

    # calculate the Discrete Fourier Transform
    frame = x[int(onset):int(onset+fs/2)]
    frame = np.hamming(fs/2)*frame
    X = abs(np.fft.fft(frame, int(fs/fR)));
    X = X[:K]
    #X = X/max(X)
    S[:,r] = X

    # Estimate the noise level:
    NL = np.zeros(K)
    fb = 300
    for k in range(len(X)):
        # [-fb/2 fb/2] median filter
        # NL[k] = np.median(X[max(1,int(k-round(fb/2/fR))):min(len(X),int(k+round(fb/2/fR)))])
        NL[k] = np.median(X[max(0,int(k-round(fb/2/fR))):min(len(X),int(k+round(fb/2/fR)))+1])
        NL[k] = NL[k]/(log(4)**0.5)*(-2*log(10**(-4)))**0.5;
    gt = []
    dgt = []

    # generate the mainlobes of hamming windows with different window sizes
    tao2 = tao[10-int(ceil(r/10))-1:]
    for i in range(len(tao2)):
        gt.append([])
        dgt.append([])
        t = tao2[i]
        fg = np.arange(-2/t,2/t+1,1)
        for j in fg:
            gt[i].append(sin(pi*j*t)/(pi*t*(j-t**2*j**3)));    
            dgt[i].append((pi*t*cos(pi*j*t)*(j-t**2*j**3)-sin(pi*j*t)*(1-3*t**2*j**2))/((pi*t*(j-t**2*j**3))**2)/j)
        
        gt[i][int(ceil(len(gt[i])/2))-1] = 1;
        gt[i][int(ceil(len(gt[i])/4))-1] = 1/2;
        gt[i][int(ceil(3*len(gt[i])/4))-1] = 1/2;
            
        dgt[i][int(ceil(len(gt[i])/2))-1] = (6-pi**2)/(3*pi)*t;
        dgt[i][int(ceil(len(gt[i])/4))-1]  = -(3*t)/(4*pi);
        dgt[i][int(ceil(3*len(gt[i])/4))-1] = -(3*t)/(4*pi);

    # based on "A parameteric model and estimation techniques for the inharmonicity and tuning of the piano"
    a, f, B1[r], f0[r], V = inharmonicity(X,gt,dgt,beta,_lambda,_iter,B[r],f0[r],N,NL)
    f0_final =  f0[r]*fR

    # print("For note MIDI-No.%s:"%(midiNum)) 
    # print("the estimated fundamental frequency is %s"%(f0_final))
    # print("the estimated inharmonicity coefficient is %s" %B1[r])
    # print(a)
    # print(f)
    # print(V)

    return f0_final, B1[r], a, f, V, x




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



st.title('Recording to Inharmonicity')

# with st.expander("Click to read more:"):

#     st.markdown(
#     """
#     - A real string has a bending stiffness.
#     - The restoring force for transverse vibrations is composed by the tension in the string and an additional wave length dependend part (dispersion).
#     - The partials do not follow a straight harmonic series:
#     """
#     )

#     st.latex(r''' f_n > n \cdot F_0 ''')


# kammerton = st.number_input("Choose a concert pitch:", value=440, step=1)
# st.write("The current concert pitch is ", kammerton, " Hz.")

kammerton = 440

def f(key):
    return np.round(kammerton * 2**((key-49)/12),4)

key = st.selectbox(
    "Select the key you will record:",
    note_names, index=48)

key_num = note_names.index(key)+1

#st.write("The current key is ", key, " with a fundamental frequency of", f(key_num), "Hz in Equal temperament.")

#st.subheader("Ideal String")

st.write("Record a single note (", key ,") on the piano to calculate the inharmonicity.")
audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    format="webm",
    callback=None,
    args=(),
    kwargs={},
    key=None
)
if audio:
    st.audio(audio['bytes'])

if audio:

    wrapper = BytesIO(audio['bytes'])

    #data = read(BytesIO(audio['bytes']))
    #st.line_chart(wrapper, x_label="Amplitude", y_label="Sample")
    #st.write(data)
    #st.write(wrapper)
#st.write(audio)

#st.write(data)
# wav_file_path = '../D1_PROD07_key49_take1_mono_50kHz.wav'

# x, sr = librosa.load(wav_file_path, sr=50000)

# st.audio(x)


# st.header("Inharmonicity Calculation")

# l = st.number_input("Insert string length [mm]:", value=402.00, min_value=40.00, max_value=2500.00, step=0.01)

# d = st.selectbox(
#     "Select a string diameter [mm]:",
#     string_diameters, index=11)

# st.markdown("__Attention__: For the inharmonicity calculation, the fundamental frequency of the string without stiffness would be needed, which is unknown. Therefore, the fundamental frequency of the real string is used instead. This introduces an error in the inharmonicity calculation.")

# def delta_inharmonicity(d, f, l): # d and l in mm
#     return 3.4 * 10**15 * d**2 / (f**2 * l**4)

# actual_delta_inharmonicity = np.round(delta_inharmonicity(d, f(key_num), l),5)

# st.write("The inharmonicity of the string after Young is $\delta = $", actual_delta_inharmonicity , "cent .")

# def convert_delta_to_B(delta,n):
#     return (2**((2*delta/1200))-1) / n**2

# actual_B_inharmonicity = np.round(convert_delta_to_B(actual_delta_inharmonicity, 1),5)

# st.write("The corresponding inharmonicity coefficient after Fletcher is $B=$ ", actual_B_inharmonicity, ".")

# factor = st.slider("Select a factor for $B$:", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# n = st.number_input("Insert number of harmonics:", value=20, min_value=1)

# def calculate_inharmonic_partials(f0, B, n):
#     return n* f0 * np.sqrt(1 + B * n**2) 

# list_of_inharmonic_partial_frequencies = [calculate_inharmonic_partials(f(key_num), actual_B_inharmonicity*factor, k) for k in np.arange(1,n+1,1)]

# damping_factor = st.slider("Select a damping factor:", min_value=0.0, max_value=3.0, value=.3, step=.1)


# #st.write("I'm ", age, "years old")

# #frequencies1 = [f(key_num) * k for k in np.arange(1,n+1,1)]  # Frequencies in Hz
# amplitudes = [0-k for k in np.arange(1,n+1,1)]  # Amplitudes in dB
# damping_factors = damping_factor*np.arange(n+1)  # Damping factors in dB/sec
# signal = generate_wav_file(list_of_inharmonic_partial_frequencies, amplitudes, damping_factors)

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

# def taylor_string_load(f, l, d, rho):
#     return (np.pi * rho * (f * l * d)**2)

# actual_load = np.round(taylor_string_load(f(key_num), l/1000, d/1000, rho),2)
# max_load = string_load_capacities[string_diameters.index(d)]

# percentage_of_max_load = np.round(actual_load/max_load*100,2)

# st.write("The actual load is ", actual_load, "N, which is ", percentage_of_max_load, "% of the maximum load capacity (", max_load, " N, including a safety factor of 0.75).")


# # df = pd.DataFrame({"Diameter (mm)": string_diameters, "Tensile strength (N/mm^2)": tensile_strengths, "Max load capacity (*0.75) (N)": string_load_capacities})

# # st.dataframe(df)

# st.header("String Stretching")

# def string_stretching(load, l, d, E):
#     return (load * l) / (d**2 * (np.pi/4) * E)

# actual_string_stretching = np.round(string_stretching(actual_load, l, d, E),4)

# actual_string_stretching_percent = np.round(actual_string_stretching/l*100,2)

# st.write("The actual string stretching is ", actual_string_stretching, "mm or ", actual_string_stretching_percent,  "%.")