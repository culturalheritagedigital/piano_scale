import numpy as np
import streamlit as st
from streamlit_mic_recorder import mic_recorder
#from io import BytesIO
import librosa
from math import log,ceil,pi,sin,cos
import operator
import pandas as pd

import numpy as np
import streamlit as st
from math import log,ceil,pi,sin,cos
import operator

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

def taylor_force(f, l, d, rho):
    return np.round((f**2 * l**2 * d**2 * np.pi * rho )/ 10**12 ,2)    
    """
    Calculates the force of a string using the Taylor formula.
    Parameters:
    :param f: frequency in Hz
    :param l: vibrating length in mm
    :param d: diameter in mm
    :param rho: density in kg/m^3
    Returns:
    :return F: force in N
    """

def B_to_delta(B,n):
    return 1200 * np.log2(np.sqrt(1+B * n**2))
    """
    Converts the inharmonicity coefficient B to the detuning factor delta.
    Parameters:
    :param B: inharmonicity coefficient
    :param n: overtone number
    Returns:
    :return delta: detuning factor in cents
    """

def delta_to_B(delta,n):
    return (2**((2*delta/1200))-1) / n**2
    """
    Converts the detuning factor delta to the inharmonicity coefficient B.
    Parameters:
    :param delta: detuning factor in cents
    :param n: overtone number
    Returns:
    :return B: inharmonicity coefficient
    """



delta_soll_laible = [ float("NaN")  ,  float("NaN")  ,  float("NaN")  ,  float("NaN")  ,  float("NaN")  ,  float("NaN")  ,  float("NaN")  , float("NaN")   ,
         float("NaN") ,  float("NaN")  ,   float("NaN") ,  float("NaN")  , float("NaN")   ,  float("NaN")  ,  float("NaN")  ,  float("NaN")  ,
         float("NaN") ,  float("NaN")  ,   float("NaN") ,   float("NaN") ,  0.049,  0.054,  0.059,  0.064,
        0.07 ,  0.076,  0.083,  0.091,  0.099,  0.108,  0.118,  0.128,
        0.14 ,  0.153,  0.166,  0.182,  0.198,  0.216,  0.235,  0.257,
        0.28 ,  0.305,  0.333,  0.363,  0.396,  0.432,  0.471,  0.514,
        0.56 ,  0.611,  0.666,  0.726,  0.792,  0.864,  0.942,  1.027,
        1.12 ,  1.221,  1.332,  1.452,  1.584,  1.727,  1.884,  2.054,
        2.24 ,  2.443,  2.664,  2.905,  3.168,  3.455,  3.767,  4.108,
        4.48 ,  4.885,  5.328,  5.81 ,  6.336,  6.909,  7.534,  8.216,
        8.96 ,  9.771, 10.655, 11.62 , 12.671, 13.818, 15.069, 16.433]

delta_soll_fenner = [ float("NaN")  ,  float("NaN")  ,  float("NaN")  , float("NaN")   ,  float("NaN")  ,  float("NaN")  ,  float("NaN")  , float("NaN")   ,
        float("NaN")  ,  float("NaN")  ,  float("NaN")  ,   float("NaN") , float("NaN")   ,  float("NaN")  ,  float("NaN")  ,  float("NaN")  ,
        float("NaN")  ,  float("NaN")  ,  float("NaN")  ,   float("NaN") ,  0.062,  0.067,  0.074,  0.08 ,
        0.088,  0.095,  0.104,  0.113,  0.124,  0.135,  0.147,  0.16 ,
        0.175,  0.191,  0.208,  0.227,  0.247,  0.27 ,  0.294,  0.321,
        0.35 ,  0.382,  0.416,  0.454,  0.495,  0.54 ,  0.589,  0.642,
        0.7  ,  0.763,  0.832,  0.908,  0.99 ,  1.08 ,  1.177,  1.284,
        1.4  ,  1.527,  1.665,  1.816,  1.98 ,  2.159,  2.355,  2.568,
        2.8  ,  3.053,  3.33 ,  3.631,  3.96 ,  4.318,  4.709,  5.135,
        5.6  ,  6.107,  6.66 ,  7.262,  7.92 ,  8.636,  9.418, 10.27 ,
       11.2  , 12.214, 13.319, 14.525, 15.839, 17.273, 18.836, 20.541]

def audio_bytes_to_numpy(audio_dict):
    if 'bytes' not in audio_dict:
        st.warning("Warnung: 'bytes' nicht in audio_dict gefunden.")
        return None, None
    
    audio_bytes = audio_dict['bytes']
    sample_rate = audio_dict.get('sample_rate', None)
    
    if not audio_bytes:
        st.warning("Warnung: audio_bytes ist leer.")
        return None, None
    
    try:
        # Lade WAV-Daten direkt in NumPy-Array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Normalisiere auf Bereich [-1, 1]
        audio_array_float = audio_array.astype(np.float32) / 32768.0
        
        return audio_array_float, sample_rate
    except Exception as e:
        st.error(f"Fehler bei der Verarbeitung der Audio-Daten: {e}")
        return None, None
    
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

def estimate_inharmonicity(wav_file_path, midiNum ,sr=48000):
    #wav_file = "IS-v96-m60.wav"
    #midiNum = 60
	# noteFileS = [index for index, value in enumerate(wav_file) if value == '/'][-1]
	# midiNumS = [index for index, value in enumerate(wav_file) if value == 'm'][-1]
	# midiNumE = [index for index, value in enumerate(wav_file) if value == '.'][-1]
	# midiNum = int(wav_file[midiNumS+1:midiNumE])
	# noteFile = wav_file[noteFileS+1:midiNumE]
    # x = MonoLoader(filename=wav_file)()

    fs = sr
    x = wav_file_path
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

kammerton = 443
def f(key):
    return np.round(kammerton * 2**((key-49)/12),4)

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


# Streamlit App
st.title("Inharmonicity estimation from recording")

st.write("Diese App berechnet die Inharmonizität eines aufgenommenen Klavierklangs:")
st.write("1. Wählen Sie die Taste aus, die Sie aufnehmen möchten.")

key = st.selectbox(
    "Taste:",
    note_names, index=48)

key_num = note_names.index(key)+1

st.write("The current key is ", key, " with a fundamental frequency of", f(key_num), "Hz in Equal temperament.")

st.write("2. Drücken Sie auf 'Start recording' und spielen Sie die Taste.")
with st.expander("Aufnahmehinweise:"):

    st.markdown("- Die Aufnahme sollte in einem ruhigen Raum erfolgen.")
    st.markdown("- Die Aufnahme darf nur den Ton __einer__ gespielte Taste enthalten.")
    st.markdown("- Vor dem Spielen der Taste sollte mind. ca. 0.5 Sekunden Stille aufgenommen werden.")
    st.markdown("- Die Aufnahme sollte mind. ca. 2 Sekunden lang sein.")
    st.markdown("__Abfolge:__ start recording - kurz warten - Taste spielen - stop recording")

st.write("3. Drücken Sie auf 'Stop recording' und warten Sie auf die Verarbeitung der Daten.")

audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    format="wav",
    key="audio_recorder"
)

if audio:
    st.audio(audio['bytes'])

    numpy_array, sample_rate = audio_bytes_to_numpy(audio)
    
    if numpy_array is not None:
        st.success("Recording was successful.")
        
        st.line_chart(numpy_array)
        st.write(f"Sample Rate: {sample_rate} [Hz], recording length: {np.round(numpy_array.shape[0]/sample_rate,2)} [s]")

        st.write("4. Hören Sie sich die Aufnahme an: Ist (nur) ein Klavierton zu hören? Ist am Anfang etwas Stille aufgenommen? Wenn nicht, wiederholen Sie die Aufnahme.")
        st.write("5. Drücken Sie auf 'Process audio data' um die Inharmonizität zu berechnen.")
        data_ok = st.button("Process audio data")

        if data_ok:
            # search the max position in numpy_array
            max_pos = np.argmax(numpy_array[100:])
            # cut the numpy_array at the max position
            numpy_array = numpy_array[max_pos:max_pos+sample_rate]
            #st.line_chart(numpy_array)
            four = np.abs(np.fft.fft(numpy_array))
            four = four/np.max(four)
            fourlog = 20*np.log10(four/np.max(four))
            st.line_chart(fourlog[:12000])

            f0, B, a, f, V, x = estimate_inharmonicity(numpy_array, key_num + 20, sr=sample_rate)
            st.write("Taste: " + str(key)) 
            st.write("Theoretische Grundfrequenz der idealen Saite: ", np.round(f0,2), "Hz")
            st.write("Reale Grundfrequenz der aufgenommenen Saite: ", f[0], "Hz")
            st.write("Inharmonizitätskoeffizient $B$ nach Fletcher:  ", np.round(B,6))
            st.write("Inharmonizitätskoeffizient $\delta$ nach Young:  ", np.round(B_to_delta(B,1),2), " Cent.")
            st.write("$\delta_{soll}$ nach Laible: ", delta_soll_laible[key_num], " Cent.")
            st.write("$\delta_{soll}$ nach Fenner: ", delta_soll_fenner[key_num], " Cent.")
      
    else:
        st.error("Error processing audio data.")
else:
    st.info("Waiting for audio recording...")