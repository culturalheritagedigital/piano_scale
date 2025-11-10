import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile

st.set_page_config(
    page_title="Gedämpfte Schwingungen",
    page_icon="〰️",
    layout="wide"
)

def generate_damped_tone(frequency, A0, delta, duration=3.0):
    """Generiert einen gedämpften Ton"""
    sample_rate = 48000
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Gedämpfte Schwingung
    decay = np.exp(-delta * time)
    signal = A0 * np.sin(2 * np.pi * frequency * time) * decay
    
    # Fadeout am Ende
    hamm = np.hamming(48000)[24000:48000]
    ones = np.ones(int(sample_rate * (duration - 0.5)))
    fadeout = np.append(ones, hamm)
    
    # Normalisierung
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    
    signal = signal * fadeout
    signal = (32767 * signal).astype(np.int16)
    
    return signal, time

st.title('Gedämpfte Schwingungen')

st.markdown("""
Diese App visualisiert gedämpfte harmonische Schwingungen, wie sie bei Klaviersaiten, 
Stimmgabeln und anderen schwingenden Systemen auftreten.

**Mathematische Beschreibung:** 
$$y(t) = A_0 \cdot \sin(\omega_d t + \varphi_0) \cdot e^{-\delta t}$$

wobei:
- $A_0$ = Anfangsamplitude
- $\delta$ = Dämpfungskonstante
- $\omega_d$ = Kreisfrequenz
- $\varphi_0$ = Nullphasenwinkel
""")

# Sidebar für Parameter
st.sidebar.header("Parameter der Schwingung")

# Tabs für verschiedene Ansichten
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Interaktive Visualisierung", 
    "🔊 Akustisches Beispiel",
    "📈 Vergleich verschiedener Dämpfungen",
    "🧮 Berechnungen"
])

with tab1:
    st.header("Interaktive Visualisierung einer gedämpften Schwingung")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Parameter einstellen")
        A0 = st.slider("Anfangsamplitude $A_0$ [cm]:", 
                      min_value=0.5, max_value=10.0, value=5.0, step=0.5)
        
        delta = st.slider("Dämpfungskonstante $\delta$ [s⁻¹]:", 
                         min_value=0.0, max_value=2.0, value=0.3, step=0.05)
        
        frequency = st.slider("Frequenz $f$ [Hz]:", 
                            min_value=0.5, max_value=5.0, value=1.0, step=0.1)
        
        phi0 = st.slider("Nullphasenwinkel $\varphi_0$ [rad]:", 
                        min_value=0.0, max_value=2*np.pi, value=0.0, step=0.1)
        
        duration = st.slider("Darstellungsdauer [s]:", 
                           min_value=5, max_value=30, value=15, step=1)
    
    with col1:
        # Berechnung der Schwingung
        omega_d = 2 * np.pi * frequency
        t = np.linspace(0, duration, 1000)
        
        # Gedämpfte Schwingung
        y = A0 * np.sin(omega_d * t + phi0) * np.exp(-delta * t)
        
        # Einhüllende
        envelope_pos = A0 * np.exp(-delta * t)
        envelope_neg = -A0 * np.exp(-delta * t)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Einhüllende (gestrichelt)
        ax.plot(t, envelope_pos, 'r--', linewidth=2, label='Einhüllende: $A(t) = A_0 \cdot e^{-\delta t}$')
        ax.plot(t, envelope_neg, 'r--', linewidth=2)
        
        # Gedämpfte Schwingung
        ax.plot(t, y, 'b-', linewidth=1.5, label='Gedämpfte Schwingung')
        
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Zeit $t$ [s]', fontsize=12)
        ax.set_ylabel('Auslenkung $y$ [cm]', fontsize=12)
        ax.set_title('Gedämpfte harmonische Schwingung', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xlim(0, duration)
        
        st.pyplot(fig)
        plt.close()
        
        # Zusätzliche Berechnungen
        st.subheader("Berechnete Werte")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            T_d = 1/frequency if frequency > 0 else 0
            st.metric("Periodendauer $T_d$", f"{T_d:.3f} s")
            
        with col_b:
            tau = 1/delta if delta > 0 else np.inf
            st.metric("Abklingzeit $\\tau = 1/\delta$", 
                     f"{tau:.2f} s" if tau != np.inf else "∞")
        
        with col_c:
            T_half = 0.693/delta if delta > 0 else np.inf
            st.metric("Halbwertszeit $T_{1/2}$", 
                     f"{T_half:.2f} s" if T_half != np.inf else "∞")

with tab2:
    st.header("Akustisches Beispiel: Gedämpfter Ton")
    
    st.write("""
    Hören Sie sich an, wie verschiedene Dämpfungskonstanten den Klang beeinflussen.
    Eine Klaviersaite wird modelliert mit verschiedenen Dämpfungen.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Tonparameter")
        
        # Notenauswahl
        note_names = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
        note_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        
        selected_note = st.selectbox("Wählen Sie einen Ton:", note_names, index=5)
        tone_frequency = note_freqs[note_names.index(selected_note)]
        
        st.write(f"Frequenz: {tone_frequency:.2f} Hz")
        
        tone_delta = st.slider("Dämpfungskonstante $\delta$ [s⁻¹] für den Ton:", 
                              min_value=0.1, max_value=3.0, value=0.8, step=0.1,
                              key="tone_delta")
        
        tone_duration = st.slider("Tondauer [s]:", 
                                 min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                                 key="tone_duration")
        
        # Generiere Ton mit Teiltönen (harmonischen)
        n_harmonics = 10
        signal_combined = np.zeros(int(48000 * tone_duration))
        time_audio = np.linspace(0, tone_duration, int(48000 * tone_duration))
        
        for k in range(1, n_harmonics + 1):
            freq_k = tone_frequency * k
            amplitude_k = 1.0 / k  # Abnehmende Amplitude für Obertöne
            damping_k = tone_delta * k  # Höhere Obertöne dämpfen schneller
            
            decay = np.exp(-damping_k * time_audio)
            wave = amplitude_k * np.sin(2 * np.pi * freq_k * time_audio) * decay
            signal_combined += wave
        
        # Fadeout
        hamm = np.hamming(48000)[24000:48000]
        ones = np.ones(int(48000 * (tone_duration - 0.5)))
        fadeout = np.append(ones, hamm)
        
        if len(signal_combined) > len(fadeout):
            signal_combined = signal_combined[:len(fadeout)]
        
        signal_combined = signal_combined * fadeout
        
        # Normalisierung
        if np.max(np.abs(signal_combined)) > 0:
            signal_combined = signal_combined / np.max(np.abs(signal_combined))
        signal_combined = (32767 * signal_combined).astype(np.int16)
        
        st.audio(signal_combined, format="audio/wav", sample_rate=48000)
    
    with col2:
        st.subheader("Visualisierung des Tons")
        
        # Zeige die ersten 0.5 Sekunden
        display_duration = min(0.5, tone_duration)
        samples_to_display = int(48000 * display_duration)
        time_display = time_audio[:samples_to_display]
        signal_display = signal_combined[:samples_to_display] / 32767.0
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(time_display, signal_display, 'b-', linewidth=0.8)
        ax2.set_xlabel('Zeit [s]', fontsize=11)
        ax2.set_ylabel('Amplitude', fontsize=11)
        ax2.set_title(f'Zeitverlauf des Tons {selected_note} (erste {display_duration} s)', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, display_duration)
        
        st.pyplot(fig2)
        plt.close()
        
        st.info(f"""
        **Physikalische Parameter:**
        - Grundfrequenz: {tone_frequency:.2f} Hz
        - Dämpfungskonstante: {tone_delta:.2f} s⁻¹
        - Abklingzeit: {1/tone_delta:.2f} s
        - Halbwertszeit: {0.693/tone_delta:.2f} s
        """)

with tab3:
    st.header("Vergleich verschiedener Dämpfungsstärken")
    
    st.write("""
    Vergleichen Sie verschiedene Dämpfungskonstanten in einer Grafik.
    Dies zeigt deutlich, wie die Dämpfung das Abklingverhalten beeinflusst.
    """)
    
    # Parameter für Vergleich
    A0_comp = 5.0
    freq_comp = 1.0
    omega_comp = 2 * np.pi * freq_comp
    t_comp = np.linspace(0, 15, 1000)
    
    # Verschiedene Dämpfungen
    st.subheader("Wählen Sie drei Dämpfungskonstanten zum Vergleich")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delta1 = st.slider("Schwache Dämpfung $\delta_1$ [s⁻¹]:", 
                          min_value=0.05, max_value=0.5, value=0.1, step=0.05,
                          key="delta1")
    
    with col2:
        delta2 = st.slider("Mittlere Dämpfung $\delta_2$ [s⁻¹]:", 
                          min_value=0.2, max_value=1.0, value=0.4, step=0.05,
                          key="delta2")
    
    with col3:
        delta3 = st.slider("Starke Dämpfung $\delta_3$ [s⁻¹]:", 
                          min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                          key="delta3")
    
    # Berechne Schwingungen
    y1 = A0_comp * np.sin(omega_comp * t_comp) * np.exp(-delta1 * t_comp)
    y2 = A0_comp * np.sin(omega_comp * t_comp) * np.exp(-delta2 * t_comp)
    y3 = A0_comp * np.sin(omega_comp * t_comp) * np.exp(-delta3 * t_comp)
    
    # Plotting
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    
    ax3.plot(t_comp, y1, 'b-', linewidth=2, 
            label=f'Schwache Dämpfung: $\delta_1 = {delta1:.2f}$ s$^{{-1}}$')
    ax3.plot(t_comp, y2, 'g-', linewidth=2, 
            label=f'Mittlere Dämpfung: $\delta_2 = {delta2:.2f}$ s$^{{-1}}$')
    ax3.plot(t_comp, y3, 'r-', linewidth=2, 
            label=f'Starke Dämpfung: $\delta_3 = {delta3:.2f}$ s$^{{-1}}$')
    
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax3.set_xlabel('Zeit $t$ [s]', fontsize=13)
    ax3.set_ylabel('Auslenkung $y$ [cm]', fontsize=13)
    ax3.set_title('Vergleich verschiedener Dämpfungsstärken', fontsize=15, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12, loc='upper right')
    ax3.set_xlim(0, 15)
    
    st.pyplot(fig3)
    plt.close()
    
    # Vergleichstabelle
    st.subheader("Charakteristische Zeiten im Vergleich")
    
    comparison_data = {
        'Parameter': ['Dämpfungskonstante δ [s⁻¹]', 
                     'Abklingzeit τ = 1/δ [s]', 
                     'Halbwertszeit T₁/₂ [s]',
                     'Zeit bis 10% von A₀ [s]'],
        'Schwache Dämpfung': [
            f'{delta1:.3f}',
            f'{1/delta1:.2f}',
            f'{0.693/delta1:.2f}',
            f'{np.log(10)/delta1:.2f}'
        ],
        'Mittlere Dämpfung': [
            f'{delta2:.3f}',
            f'{1/delta2:.2f}',
            f'{0.693/delta2:.2f}',
            f'{np.log(10)/delta2:.2f}'
        ],
        'Starke Dämpfung': [
            f'{delta3:.3f}',
            f'{1/delta3:.2f}',
            f'{0.693/delta3:.2f}',
            f'{np.log(10)/delta3:.2f}'
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.table(df_comparison)

with tab4:
    st.header("Berechnungen und logarithmisches Dekrement")
    
    st.write("""
    Berechnen Sie wichtige Kenngrößen einer gedämpften Schwingung.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Berechnung der Dämpfungskonstante")
        
        st.write("**Gegeben:** Zwei Amplitudenwerte zu verschiedenen Zeiten")
        
        A_t0 = st.number_input("Amplitude zur Zeit t₀ [cm]:", 
                              min_value=0.1, max_value=10.0, value=5.0, step=0.1)
        
        t0 = st.number_input("Zeitpunkt t₀ [s]:", 
                            min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        
        A_t1 = st.number_input("Amplitude zur Zeit t₁ [cm]:", 
                              min_value=0.01, max_value=10.0, value=2.0, step=0.1)
        
        t1 = st.number_input("Zeitpunkt t₁ [s]:", 
                            min_value=0.1, max_value=20.0, value=3.0, step=0.1)
        
        if t1 > t0 and A_t0 > A_t1:
            # Berechnung
            delta_calculated = -np.log(A_t1/A_t0) / (t1 - t0)
            
            st.success(f"**Berechnete Dämpfungskonstante:** δ = {delta_calculated:.4f} s⁻¹")
            
            st.write("**Rechenweg:**")
            st.latex(r'A(t) = A_0 \cdot e^{-\delta t}')
            st.latex(rf'\frac{{A(t_1)}}{{A(t_0)}} = e^{{-\delta (t_1 - t_0)}}')
            st.latex(rf'\delta = -\frac{{\ln(A(t_1)/A(t_0))}}{{t_1 - t_0}} = -\frac{{\ln({A_t1:.2f}/{A_t0:.2f})}}{{{t1:.2f} - {t0:.2f}}} = {delta_calculated:.4f} \text{{ s}}^{{-1}}')
            
            # Weitere Größen
            tau_calc = 1/delta_calculated
            T_half_calc = 0.693/delta_calculated
            
            st.write(f"**Abklingzeit:** τ = 1/δ = {tau_calc:.3f} s")
            st.write(f"**Halbwertszeit:** T₁/₂ = 0.693/δ = {T_half_calc:.3f} s")
        else:
            st.warning("Bitte stellen Sie sicher, dass t₁ > t₀ und A(t₀) > A(t₁)")
    
    with col2:
        st.subheader("Logarithmisches Dekrement")
        
        st.write("""
        Das logarithmische Dekrement Λ ist der natürliche Logarithmus des Verhältnisses 
        zweier aufeinanderfolgender Amplitudenmaxima:
        """)
        
        st.latex(r'\Lambda = \ln\left(\frac{A_n}{A_{n+1}}\right)')
        
        st.write("**Zusammenhang mit der Dämpfungskonstante:**")
        st.latex(r'\Lambda = \delta \cdot T_d')
        
        st.write("---")
        
        st.write("**Berechnung des logarithmischen Dekrements:**")
        
        A_n = st.number_input("Amplitude A_n [cm]:", 
                             min_value=0.1, max_value=10.0, value=5.0, step=0.1,
                             key="A_n")
        
        A_n1 = st.number_input("Folgende Amplitude A_(n+1) [cm]:", 
                              min_value=0.1, max_value=10.0, value=4.1, step=0.1,
                              key="A_n1")
        
        T_period = st.number_input("Periodendauer T_d [s]:", 
                                  min_value=0.01, max_value=10.0, value=0.4, step=0.01,
                                  key="T_period")
        
        if A_n > A_n1:
            Lambda = np.log(A_n / A_n1)
            delta_from_Lambda = Lambda / T_period
            
            st.success(f"**Logarithmisches Dekrement:** Λ = {Lambda:.4f}")
            st.success(f"**Dämpfungskonstante:** δ = Λ/T_d = {delta_from_Lambda:.4f} s⁻¹")
            
            st.write("**Interpretation:**")
            if Lambda < 0.1:
                st.info("Sehr schwache Dämpfung - die Schwingung klingt langsam ab.")
            elif Lambda < 0.5:
                st.info("Moderate Dämpfung - typisch für gut schwingende Systeme.")
            else:
                st.info("Starke Dämpfung - die Schwingung klingt schnell ab.")
        else:
            st.warning("Bitte stellen Sie sicher, dass A_n > A_(n+1)")

# Zusätzliche Informationen in der Sidebar
st.sidebar.markdown("---")
st.sidebar.header("Typische Werte")
st.sidebar.markdown("""
**Klaviersaite (Mittellage, A4):**
- δ ≈ 0.5 - 2.0 s⁻¹
- τ ≈ 0.5 - 2.0 s

**Stimmgabel:**
- δ ≈ 0.05 - 0.2 s⁻¹
- τ ≈ 5 - 20 s

**Glocke:**
- δ ≈ 0.01 - 0.05 s⁻¹
- τ ≈ 20 - 100 s
""")

st.sidebar.markdown("---")
# st.sidebar.info("""
# **Hinweis:** Diese App dient der Visualisierung gedämpfter Schwingungen 
# für den Unterricht in Akustik und ist speziell für angehende Klavierbaumeister konzipiert.
# """)