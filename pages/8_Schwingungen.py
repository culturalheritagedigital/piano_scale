import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Gedämpfte Schwingungen",
    page_icon="〰️",
    layout="wide"
)

def generate_damped_tone(frequency, A0, delta, duration=3.0, n_harmonics=10):
    """Generiert einen gedämpften Ton mit Obertönen"""
    sample_rate = 48000
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples, endpoint=False)
    
    signal = np.zeros(num_samples)
    
    # Generiere Obertöne
    for k in range(1, n_harmonics + 1):
        freq_k = frequency * k
        amplitude_k = 1.0 / k
        damping_k = delta * k
        
        decay = np.exp(-damping_k * time)
        wave = amplitude_k * np.sin(2 * np.pi * freq_k * time) * decay
        signal += wave
    
    # Fadeout am Ende
    hamm = np.hamming(48000)[24000:48000]
    ones = np.ones(int(sample_rate * (duration - 0.5)))
    fadeout = np.append(ones, hamm)
    
    if len(signal) > len(fadeout):
        signal = signal[:len(fadeout)]
    
    signal = signal * fadeout
    
    # Normalisierung
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    
    signal = (32767 * signal).astype(np.int16)
    
    return signal, time

st.title('Gedämpfte Schwingungen')

st.markdown("""
Eine gedämpfte Schwingung tritt auf, wenn ein schwingendes System Energie an seine Umgebung abgibt, 
was zu einem allmählichen Abklingen der Amplitude über die Zeit führt. 
Dies ist in vielen realen Systemen der Fall, wie z.B. bei einer schwingenden Saite eines Musikinstruments, die durch Luftwiderstand und innere Reibung gedämpft wird.

**Mathematische Beschreibung:** 
$$y(t) = A_0 \cdot \sin(\omega_d t + \\varphi_0) \cdot e^{-\delta t}$$

wobei:
- $A_0$ = Anfangsamplitude
- $\delta$ = Dämpfungskonstante
- $\omega_d$ = Kreisfrequenz
- $\\varphi_0$ = Nullphasenwinkel
""")

# Tabs für verschiedene Ansichten
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Interaktive Visualisierung", 
    "🔊 Akustisches Beispiel",
    "📈 Vergleich verschiedener Dämpfungen",
    "🧮 Berechnungen"
])

with tab1:
    st.header("Visualisierung einer gedämpften Schwingung")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Parameter einstellen")
        A0 = st.slider("Anfangsamplitude $A_0$ [cm]:", 
                      min_value=0.5, max_value=10.0, value=5.0, step=0.5)
        
        delta = st.slider("Dämpfungskonstante $\delta$ [s⁻¹]:", 
                         min_value=0.0, max_value=2.0, value=0.3, step=0.05)
        
        frequency = st.slider("Frequenz $f$ [Hz]:", 
                            min_value=0.5, max_value=5.0, value=1.0, step=0.1)
        
        phi0 = st.slider("Nullphasenwinkel $\\varphi_0$ [rad]:", 
                        min_value=0.0, max_value=2*np.pi, value=0.0, step=0.1)
        
        duration = st.slider("Darstellungsdauer [s]:", 
                           min_value=5, max_value=30, value=15, step=1)
        
        show_envelope = st.checkbox("Zeige Einhüllende", value=True)
    
    with col1:
        # Berechnung der Schwingung
        omega_d = 2 * np.pi * frequency
        t = np.linspace(0, duration, 1000)
        
        # Gedämpfte Schwingung
        y = A0 * np.sin(omega_d * t + phi0) * np.exp(-delta * t)
        
        # Einhüllende
        envelope_pos = A0 * np.exp(-delta * t)
        envelope_neg = -A0 * np.exp(-delta * t)
        
        # DataFrame für Plotting erstellen
        if show_envelope:
            df = pd.DataFrame({
                'Gedämpfte Schwingung': y,
                'Einhüllende +': envelope_pos,
                'Einhüllende -': envelope_neg
            }, index=t)
        else:
            df = pd.DataFrame({
                'Gedämpfte Schwingung': y
            }, index=t)
        
        st.line_chart(df)
        
        st.caption("**Achsen:** X-Achse = Zeit [s], Y-Achse = Auslenkung [cm] | **Legende:** Blaue Linie = Gedämpfte Schwingung; Rote Linien = Einhüllende $A(t) = A_0 \cdot e^{-\delta t}$")
        
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
        
        # Zusätzliche Informationen
        if delta > 0:
            st.info(f"""
            **Physikalische Interpretation:**
            - Nach {tau:.2f} s ist die Amplitude auf **{100/np.e:.1f}%** (≈37%) abgefallen
            - Nach {T_half:.2f} s ist die Amplitude auf **50%** abgefallen
            - Nach {3*tau:.2f} s ist die Amplitude auf **{100/np.e**3:.1f}%** (≈5%) abgefallen
            """)

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
        
        st.write(f"**Frequenz:** {tone_frequency:.2f} Hz")
        
        tone_delta = st.slider("Dämpfungskonstante $\delta$ [s⁻¹] für den Ton:", 
                              min_value=0.1, max_value=3.0, value=0.8, step=0.1,
                              key="tone_delta")
        
        tone_duration = st.slider("Tondauer [s]:", 
                                 min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                                 key="tone_duration")
        
        n_harmonics = st.slider("Anzahl der Obertöne:", 
                               min_value=1, max_value=20, value=10, step=1)
        
        # Generiere Ton
        signal_combined, time_audio = generate_damped_tone(
            tone_frequency, 1.0, tone_delta, tone_duration, n_harmonics
        )
        
        st.audio(signal_combined, format="audio/wav", sample_rate=48000)
        
        st.success(f"""
        **Generierter Ton:**
        - Note: **{selected_note}**
        - Grundfrequenz: **{tone_frequency:.2f} Hz**
        - Anzahl Obertöne: **{n_harmonics}**
        """)
    
    with col2:
        st.subheader("Visualisierung des Tons")
        
        # Zeige die ersten 0.2 Sekunden für bessere Sichtbarkeit
        display_duration = min(0.2, tone_duration)
        samples_to_display = int(48000 * display_duration)
        time_display = time_audio[:samples_to_display]
        signal_display = signal_combined[:samples_to_display] / 32767.0
        
        # DataFrame für Plotting
        df_audio = pd.DataFrame({
            'Amplitude': signal_display
        }, index=time_display)
        
        st.line_chart(df_audio)
        st.caption(f"Zeitverlauf des Tons {selected_note} (erste {display_duration} s) | X-Achse = Zeit [s], Y-Achse = Amplitude")
        
        st.info(f"""
        **Physikalische Parameter:**
        - Grundfrequenz: **{tone_frequency:.2f} Hz**
        - Dämpfungskonstante: **{tone_delta:.2f} s⁻¹**
        - Abklingzeit τ: **{1/tone_delta:.2f} s**
        - Halbwertszeit T₁/₂: **{0.693/tone_delta:.2f} s**
        
        **Obertöne:** Die Obertöne dämpfen schneller ab (δₖ = k·δ),
        was dem natürlichen Verhalten von Saiten entspricht.
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
    
    # DataFrame für Plotting
    df_comp = pd.DataFrame({
        f'Schwache Dämpfung (δ₁={delta1:.2f} s⁻¹)': y1,
        f'Mittlere Dämpfung (δ₂={delta2:.2f} s⁻¹)': y2,
        f'Starke Dämpfung (δ₃={delta3:.2f} s⁻¹)': y3
    }, index=t_comp)
    
    st.line_chart(df_comp)
    
    st.caption("**Beobachtung:** Je größer die Dämpfungskonstante δ, desto schneller klingt die Schwingung ab. | X-Achse = Zeit [s], Y-Achse = Auslenkung [cm]")
    
    # Vergleichstabelle
    st.subheader("Charakteristische Zeiten im Vergleich")
    
    comparison_data = {
        'Parameter': ['Dämpfungskonstante δ [s⁻¹]', 
                     'Abklingzeit τ = 1/δ [s]', 
                     'Halbwertszeit T₁/₂ [s]',
                     'Zeit bis 10% von A₀ [s]',
                     'Zeit bis 1% von A₀ [s]'],
        'Schwache Dämpfung': [
            f'{delta1:.3f}',
            f'{1/delta1:.2f}',
            f'{0.693/delta1:.2f}',
            f'{np.log(10)/delta1:.2f}',
            f'{np.log(100)/delta1:.2f}'
        ],
        'Mittlere Dämpfung': [
            f'{delta2:.3f}',
            f'{1/delta2:.2f}',
            f'{0.693/delta2:.2f}',
            f'{np.log(10)/delta2:.2f}',
            f'{np.log(100)/delta2:.2f}'
        ],
        'Starke Dämpfung': [
            f'{delta3:.3f}',
            f'{1/delta3:.2f}',
            f'{0.693/delta3:.2f}',
            f'{np.log(10)/delta3:.2f}',
            f'{np.log(100)/delta3:.2f}'
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
            st.latex(r'\frac{A(t_1)}{A(t_0)} = e^{-\delta (t_1 - t_0)}')
            st.latex(r'\delta = -\frac{\ln(A(t_1)/A(t_0))}{t_1 - t_0}')
            st.latex(rf'\delta = -\frac{{\ln({A_t1:.2f}/{A_t0:.2f})}}{{{t1:.2f} - {t0:.2f}}} = {delta_calculated:.4f} \text{{ s}}^{{-1}}')
            
            # Weitere Größen
            tau_calc = 1/delta_calculated
            T_half_calc = 0.693/delta_calculated
            
            st.write(f"**Abklingzeit:** τ = 1/δ = {tau_calc:.3f} s")
            st.write(f"**Halbwertszeit:** T₁/₂ = 0.693/δ = {T_half_calc:.3f} s")
            
            # Visualisierung der berechneten Schwingung
            st.write("**Visualisierung der berechneten Schwingung:**")
            t_vis = np.linspace(t0, max(10, t1*2), 500)
            y_vis = A_t0 * np.exp(-delta_calculated * (t_vis - t0))
            
            df_vis = pd.DataFrame({
                'Amplitude [cm]': y_vis
            }, index=t_vis)
            
            st.line_chart(df_vis)
            st.caption(f"Exponentielles Abklingen mit δ = {delta_calculated:.4f} s⁻¹ | X-Achse = Zeit [s], Y-Achse = Amplitude [cm]")
            
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
        
        A_n = st.number_input("Amplitude Aₙ [cm]:", 
                             min_value=0.1, max_value=10.0, value=5.0, step=0.1,
                             key="A_n")
        
        A_n1 = st.number_input("Folgende Amplitude Aₙ₊₁ [cm]:", 
                              min_value=0.1, max_value=10.0, value=4.1, step=0.1,
                              key="A_n1")
        
        T_period = st.number_input("Periodendauer Td [s]:", 
                                  min_value=0.01, max_value=10.0, value=0.4, step=0.01,
                                  key="T_period")
        
        if A_n > A_n1:
            Lambda = np.log(A_n / A_n1)
            delta_from_Lambda = Lambda / T_period
            
            st.success(f"**Logarithmisches Dekrement:** Λ = {Lambda:.4f}")
            st.success(f"**Dämpfungskonstante:** δ = Λ/Td = {delta_from_Lambda:.4f} s⁻¹")
            
            # Weitere Berechnungen
            tau_lambda = 1/delta_from_Lambda
            T_half_lambda = 0.693/delta_from_Lambda
            
            st.write(f"**Abklingzeit:** τ = {tau_lambda:.3f} s")
            st.write(f"**Halbwertszeit:** T₁/₂ = {T_half_lambda:.3f} s")
            
            st.write("**Interpretation:**")
            if Lambda < 0.1:
                st.info("✅ Sehr schwache Dämpfung - die Schwingung klingt langsam ab.")
            elif Lambda < 0.5:
                st.info("✅ Moderate Dämpfung - typisch für gut schwingende Systeme.")
            else:
                st.info("⚠️ Starke Dämpfung - die Schwingung klingt schnell ab.")
            
            # Anzahl Schwingungen bis auf 50% und 10%
            n_50 = T_half_lambda / T_period
            n_10 = np.log(10) / delta_from_Lambda / T_period
            
            st.write(f"""
            **Praktische Werte:**
            - Nach **{n_50:.1f} Schwingungen** ist die Amplitude auf 50% abgefallen
            - Nach **{n_10:.1f} Schwingungen** ist die Amplitude auf 10% abgefallen
            """)
            
        else:
            st.warning("Bitte stellen Sie sicher, dass Aₙ > Aₙ₊₁")

# Zusätzliche Informationen in der Sidebar
# st.sidebar.header("ℹ️ Typische Werte")
# st.sidebar.markdown("""
# **Klaviersaite (Mittellage, A4):**
# - δ ≈ 0.5 - 2.0 s⁻¹
# - τ ≈ 0.5 - 2.0 s
# - Nachhall: 3-8 Sekunden

# **Stimmgabel:**
# - δ ≈ 0.05 - 0.2 s⁻¹
# - τ ≈ 5 - 20 s
# - Sehr lange Nachhalldauer

# **Glocke:**
# - δ ≈ 0.01 - 0.05 s⁻¹
# - τ ≈ 20 - 100 s
# - Extrem lange Nachhalldauer

# **Basssaite (E1):**
# - δ ≈ 0.3 - 1.0 s⁻¹
# - τ ≈ 1 - 3 s
# - Längerer Nachhall als hohe Töne
# """)

# st.sidebar.markdown("---")
# st.sidebar.header("📚 Wichtige Formeln")
# st.sidebar.latex(r'y(t) = A_0 \cdot e^{-\delta t} \cdot \sin(\omega t)')
# st.sidebar.latex(r'\tau = \frac{1}{\delta}')
# st.sidebar.latex(r'T_{1/2} = \frac{0.693}{\delta}')
# st.sidebar.latex(r'\Lambda = \delta \cdot T_d')

# # st.sidebar.markdown("---")
# st.sidebar.info("""
# **Hinweis:** Diese App dient der Visualisierung gedämpfter Schwingungen 
# für den Unterricht in Akustik und ist speziell für angehende Klavierbaumeister konzipiert.

# Basierend auf dem Arbeitsblatt "Gedämpfte Schwingungen" (TRI 1.3).
# """)
