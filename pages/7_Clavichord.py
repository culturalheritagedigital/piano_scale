import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Sonare",
    page_icon="🎹",
)

def create_interval_table(scale_length=650):
    # Definiere die Intervalle
    data = {
        'Halbtonschritte': range(13),  # 0 bis 12
        'Intervall': [
            'Grundton',
            'kleine Sekunde',
            'große Sekunde',
            'kleine Terz',
            'große Terz',
            'Quarte',
            'Tritonus',
            'Quinte',
            'kleine Sexte',
            'große Sexte',
            'kleine Septime',
            'große Septime',
            'Oktave'
        ]
    }
    
    # Berechne die absoluten Längen
    data['absolute Länge (mm)'] = [
        scale_length - (scale_length / (2 ** (n/12))) 
        if n > 0 
        else 0 
        for n in data['Halbtonschritte']
    ]
    
    # Berechne die relativen Längen
    data['relative Länge (%)'] = [
        (length / scale_length) * 100 
        if length > 0 
        else 0 
        for length in data['absolute Länge (mm)']
    ]
    
    # Erstelle DataFrame
    df = pd.DataFrame(data)
    
    # Formatiere die Zahlen
    df['absolute Länge (mm)'] = df['absolute Länge (mm)'].round(2)
    df['relative Länge (%)'] = df['relative Länge (%)'].round(2)
    
    return df

st.title('Clavichord')

st.subheader("Tangentenpositionen")

with st.expander("Mehr Infos:"):
    st.markdown(
    """
Die Tabelle zeigt:

- Die Halbtonschritte von 0 (Grundton) bis 12 (Oktave)
- Die entsprechenden musikalischen Intervallbezeichnungen
- Die absolute Position der Tangente vom Anhangstift aus in Millimetern
- Die relative Position als Prozentsatz der Gesamtmensur

Die 0-Position (Grundton) entspricht dem Anhangstift. Alle anderen Positionen geben den Abstand vom Anhangstift an, an dem die Tangente platziert werden muss, um das entsprechende Intervall zu erzeugen.

**Berechnung der Tangentenposition**

In gleichstufig temperierter Stimmung teilt jeder Halbton die Frequenz im Verhältnis $2^{1/12}$. Die schwingende Saitenlänge für einen Ton $n$ Halbtöne über dem Grundton beträgt:
    """
    )
    st.latex(r"L_n = \frac{L_0}{2^{n/12}}")
    st.markdown(
    """
wobei $L_0$ die Gesamtmensur (schwingende Saitenlänge beim Grundton) ist.

Der Abstand der Tangente vom Anhangstift ergibt sich als Differenz zur Gesamtmensur:
    """
    )
    st.latex(r"d_n = L_0 - L_n = L_0 \left(1 - \frac{1}{2^{n/12}}\right)")
    st.markdown(
    """
- $n = 0$: Grundton — Tangente liegt am Anhangstift ($d = 0$)
- $n = 12$: Oktave — Tangente halbiert die Saite ($d = L_0 / 2$)
    """
    )

saitenlaenge = st.number_input("Geben Sie die Länge der Saite an:", value=650, step=1)

st.write("Bundpositionen für eine Mensur von ", saitenlaenge, " mm:")

df = create_interval_table(saitenlaenge)
st.dataframe(df)