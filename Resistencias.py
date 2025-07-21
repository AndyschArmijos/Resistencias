import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Calculadora de Viga Simplemente Apoyada",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Calculadora de Resistencia de Viga Simplemente Apoyada")
st.markdown("---")

# Sidebar para parámetros de entrada
st.sidebar.header("📊 Parámetros de la Viga")

# Parámetros geométricos
L = st.sidebar.number_input("Longitud de la viga (m)", min_value=0.1, max_value=50.0, value=6.0, step=0.1)
b = st.sidebar.number_input("Base de la sección (cm)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
h = st.sidebar.number_input("Altura de la sección (cm)", min_value=1.0, max_value=150.0, value=40.0, step=1.0)

# Cargas
st.sidebar.subheader("🔽 Cargas")
tipo_carga = st.sidebar.selectbox("Tipo de carga", ["Carga puntual central", "Carga distribuida uniforme", "Ambas cargas"])

P = 0
w = 0

if tipo_carga in ["Carga puntual central", "Ambas cargas"]:
    P = st.sidebar.number_input("Carga puntual P (kN)", min_value=0.0, value=50.0, step=1.0)

if tipo_carga in ["Carga distribuida uniforme", "Ambas cargas"]:
    w = st.sidebar.number_input("Carga distribuida w (kN/m)", min_value=0.0, value=10.0, step=0.5)

# Material
st.sidebar.subheader("🧱 Propiedades del Material")
fy = st.sidebar.number_input("Resistencia a la fluencia fy (MPa)", min_value=50.0, max_value=600.0, value=250.0, step=10.0)
fc = st.sidebar.number_input("Resistencia del concreto f'c (MPa)", min_value=10.0, max_value=100.0, value=21.0, step=1.0)

# Conversión de unidades
b_m = b / 100  # cm a m
h_m = h / 100  # cm a m
P_N = P * 1000  # kN a N
w_N = w * 1000  # kN/m a N/m
fy_Pa = fy * 1e6  # MPa a Pa

# Cálculos principales
def calcular_momento_cortante(x, L, P, w):
    """Calcula momento y cortante en una posición x"""
    # Reacciones
    if P > 0 and w > 0:
        Ra = Rb = (P + w * L) / 2
    elif P > 0:
        Ra = Rb = P / 2
    elif w > 0:
        Ra = Rb = w * L / 2
    else:
        Ra = Rb = 0
    
    # Cortante
    if x <= L/2:
        V = Ra - w * x
        if P > 0 and x >= L/2:
            V = V - P
    else:
        V = -Rb + w * (L - x)
    
    # Momento
    M = Ra * x - w * x**2 / 2
    if P > 0 and x >= L/2:
        M = M - P * (x - L/2)
    
    return M, V

# Propiedades de la sección
A = b_m * h_m  # Área
I = b_m * h_m**3 / 12  # Momento de inercia
c = h_m / 2  # Distancia al centroide
S = I / c  # Módulo de sección

# Cálculo de momentos máximos
x_vals = np.linspace(0, L, 1000)
momentos = []
cortantes = []

for x in x_vals:
    M, V = calcular_momento_cortante(x, L, P_N, w_N)
    momentos.append(M)
    cortantes.append(V)

M_max = max(momentos)
V_max = max([abs(v) for v in cortantes])

# Esfuerzos
sigma_max = M_max / S  # Esfuerzo normal máximo
tau_max = 1.5 * V_max / A  # Esfuerzo cortante máximo (aproximado para sección rectangular)

# Layout principal
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Resultados del Análisis")
    
    # Mostrar propiedades geométricas
    st.subheader("📐 Propiedades Geométricas")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Área (cm²)", f"{A*10000:.1f}")
        st.metric("Momento de Inercia (cm⁴)", f"{I*100000000:.0f}")
    with col_b:
        st.metric("Módulo de Sección (cm³)", f"{S*1000000:.0f}")
        st.metric("Altura Útil (cm)", f"{h:.1f}")
    
    st.subheader("🔍 Esfuerzos Máximos")
    col_c, col_d = st.columns(2)
    with col_c:
        st.metric("Momento Máximo", f"{M_max/1000:.2f} kN⋅m")
        st.metric("Esfuerzo Normal σ", f"{sigma_max/1e6:.2f} MPa")
    with col_d:
        st.metric("Cortante Máximo", f"{V_max/1000:.2f} kN")
        st.metric("Esfuerzo Cortante τ", f"{tau_max/1e6:.2f} MPa")
    
    # Verificación de resistencia
    st.subheader("✅ Verificación de Resistencia")
    factor_seguridad_flexion = fy_Pa / sigma_max if sigma_max > 0 else float('inf')
    factor_seguridad_cortante = (fy_Pa * 0.6) / tau_max if tau_max > 0 else float('inf')
    
    if factor_seguridad_flexion >= 1.67:
        st.success(f"✅ Flexión: Factor de seguridad = {factor_seguridad_flexion:.2f}")
    else:
        st.error(f"❌ Flexión: Factor de seguridad = {factor_seguridad_flexion:.2f} (< 1.67)")
    
    if factor_seguridad_cortante >= 1.67:
        st.success(f"✅ Cortante: Factor de seguridad = {factor_seguridad_cortante:.2f}")
    else:
        st.error(f"❌ Cortante: Factor de seguridad = {factor_seguridad_cortante:.2f} (< 1.67)")

with col2:
    st.header("📊 Diagramas")
    
    # Crear gráficos
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Diagrama de la viga
    ax1.plot([0, L], [0, 0], 'k-', linewidth=8, label='Viga')
    ax1.plot([0, 0], [-0.1, 0.1], 'k-', linewidth=4)  # Apoyo izquierdo
    ax1.plot([L, L], [-0.1, 0.1], 'k-', linewidth=4)  # Apoyo derecho
    
    # Dibujar cargas
    if P > 0:
        ax1.arrow(L/2, 0.3, 0, -0.2, head_width=L*0.02, head_length=0.03, fc='red', ec='red')
        ax1.text(L/2, 0.35, f'P = {P} kN', ha='center', fontsize=10)
    
    if w > 0:
        for i in range(int(L*10)):
            x_arrow = i/10
            if x_arrow <= L:
                ax1.arrow(x_arrow, 0.15, 0, -0.1, head_width=L*0.01, head_length=0.015, fc='blue', ec='blue')
        ax1.text(L/2, 0.2, f'w = {w} kN/m', ha='center', fontsize=10)
    
    ax1.set_xlim(-L*0.1, L*1.1)
    ax1.set_ylim(-0.2, 0.5)
    ax1.set_title('Esquema de la Viga', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitud (m)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Diagrama de momento flector
    ax2.plot(x_vals, np.array(momentos)/1000, 'b-', linewidth=2, label='Momento')
    ax2.fill_between(x_vals, np.array(momentos)/1000, alpha=0.3)
    ax2.set_title('Diagrama de Momento Flector', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Posición (m)')
    ax2.set_ylabel('Momento (kN⋅m)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Diagrama de fuerza cortante
    ax3.plot(x_vals, np.array(cortantes)/1000, 'r-', linewidth=2, label='Cortante')
    ax3.fill_between(x_vals, np.array(cortantes)/1000, alpha=0.3)
    ax3.set_title('Diagrama de Fuerza Cortante', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Posición (m)')
    ax3.set_ylabel('Cortante (kN)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig)

# Tabla de valores
if st.checkbox("📋 Mostrar tabla de valores detallada"):
    st.subheader("Tabla de Valores")
    n_points = 21
    x_table = np.linspace(0, L, n_points)
    tabla_datos = []
    
    for x in x_table:
        M, V = calcular_momento_cortante(x, L, P_N, w_N)
        tabla_datos.append({
            'Posición (m)': f"{x:.2f}",
            'Momento (kN⋅m)': f"{M/1000:.2f}",
            'Cortante (kN)': f"{V/1000:.2f}"
        })
    
    df = pd.DataFrame(tabla_datos)
    st.dataframe(df, use_container_width=True)

# Información adicional
with st.expander("ℹ️ Información sobre cálculos"):
    st.markdown("""
    ### Fórmulas utilizadas:
    
    **Propiedades geométricas:**
    - Área: A = b × h
    - Momento de inercia: I = b × h³ / 12
    - Módulo de sección: S = I / (h/2)
    
    **Esfuerzos:**
    - Esfuerzo normal: σ = M / S
    - Esfuerzo cortante: τ = 1.5 × V / A (sección rectangular)
    
    **Factor de seguridad recomendado:** 1.67 para estructuras de acero
    
    **Nota:** Los cálculos asumen comportamiento elástico lineal y sección rectangular homogénea.
    """)

st.markdown("---")
st.markdown("*Desarrollado para análisis estructural básico de vigas simplemente apoyadas*")