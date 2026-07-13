import streamlit as st
import os
import sys
import io
import queue
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import our custom modules
from risk_engine import run_monte_carlo_salsa, run_monte_carlo_coffee, run_monte_carlo_brewery
from agents_setup import run_strategic_crew
from colab_generator import generate_colab_notebook
from doc_parser import parse_business_file, extract_parameters_via_gemini

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Risk & Uncertainty Decision App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state keys for custom scenario if they don't exist
defaults = {
    'custom_title': "Proyecto de Negocio Personalizado",
    'custom_description': "Define aquí el contexto del negocio y los riesgos...",
    'custom_revenue': 100000.0,
    'custom_cogs': 40000.0,
    'custom_share': 0.40,
    'custom_fixed': 30000.0,
    'custom_risk_prob': 0.30,
    'custom_risk_increase': 0.50,
    'custom_low_contraction': 0.15,
    'custom_low_std': 0.05,
    'custom_high_spike': 0.40,
    'custom_high_std': 0.10
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Custom premium CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .metric-box {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1rem;
        flex: 1;
        margin: 0 0.5rem;
        text-align: center;
    }
    
    .metric-box-title {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-box-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    .metric-box-value.positive {
        color: #10B981;
    }
    
    .metric-box-value.negative {
        color: #EF4444;
    }
    
    .metric-box-value.neutral {
        color: #38BDF8;
    }
    
    .terminal-container {
        background-color: #090D16;
        color: #34D399;
        font-family: 'Courier New', Courier, monospace;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #10B981;
        max-height: 400px;
        overflow-y: auto;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Custom stdout redirector to capture logs live
class StreamlitRedirect:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""
        
    def write(self, text):
        # Clean terminal color codes or escape sequences often sent by CrewAI / LiteLLM
        cleaned_text = text.replace('\x1b[1;30m', '').replace('\x1b[0m', '').replace('\x1b[1;32m', '').replace('\x1b[1;31m', '').replace('\x1b[32m', '').replace('\x1b[1;36m', '').replace('\x1b[92m', '').replace('\x1b[99m', '')
        self.text += cleaned_text
        # Limit size to avoid browser crash
        if len(self.text) > 100000:
            self.text = self.text[-80000:]
        self.placeholder.code(self.text)
        
    def flush(self):
        pass

# Header Section
st.markdown("""
<div class="main-header">
    <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.8; font-weight: 600;">Sistemas de Soporte a Decisiones</span>
    <h1 style="margin: 0.5rem 0 0 0; font-weight: 700; font-size: 2.5rem;">Análisis de Riesgo & Decisiones bajo Incertidumbre</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">Plataforma multiagente inteligente impulsada por simulación cuantitativa de Monte Carlo y CrewAI.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Setup
st.sidebar.markdown("### 🔑 Configuración del Entorno")
gemini_key_placeholder = ""
if "GEMINI_API_KEY" in os.environ:
    gemini_key_placeholder = os.environ["GEMINI_API_KEY"]

api_key = st.sidebar.text_input(
    "Gemini API Key", 
    value=gemini_key_placeholder,
    type="password",
    help="Necesaria para ejecutar los agentes locales de CrewAI. Los datos se procesan en memoria y no se guardan."
)

model_option = st.sidebar.selectbox(
    "Modelo de Lenguaje (LLM)",
    ["gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro", "gemini/gemini-2.5-flash"],
    index=0,
    help="Modelo de Google Gemini a utilizar por los agentes."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Selección del Escenario")
scenario_choice = st.sidebar.selectbox(
    "Escenario de Negocio",
    [
        "Empresa de Salsas (Cancún)",
        "Exportadora de Café (Veracruz)",
        "Cervecería Artesanal (Baja California)",
        "Personalizado (Definir variables)"
    ]
)

# ----------------- SCENARIO INITIALIZATION & INPUTS -----------------

scenario_title = ""
scenario_description = ""
scenario_type = ""

# Keep track of active parameters to generate Colab notebooks
colab_params = {}

if scenario_choice == "Empresa de Salsas (Cancún)":
    scenario_type = "salsa"
    scenario_title = "Empresa de Salsas - Cancún, Quintana Roo"
    scenario_description = (
        "Una PYME fabricante de salsas artesanales en Cancún enfrenta volatilidad en el costo del chile jalapeño "
        "debido a factores climáticos extremos de fin de año (35% de probabilidad de desabasto o incremento del 45% en costos). "
        "Paralelamente, el negocio opera bajo la estacionalidad del turismo en Quintana Roo, "
        "con una contracción del 20% en la demanda local de restaurantes/hoteles en temporada baja (Septiembre - Noviembre) "
        "y un repunte del 60% en la temporada alta de fin de año (Diciembre)."
    )
    
    st.sidebar.markdown("### 🎛️ Parámetros del Escenario")
    baseline_revenue = st.sidebar.number_input("Ingresos Mensuales Base ($)", value=100000.0, step=1000.0)
    baseline_cogs = st.sidebar.number_input("Costo de Ventas Base (COGS) ($)", value=40000.0, step=1000.0)
    chile_share = st.sidebar.slider("Proporción de Chile Jalapeño en COGS", 0.05, 0.80, 0.30)
    fixed_costs = st.sidebar.number_input("Costos Fijos Mensuales ($)", value=30000.0, step=1000.0)
    
    st.sidebar.markdown("#### Factores de Incertidumbre")
    chile_risk_prob = st.sidebar.slider("Probabilidad de Crisis de Chile (0-1)", 0.0, 1.0, 0.35)
    chile_risk_increase = st.sidebar.slider("Aumento del Chile en Crisis (0-2)", 0.0, 2.0, 0.45)
    
    low_season_contraction_mean = st.sidebar.slider("Contracción Media Temp. Baja (0-1)", 0.0, 1.0, 0.20)
    low_season_contraction_std = st.sidebar.slider("Desv. Estándar Temp. Baja", 0.01, 0.20, 0.05)
    
    high_season_spike_mean = st.sidebar.slider("Aumento Medio Temp. Alta (0-2)", 0.0, 2.0, 0.60)
    high_season_spike_std = st.sidebar.slider("Desv. Estándar Temp. Alta", 0.01, 0.30, 0.10)
    
    simulations = st.sidebar.slider("Iteraciones Monte Carlo", 500, 15000, 10000, step=500)
    
    # Save variables for simulation run
    sim_args = {
        'baseline_revenue': baseline_revenue,
        'baseline_cogs': baseline_cogs,
        'chile_share': chile_share,
        'fixed_costs': fixed_costs,
        'chile_risk_prob': chile_risk_prob,
        'chile_risk_increase': chile_risk_increase,
        'low_season_contraction_mean': low_season_contraction_mean,
        'low_season_contraction_std': low_season_contraction_std,
        'high_season_spike_mean': high_season_spike_mean,
        'high_season_spike_std': high_season_spike_std,
        'num_simulations': simulations
    }
    
    colab_params = {
        'scenario_title': scenario_title,
        'scenario_description': scenario_description,
        'baseline_revenue': baseline_revenue,
        'baseline_cogs': baseline_cogs,
        'chile_share': chile_share,
        'fixed_costs': fixed_costs,
        'chile_risk_prob': chile_risk_prob,
        'chile_risk_increase': chile_risk_increase,
        'low_season_contraction_mean': low_season_contraction_mean,
        'low_season_contraction_std': low_season_contraction_std,
        'high_season_spike_mean': high_season_spike_mean,
        'high_season_spike_std': high_season_spike_std
    }

elif scenario_choice == "Exportadora de Café (Veracruz)":
    scenario_type = "coffee"
    scenario_title = "Exportadora de Café Orgánico - Veracruz"
    scenario_description = (
        "Una cooperativa de Veracruz exporta café gourmet a Estados Unidos y Europa. "
        "Su facturación está nominada en USD, por lo que la volatilidad cambiaria representa un riesgo de flujo (SD del 8% cambiario). "
        "Asimismo, existe un riesgo del 40% de que una sobreoferta en Brasil o Vietnam cause una caída del 25% en los precios internacionales del café. "
        "Finalmente, los costos logísticos internacionales de fletes marítimos tienen un 15% de probabilidad de incrementarse un 30%."
    )
    
    st.sidebar.markdown("### 🎛️ Parámetros del Escenario")
    baseline_revenue = st.sidebar.number_input("Ingresos Mensuales Base ($)", value=150000.0, step=1000.0)
    baseline_cogs = st.sidebar.number_input("Costo de Ventas Base (COGS) ($)", value=70000.0, step=1000.0)
    fixed_costs = st.sidebar.number_input("Costos Fijos Mensuales ($)", value=40000.0, step=1000.0)
    
    st.sidebar.markdown("#### Factores de Incertidumbre")
    exchange_rate_volatility = st.sidebar.slider("Volatilidad de Tipo de Cambio (SD %)", 0.01, 0.25, 0.08)
    coffee_drop_prob = st.sidebar.slider("Probabilidad Caída de Precio Café (0-1)", 0.0, 1.0, 0.40)
    coffee_drop_pct = st.sidebar.slider("Magnitud de Caída del Café (0-1)", 0.0, 0.60, 0.25)
    logistics_hike_prob = st.sidebar.slider("Probabilidad de Crisis Logística (0-1)", 0.0, 1.0, 0.15)
    logistics_hike_pct = st.sidebar.slider("Aumento del Flete en Crisis (0-1)", 0.0, 1.0, 0.30)
    
    simulations = st.sidebar.slider("Iteraciones Monte Carlo", 500, 15000, 10000, step=500)
    
    sim_args = {
        'baseline_revenue': baseline_revenue,
        'baseline_cogs': baseline_cogs,
        'fixed_costs': fixed_costs,
        'exchange_rate_volatility': exchange_rate_volatility,
        'coffee_drop_prob': coffee_drop_prob,
        'coffee_drop_pct': coffee_drop_pct,
        'logistics_hike_prob': logistics_hike_prob,
        'logistics_hike_pct': logistics_hike_pct,
        'num_simulations': simulations
    }
    
    colab_params = {
        'scenario_title': scenario_title,
        'scenario_description': scenario_description,
        'baseline_revenue': baseline_revenue,
        'baseline_cogs': baseline_cogs,
        'fixed_costs': fixed_costs,
        'exchange_rate_volatility': exchange_rate_volatility,
        'coffee_drop_prob': coffee_drop_prob,
        'coffee_drop_pct': coffee_drop_pct,
        'logistics_hike_prob': logistics_hike_prob,
        'logistics_hike_pct': logistics_hike_pct
    }

elif scenario_choice == "Cervecería Artesanal (Baja California)":
    scenario_type = "brewery"
    scenario_title = "Cervecería Artesanal Costera - Baja California"
    scenario_description = (
        "Una cervecería artesanal de Ensenada, Baja California, comercializa sus cervezas a nivel regional. "
        "El negocio sufre de escasez de agua, con un 20% de probabilidad de tener que comprar camiones cisterna que "
        "triplican el costo de su agua. Por otra parte, las tarifas y costos de la malta importada fluctúan entre "
        "un 10% y 30% adicionales (distribución triangular). Además, la demanda turística local (que representa un 40% de las ventas) "
        "tiene una variación normal con un crecimiento medio del 10% y una desviación estándar del 15%."
    )
    
    st.sidebar.markdown("### 🎛️ Parámetros del Escenario")
    baseline_revenue = st.sidebar.number_input("Ingresos Mensuales Base ($)", value=80000.0, step=1000.0)
    baseline_cogs = st.sidebar.number_input("Costo de Ventas Base (COGS) ($)", value=30000.0, step=1000.0)
    fixed_costs = st.sidebar.number_input("Costos Fijos Mensuales ($)", value=25000.0, step=1000.0)
    
    st.sidebar.markdown("#### Factores de Incertidumbre")
    water_drought_prob = st.sidebar.slider("Probabilidad de Sequía (Agua cara)", 0.0, 1.0, 0.20)
    water_cost_multiplier = st.sidebar.slider("Multiplicador Costo de Agua", 1.5, 5.0, 3.0)
    barley_hike_min = st.sidebar.slider("Aumento Mínimo Malta Importada", 0.0, 0.30, 0.10)
    barley_hike_max = st.sidebar.slider("Aumento Máximo Malta Importada", 0.15, 0.60, 0.30)
    
    tourism_mean = st.sidebar.slider("Crecimiento Medio Demanda Turística", -0.20, 0.50, 0.10)
    tourism_std = st.sidebar.slider("Volatilidad de la Demanda Turística (SD)", 0.01, 0.30, 0.15)
    
    simulations = st.sidebar.slider("Iteraciones Monte Carlo", 500, 15000, 10000, step=500)
    
    sim_args = {
        'baseline_revenue': baseline_revenue,
        'baseline_cogs': baseline_cogs,
        'fixed_costs': fixed_costs,
        'water_drought_prob': water_drought_prob,
        'water_cost_multiplier': water_cost_multiplier,
        'barley_hike_min': barley_hike_min,
        'barley_hike_max': barley_hike_max,
        'tourism_mean': tourism_mean,
        'tourism_std': tourism_std,
        'num_simulations': simulations
    }
    
    colab_params = {
        'scenario_title': scenario_title,
        'scenario_description': scenario_description,
        'baseline_revenue': baseline_revenue,
        'baseline_cogs': baseline_cogs,
        'fixed_costs': fixed_costs,
        'water_drought_prob': water_drought_prob,
        'water_cost_multiplier': water_cost_multiplier,
        'barley_hike_min': barley_hike_min,
        'barley_hike_max': barley_hike_max,
        'tourism_mean': tourism_mean,
        'tourism_std': tourism_std
    }

else: # Personalizado
    scenario_type = "salsa" # Use the salsa structure as template for customizable general inputs
    scenario_title = st.sidebar.text_input("Título del Escenario", key="custom_title")
    scenario_description = st.sidebar.text_area("Descripción del Escenario", key="custom_description")
    
    st.sidebar.markdown("### 🎛️ Parámetros Financieros Base")
    baseline_revenue = st.sidebar.number_input("Ingresos Mensuales Base ($)", step=1000.0, key="custom_revenue")
    baseline_cogs = st.sidebar.number_input("Costo de Ventas Base (COGS) ($)", step=1000.0, key="custom_cogs")
    chile_share = st.sidebar.slider("Proporción de Materia Prima Crítica en COGS", 0.05, 0.90, step=0.05, key="custom_share")
    fixed_costs = st.sidebar.number_input("Costos Fijos Mensuales ($)", step=1000.0, key="custom_fixed")
    
    st.sidebar.markdown("#### Factores de Incertidumbre")
    chile_risk_prob = st.sidebar.slider("Probabilidad de Crisis en Materia Prima (0-1)", 0.0, 1.0, step=0.05, key="custom_risk_prob")
    chile_risk_increase = st.sidebar.slider("Aumento en Crisis (0-2)", 0.0, 2.0, step=0.05, key="custom_risk_increase")
    
    low_season_contraction_mean = st.sidebar.slider("Contracción Media Demanda (0-1)", 0.0, 1.0, step=0.05, key="custom_low_contraction")
    low_season_contraction_std = st.sidebar.slider("Desv. Estándar Contracción", 0.01, 0.20, step=0.01, key="custom_low_std")
    
    high_season_spike_mean = st.sidebar.slider("Expansión Media Demanda (0-2)", 0.0, 2.0, step=0.05, key="custom_high_spike")
    high_season_spike_std = st.sidebar.slider("Desv. Estándar Expansión", 0.01, 0.30, step=0.01, key="custom_high_std")
    
    simulations = st.sidebar.slider("Iteraciones Monte Carlo", 500, 15000, 10000, step=500)
    
    sim_args = {
        'baseline_revenue': baseline_revenue,
        'baseline_cogs': baseline_cogs,
        'chile_share': chile_share,
        'fixed_costs': fixed_costs,
        'chile_risk_prob': chile_risk_prob,
        'chile_risk_increase': chile_risk_increase,
        'low_season_contraction_mean': low_season_contraction_mean,
        'low_season_contraction_std': low_season_contraction_std,
        'high_season_spike_mean': high_season_spike_mean,
        'high_season_spike_std': high_season_spike_std,
        'num_simulations': simulations
    }
    
    colab_params = {
        'scenario_title': scenario_title,
        'scenario_description': scenario_description,
        'baseline_revenue': baseline_revenue,
        'baseline_cogs': baseline_cogs,
        'chile_share': chile_share,
        'fixed_costs': fixed_costs,
        'chile_risk_prob': chile_risk_prob,
        'chile_risk_increase': chile_risk_increase,
        'low_season_contraction_mean': low_season_contraction_mean,
        'low_season_contraction_std': low_season_contraction_std,
        'high_season_spike_mean': high_season_spike_mean,
        'high_season_spike_std': high_season_spike_std
    }

# ----------------- TABS LAYOUT -----------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Escenario & Contexto", 
    "📊 Simulación de Monte Carlo", 
    "🤖 Análisis Multiagente (CrewAI)", 
    "☁️ Integración Google Colab"
])

with tab1:
    st.markdown(f"### 📋 Contexto Estratégico: {scenario_title}")
    st.info(scenario_description)
    
    if scenario_choice == "Personalizado (Definir variables)":
        st.markdown("### 📥 Cargar Insumos de Caso Real")
        st.write(
            "Sube un archivo con los datos financieros o descripción del negocio (formatos soportados: **CSV, Excel, PDF, Word**). "
            "Si ingresaste tu Gemini API Key en la barra lateral, utilizaremos inteligencia artificial para analizar el documento y rellenar automáticamente los parámetros de simulación."
        )
        
        uploaded_file = st.file_uploader(
            "Cargar archivo de insumo (máx. 5MB)", 
            type=["csv", "xlsx", "xls", "pdf", "docx", "doc"],
            key="uploaded_business_file"
        )
        
        if uploaded_file is not None:
            if 'last_uploaded_filename' not in st.session_state or st.session_state['last_uploaded_filename'] != uploaded_file.name:
                if not api_key:
                    st.warning(
                        "⚠️ Has subido un archivo, pero no has configurado tu **Gemini API Key** en la barra lateral. "
                        "Ingresa la clave para que Gemini pueda analizar el archivo y rellenar el formulario de forma automática. "
                        "De lo contrario, puedes rellenar los datos manualmente en la barra lateral."
                    )
                else:
                    file_bytes = uploaded_file.read()
                    with st.spinner(f"Analizando '{uploaded_file.name}' con Gemini AI y extrayendo variables..."):
                        try:
                            doc_text = parse_business_file(file_bytes, uploaded_file.name)
                            # Clean the selected model name (strip "gemini/" prefix for google-generativeai SDK compatibility)
                            clean_model_name = model_option.split('/')[-1] if '/' in model_option else model_option
                            extracted = extract_parameters_via_gemini(doc_text, uploaded_file.name, api_key, model_name=clean_model_name)
                            
                            # Update session state values
                            st.session_state['custom_title'] = str(extracted.get('scenario_title', 'Proyecto de Negocio Personalizado'))
                            st.session_state['custom_description'] = str(extracted.get('scenario_description', ''))
                            st.session_state['custom_revenue'] = float(extracted.get('baseline_revenue', 100000.0))
                            st.session_state['custom_cogs'] = float(extracted.get('baseline_cogs', 40000.0))
                            st.session_state['custom_share'] = float(extracted.get('chile_share', 0.40))
                            st.session_state['custom_fixed'] = float(extracted.get('fixed_costs', 30000.0))
                            st.session_state['custom_risk_prob'] = float(extracted.get('chile_risk_prob', 0.30))
                            st.session_state['custom_risk_increase'] = float(extracted.get('chile_risk_increase', 0.50))
                            st.session_state['custom_low_contraction'] = float(extracted.get('low_season_contraction_mean', 0.15))
                            st.session_state['custom_low_std'] = float(extracted.get('low_season_contraction_std', 0.05))
                            st.session_state['custom_high_spike'] = float(extracted.get('high_season_spike_mean', 0.40))
                            st.session_state['custom_high_std'] = float(extracted.get('high_season_spike_std', 0.10))
                            
                            st.session_state['last_uploaded_filename'] = uploaded_file.name
                            st.success(f"¡Datos del caso '{st.session_state['custom_title']}' extraídos con éxito! Revisa los parámetros en el panel izquierdo.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al analizar el archivo: {str(e)}")
    
    st.markdown("#### Entendiendo el Riesgo y la Incertidumbre en los Negocios")
    st.write(
        "En economía y finanzas corporativas, la toma de decisiones se clasifica en tres entornos:\n"
        "1. **Certeza**: Se conocen con precisión las variables del mercado.\n"
        "2. **Riesgo**: No se conocen los resultados exactos, pero se dispone de datos históricos para asociar **probabilidades** a los posibles eventos (ej. clima, tipo de cambio, estacionalidad).\n"
        "3. **Incertidumbre**: Los eventos futuros son desconocidos y no se cuenta con bases estadísticas sólidas para predecir su probabilidad.\n\n"
        "Esta aplicación aborda la **toma de decisiones bajo riesgo** a través de modelos estocásticos (Monte Carlo) "
        "e integra inteligencia agentica para formular estrategias operativas frente a la incertidumbre residual."
    )
    
    st.markdown("#### Flujo de Operaciones de la Aplicación")
    st.markdown("""
    ```mermaid
    graph TD
        A[Definir Variables y Probabilidades en la UI] --> B[Simular 10,000 Escenarios Financieros]
        B --> C[Calcular Métricas de Riesgo: VaR, CVaR, Margen]
        C --> D[Graficar Distribución de Beneficio e Impactos]
        D --> E[Alimentar Datos Cuantitativos al Agente Analista]
        E --> F[Analista de Riesgos detecta Puntos de Quiebre]
        F --> G[Agente Estratega diseña Plan de Mitigación Táctico]
        G --> H[Reporte Estratégico Multiagente Final]
    ```
    """)

with tab2:
    st.markdown("### 🎲 Simulación Monte Carlo y Métricas de Riesgo Financiero")
    st.write(
        "La simulación de Monte Carlo ejecuta miles de iteraciones aleatorias basándose en las distribuciones de probabilidad "
        "definidas para las variables de entrada. Esto nos permite observar el abanico completo de resultados posibles "
        "y calcular métricas de riesgo fundamentales para proteger la caja."
    )
    
    if st.button("🎲 Ejecutar Simulación Monte Carlo", type="primary", use_container_width=True):
        with st.spinner("Ejecutando 10,000 iteraciones financieras estocásticas..."):
            # Execute corresponding simulation engine
            if scenario_type == "salsa":
                df_sims, metrics, sensitivity = run_monte_carlo_salsa(**sim_args)
                if scenario_choice == "Personalizado (Definir variables)":
                    # Rename variables to generic business ones
                    sensitivity = {
                        'Costo de Materia Prima Crítica': sensitivity['Chile Jalapeño Price Factor'],
                        'Contracción de Demanda': sensitivity['Low Season Demand Factor'],
                        'Expansión de Demanda': sensitivity['High Season Demand Factor']
                    }
                    df_sims = df_sims.rename(columns={
                        'Chile_Factor': 'Factor_Costo_Materia_Prima',
                        'Low_Season_Demand': 'Factor_Demanda_Baja',
                        'High_Season_Demand': 'Factor_Demanda_Alta'
                    })
            elif scenario_type == "coffee":
                df_sims, metrics, sensitivity = run_monte_carlo_coffee(**sim_args)
            else:
                df_sims, metrics, sensitivity = run_monte_carlo_brewery(**sim_args)
                
            # Store in session state for other tabs
            st.session_state['sim_data'] = {
                'df_sims': df_sims,
                'metrics': metrics,
                'sensitivity': sensitivity,
                'scenario_title': scenario_title,
                'scenario_description': scenario_description
            }
            st.success("¡Simulación completada con éxito!")
            
    if 'sim_data' in st.session_state:
        sd = st.session_state['sim_data']
        df_sims = sd['df_sims']
        metrics = sd['metrics']
        sensitivity = sd['sensitivity']
        
        # Display Premium Metrics Row
        # VaR and CVaR formatting color
        var_color = "positive" if metrics['var_95'] >= 0 else "negative"
        cvar_color = "positive" if metrics['cvar_95'] >= 0 else "negative"
        loss_color = "negative" if metrics['prob_loss'] > 5 else "positive"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-box">
                <div class="metric-box-title">Beneficio Neto Promedio</div>
                <div class="metric-box-value neutral">${metrics['mean_profit']:,.2f} MXN</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">Margen Neto Esperado</div>
                <div class="metric-box-value neutral">{metrics['expected_margin']:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">Probabilidad de Pérdida (Profit < 0)</div>
                <div class="metric-box-value {loss_color}">{metrics['prob_loss']:.2f}%</div>
            </div>
        </div>
        <div class="metric-container">
            <div class="metric-box">
                <div class="metric-box-title">Value at Risk (VaR 95%)</div>
                <div class="metric-box-value {var_color}">${metrics['var_95']:,.2f} MXN</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:2px;">95% de probabilidad de ganar al menos esta cantidad.</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">Conditional Value at Risk (CVaR 95%)</div>
                <div class="metric-box-value {cvar_color}">${metrics['cvar_95']:,.2f} MXN</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:2px;">Beneficio promedio en el peor 5% de los escenarios de riesgo.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Explain VaR / CVaR
        with st.expander("📚 ¿Qué significan el VaR y el CVaR en mi negocio?"):
            st.markdown("""
            * **Value at Risk (VaR 95%)**: Es una medida estadística que cuantifica el riesgo financiero. Nos dice que, con un **95% de confianza**, las ganancias de la empresa no caerán por debajo de esta cifra durante el período analizado. Si el VaR es negativo (ej. -$5,000), significa que hay un 5% de probabilidad de tener pérdidas de $5,000 o más.
            * **Conditional Value at Risk (CVaR 95%)**: También llamado *Expected Shortfall*, es una medida de pérdidas extremas. Representa la media de los peores resultados en el 5% de la cola izquierda. Mientras el VaR te dice 'el umbral a partir de la cual empieza el peor escenario', el CVaR te responde: 'si las cosas salen realmente mal y cruzamos el umbral del VaR, ¿cuál será el promedio de la pérdida?'. Es vital para evitar la insolvencia.
            """)
            
        # Graphical Analysis
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### Histograma de Distribución de Utilidades")
            fig_hist = px.histogram(
                df_sims, x='Profit', nbins=50,
                labels={'Profit': 'Ganancia Neta ($ MXN)'},
                color_discrete_sequence=['#4F46E5']
            )
            # Add metric indicators
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#EF4444", annotation_text="Equilibrio (0)", annotation_position="top left")
            fig_hist.add_vline(x=metrics['var_95'], line_dash="solid", line_color="#F59E0B", annotation_text=f"VaR 95% (${metrics['var_95']:,.0f})", annotation_position="top left")
            
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#F8FAFC",
                title_font_size=16,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(showgrid=True, gridcolor="#334155"),
                yaxis=dict(showgrid=True, gridcolor="#334155")
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_chart2:
            st.markdown("#### Análisis de Sensibilidad (Tornado)")
            
            # Format sensitivity data
            sens_df = pd.DataFrame(list(sensitivity.items()), columns=['Variable', 'Correlación'])
            sens_df = sens_df.sort_values(by='Correlación', key=abs, ascending=True)
            
            fig_sens = px.bar(
                sens_df, y='Variable', x='Correlación', orientation='h',
                color='Correlación',
                color_continuous_scale=px.colors.diverging.RdBu,
                color_continuous_midpoint=0
            )
            fig_sens.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#F8FAFC",
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(title="Coeficiente de Correlación de Pearson", showgrid=True, gridcolor="#334155"),
                yaxis=dict(title="")
            )
            st.plotly_chart(fig_sens, use_container_width=True)
            st.caption("Un valor cercano a -1.0 indica que al subir esa variable, el beneficio baja drásticamente. Un valor cercano a +1.0 indica que el beneficio sube al incrementarse esa variable.")
            
        # Summary dataframe
        with st.expander("🔍 Ver tabla resumida de datos simulados (Muestra de 10 ejecuciones)"):
            # Dynamically build formatting dictionary based on existing columns in df_sims
            format_dict = {
                'Profit': '${:,.2f} MXN',
                'Revenue': '${:,.2f} MXN',
                'COGS': '${:,.2f} MXN'
            }
            for col in df_sims.columns:
                if col in ['Chile_Factor', 'Coffee_Price_Factor', 'Logistics_Factor', 'Water_Factor', 'Barley_Factor', 'Factor_Costo_Materia_Prima']:
                    format_dict[col] = '{:.2f}x'
                elif col in ['Low_Season_Demand', 'High_Season_Demand', 'Tourism_Factor', 'Factor_Demanda_Baja', 'Factor_Demanda_Alta']:
                    format_dict[col] = '{:.2%}'
                elif col in ['FX_Factor']:
                    format_dict[col] = '{:.3f}'
                    
            st.dataframe(df_sims.head(10).style.format(format_dict, na_rep="-"))
            
    else:
        st.warning("Haz clic en 'Ejecutar Simulación Monte Carlo' para generar y visualizar los resultados financieros.")

with tab3:
    st.markdown("### 🤖 Consulta Estratégica Multiagente (CrewAI)")
    st.write(
        "Este módulo utiliza a tus agentes consultores (`Analista de Datos` y `Estratega Comercial`) "
        "para examinar los números de la simulación de Monte Carlo e idear el plan de mitigación en Quintana Roo u otra región aplicable."
    )
    
    if 'sim_data' not in st.session_state:
        st.warning("Primero debes correr la simulación en la pestaña anterior para poder analizarla con los agentes.")
    else:
        # Prompt for Gemini Key check
        if not api_key:
            st.error("⚠️ Falta la Gemini API Key. Por favor, ingrésala en la barra lateral para poder ejecutar los agentes.")
        else:
            if st.button("🤖 Iniciar Ejecución del Equipo de Agentes", type="primary", use_container_width=True):
                # We have simulation data and API Key
                sd = st.session_state['sim_data']
                
                # Terminal output redirection area
                st.markdown("#### 📺 Consola de Razonamiento de Agentes (Live Log)")
                log_placeholder = st.empty()
                log_placeholder.code("Inicializando el entorno de agentes CrewAI...")
                
                # Setup capturing of stdout
                stdout_redirect = StreamlitRedirect(log_placeholder)
                old_stdout = sys.stdout
                sys.stdout = stdout_redirect
                
                try:
                    # Run the crew
                    report_result = run_strategic_crew(
                        api_key=api_key,
                        scenario_name=sd['scenario_title'],
                        scenario_description=sd['scenario_description'],
                        metrics=sd['metrics'],
                        sensitivity=sd['sensitivity'],
                        model_name=model_option
                    )
                    
                    # Restore stdout
                    sys.stdout = old_stdout
                    
                    # Store report in session
                    st.session_state['crew_report'] = report_result
                    st.success("¡Análisis estratégico completado!")
                except Exception as e:
                    sys.stdout = old_stdout
                    st.error(f"Error durante la ejecución de los agentes: {str(e)}")
                    st.info("Verifica que tu clave de API sea válida y que tengas conexión a internet.")

            # Show report if available
            if 'crew_report' in st.session_state:
                st.markdown("---")
                st.markdown("### 📋 REPORTE ESTRATÉGICO MULTIAGENTE FINAL")
                
                # Format output
                report_content = str(st.session_state['crew_report'])
                st.markdown(report_content)
                
                # Button to download markdown report
                st.download_button(
                    label="💾 Descargar Reporte en Markdown",
                    data=report_content,
                    file_name=f"reporte_estrategico_{scenario_choice.lower().replace(' ', '_')}.md",
                    mime="text/markdown"
                )

with tab4:
    st.markdown("### ☁️ Integración con Google Colab")
    st.write(
        "Google Colab es un entorno de cuadernos Jupyter gratuito en la nube que permite ejecutar código Python "
        "y modelos de machine learning directamente en tu navegador, sin necesidad de configurar nada localmente."
    )
    
    st.markdown("""
    #### 🚀 ¿Cómo usar el cuaderno generado en Google Colab?
    1. Descarga el archivo `.ipynb` interactivo haciendo clic en el botón de abajo.
    2. Entra en [Google Colab](https://colab.research.google.com/).
    3. Haz clic en **File > Upload notebook** y selecciona el archivo descargado.
    4. Sigue los comentarios del cuaderno para ejecutar las simulaciones y lanzar los agentes CrewAI en la nube de Google.
    """)
    
    # Check if we have colab params
    if not colab_params:
        st.info("Configura un escenario en la barra lateral para habilitar la descarga del cuaderno de Colab.")
    else:
        # Generate ipynb string
        notebook_json = generate_colab_notebook(scenario_type, colab_params)
        
        st.download_button(
            label="📥 Descargar Cuaderno Jupyter (.ipynb) Personalizado",
            data=notebook_json,
            file_name=f"analisis_riesgo_{scenario_type}_colab.ipynb",
            mime="application/x-ipynb+json",
            use_container_width=True,
            type="primary"
        )
        
        st.success("¡Cuaderno Jupyter listo para descargar! Incluye tus configuraciones actuales.")
        
        with st.expander("📄 Ver estructura interna del Notebook generado (JSON)"):
            st.code(notebook_json[:1500] + "\n\n... [Contenido truncado para visualización] ...")
