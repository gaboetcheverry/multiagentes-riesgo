import streamlit as st
import os
import sys

# Disable telemetry to prevent environment/encoding issues on Streamlit Cloud
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Force Streamlit Cloud to load fresh versions of local modules from disk instead of using stale in-memory cache
for module_name in ['risk_engine', 'agents_setup', 'colab_generator', 'doc_parser']:
    if module_name in sys.modules:
        del sys.modules[module_name]
import io
import queue
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Force stdout and stderr to UTF-8 to prevent encoding errors in restricted terminal environments
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    except Exception:
        pass

# Import our custom modules
try:
    from risk_engine import run_monte_carlo_generic
    from colab_generator import generate_colab_notebook
    from doc_parser import parse_business_file, extract_parameters_via_gemini, clean_to_ascii
    import_error_main = None
except Exception as e:
    import_error_main = e

try:
    from agents_setup import run_strategic_crew
    has_crewai = True
    crewai_import_error = None
except Exception as e:
    has_crewai = False
    crewai_import_error = e

if import_error_main is not None:
    import traceback
    st.error("Error al importar módulos principales (risk_engine, colab_generator, doc_parser).")
    st.exception(import_error_main)
    st.stop()

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
    'custom_description': "Define aquí el contexto del negocio, qué insumo o materia prima es volátil y los riesgos de estacionalidad...",
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
        cleaned_text = text.replace('\x1b[1;30m', '').replace('\x1b[0m', '').replace('\x1b[1;32m', '').replace('\x1b[1;31m', '').replace('\x1b[32m', '').replace('\x1b[1;36m', '').replace('\x1b[92m', '').replace('\x1b[99m', '')
        self.text += cleaned_text
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

gemini_key_placeholder = os.environ.get("GEMINI_API_KEY", "")
openai_key_placeholder = os.environ.get("OPENAI_API_KEY", "")

api_provider = st.sidebar.selectbox(
    "Proveedor de IA Activo",
    ["Google Gemini", "OpenAI"],
    index=0,
    help="Elige qué proveedor utilizar tanto para extraer datos de los archivos como para ejecutar los agentes."
)

api_key = ""
openai_key = ""

if api_provider == "Google Gemini":
    api_key = st.sidebar.text_input(
        "Gemini API Key", 
        value=gemini_key_placeholder,
        type="password",
        help="Necesaria para ejecutar los agentes y analizar archivos con Google Gemini."
    )
    openai_key = openai_key_placeholder
else:
    openai_key = st.sidebar.text_input(
        "OpenAI API Key", 
        value=openai_key_placeholder,
        type="password",
        help="Necesaria para ejecutar los agentes y analizar archivos con OpenAI."
    )
    api_key = gemini_key_placeholder

# Optional secondary key input
with st.sidebar.expander("🔑 Configurar Clave Secundaria (Opcional)"):
    if api_provider == "Google Gemini":
        openai_key = st.sidebar.text_input("OpenAI API Key", value=openai_key, type="password")
    else:
        api_key = st.sidebar.text_input("Gemini API Key", value=api_key, type="password")

# Set the environment variables in memory
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

# Model options based on active provider
if api_provider == "Google Gemini":
    model_options = ["gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro", "gemini/gemini-2.5-flash"]
else:
    model_options = ["gpt-4o-mini", "gpt-4o", "o3-mini"]

model_option = st.sidebar.selectbox(
    "Modelo de Lenguaje (LLM)",
    model_options,
    index=0,
    help="Modelo de Inteligencia Artificial a utilizar por los agentes y el parsing de archivos."
)

# Set the scenario choice permanently to Personalizado
scenario_choice = "Personalizado"
scenario_type = "generic"

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Parámetros del Escenario")

# UI controls directly bound to the custom parameters
scenario_title = st.sidebar.text_input("Título del Escenario", value=st.session_state['custom_title'])
scenario_description = st.sidebar.text_area("Descripción del Escenario", value=st.session_state['custom_description'])

st.sidebar.markdown("#### 💰 Estructura Financiera Base")
baseline_revenue = st.sidebar.number_input("Ingresos Mensuales Base ($)", value=st.session_state['custom_revenue'], step=1000.0)
baseline_cogs = st.sidebar.number_input("Costo de Ventas Base (COGS) ($)", value=st.session_state['custom_cogs'], step=1000.0)
raw_material_share = st.sidebar.slider("Proporción de Insumo Crítico en COGS", 0.05, 0.90, value=st.session_state['custom_share'], step=0.05)
fixed_costs = st.sidebar.number_input("Costos Fijos Mensuales ($)", value=st.session_state['custom_fixed'], step=1000.0)

st.sidebar.markdown("#### ⚡ Factores de Volatilidad / Riesgo")
raw_material_risk_prob = st.sidebar.slider("Probabilidad de Crisis de Insumo (0-1)", 0.0, 1.0, value=st.session_state['custom_risk_prob'], step=0.05)
raw_material_risk_increase = st.sidebar.slider("Aumento del Insumo en Crisis (0-2)", 0.0, 2.0, value=st.session_state['custom_risk_increase'], step=0.05)

low_season_contraction_mean = st.sidebar.slider("Contracción Media Temp. Baja (0-1)", 0.0, 1.0, value=st.session_state['custom_low_contraction'], step=0.05)
low_season_contraction_std = st.sidebar.slider("Desv. Estándar Temp. Baja", 0.01, 0.20, value=st.session_state['custom_low_std'], step=0.01)

high_season_spike_mean = st.sidebar.slider("Aumento Medio Temp. Alta (0-2)", 0.0, 2.0, value=st.session_state['custom_high_spike'], step=0.05)
high_season_spike_std = st.sidebar.slider("Desv. Estándar Temp. Alta", 0.01, 0.30, value=st.session_state['custom_high_std'], step=0.01)

simulations = st.sidebar.slider("Iteraciones Monte Carlo", 500, 15000, 10000, step=500)

# Sync UI changes back to session state to make it sticky
st.session_state['custom_title'] = scenario_title
st.session_state['custom_description'] = scenario_description
st.session_state['custom_revenue'] = baseline_revenue
st.session_state['custom_cogs'] = baseline_cogs
st.session_state['custom_share'] = raw_material_share
st.session_state['custom_fixed'] = fixed_costs
st.session_state['custom_risk_prob'] = raw_material_risk_prob
st.session_state['custom_risk_increase'] = raw_material_risk_increase
st.session_state['custom_low_contraction'] = low_season_contraction_mean
st.session_state['custom_low_std'] = low_season_contraction_std
st.session_state['custom_high_spike'] = high_season_spike_mean
st.session_state['custom_high_std'] = high_season_spike_std

sim_args = {
    'baseline_revenue': baseline_revenue,
    'baseline_cogs': baseline_cogs,
    'raw_material_share': raw_material_share,
    'fixed_costs': fixed_costs,
    'raw_material_risk_prob': raw_material_risk_prob,
    'raw_material_risk_increase': raw_material_risk_increase,
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
    'raw_material_share': raw_material_share,
    'fixed_costs': fixed_costs,
    'raw_material_risk_prob': raw_material_risk_prob,
    'raw_material_risk_increase': raw_material_risk_increase,
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
    
    st.markdown("### 📥 Cargar Insumos de Caso Real")
    st.write(
        "Sube un archivo con los datos financieros o descripción del negocio (formatos soportados: **CSV, Excel, PDF, Word**). "
        "Si ingresaste tu API Key en la barra lateral, utilizaremos inteligencia artificial para analizar el documento y rellenar automáticamente los parámetros de simulación."
    )
    
    uploaded_file = st.file_uploader(
        "Cargar archivo de insumo (máx. 5MB)", 
        type=["csv", "xlsx", "xls", "pdf", "docx", "doc"],
        key="uploaded_business_file"
    )
    
    if uploaded_file is not None:
        safe_filename = clean_to_ascii(uploaded_file.name)
        if 'last_uploaded_filename' not in st.session_state or st.session_state['last_uploaded_filename'] != safe_filename:
            active_key = openai_key if api_provider == "OpenAI" else api_key
            provider_name = "OpenAI" if api_provider == "OpenAI" else "Gemini"
            
            if not active_key:
                st.warning(
                    f"⚠️ Has subido un archivo, pero no has configurado tu **{provider_name} API Key** en la barra lateral. "
                    f"Ingresa la clave para que {provider_name} pueda analizar el archivo y rellenar el formulario de forma automática. "
                    "De lo contrario, puedes rellenar los datos manualmente en la barra lateral."
                )
            else:
                file_bytes = uploaded_file.read()
                with st.spinner(f"Analizando '{safe_filename}' con {provider_name} AI y extrayendo variables..."):
                    try:
                        doc_text = parse_business_file(file_bytes, safe_filename)
                        clean_model_name = model_option.split('/')[-1] if '/' in model_option else model_option
                        extracted = extract_parameters_via_gemini(doc_text, safe_filename, active_key, model_name=clean_model_name)
                        
                        # Update session state values
                        st.session_state['custom_title'] = str(extracted.get('scenario_title', 'Proyecto de Negocio Personalizado'))
                        st.session_state['custom_description'] = str(extracted.get('scenario_description', ''))
                        st.session_state['custom_revenue'] = float(extracted.get('baseline_revenue', 100000.0))
                        st.session_state['custom_cogs'] = float(extracted.get('baseline_cogs', 40000.0))
                        st.session_state['custom_share'] = float(extracted.get('raw_material_share', 0.40))
                        st.session_state['custom_fixed'] = float(extracted.get('fixed_costs', 30000.0))
                        st.session_state['custom_risk_prob'] = float(extracted.get('raw_material_risk_prob', 0.30))
                        st.session_state['custom_risk_increase'] = float(extracted.get('raw_material_risk_increase', 0.50))
                        st.session_state['custom_low_contraction'] = float(extracted.get('low_season_contraction_mean', 0.15))
                        st.session_state['custom_low_std'] = float(extracted.get('low_season_contraction_std', 0.05))
                        st.session_state['custom_high_spike'] = float(extracted.get('high_season_spike_mean', 0.40))
                        st.session_state['custom_high_std'] = float(extracted.get('high_season_spike_std', 0.10))
                        
                        # Clear old crew report if parameters changed from a newly loaded document
                        if 'crew_report' in st.session_state:
                            del st.session_state['crew_report']
                        if 'sim_data' in st.session_state:
                            del st.session_state['sim_data']
                            
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
        H -.-> I[Exportar a Google Colab]
    ```
    """)

with tab2:
    st.markdown("### 🎲 Simulación Monte Carlo y Métricas de Riesgo Financiero")
    st.write(
        "La simulación de Monte Carlo ejecuta miles de iteraciones aleatorias basándose en las distribuciones de probabilidad "
        "definidas para las variables de entrada. Esto nos permite observar el abanico completo de resultados posibles "
        "y calcular métricas de riesgo fundamentales para proteger la caja."
    )
    
    if st.button("🎲 Ejecutar Simulación Monte Carlo", type="primary", width="stretch"):
        # Clear old crew report on a new simulation run
        if 'crew_report' in st.session_state:
            del st.session_state['crew_report']
            
        with st.spinner("Ejecutando iteraciones financieras estocásticas..."):
            df_sims, metrics, sensitivity = run_monte_carlo_generic(**sim_args)
            
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
        var_color = "positive" if metrics['var_95'] >= 0 else "negative"
        cvar_color = "positive" if metrics['cvar_95'] >= 0 else "negative"
        loss_color = "negative" if metrics['prob_loss'] > 5 else "positive"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-box">
                <div class="metric-box-title">Beneficio Neto Promedio del Período</div>
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
            st.plotly_chart(fig_hist, width="stretch")
            
        with col_chart2:
            st.markdown("#### Análisis de Sensibilidad (Tornado)")
            
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
            st.plotly_chart(fig_sens, width="stretch")
            st.caption("Un valor cercano a -1.0 indica que al subir esa variable, el beneficio baja drásticamente. Un valor cercano a +1.0 indica que el beneficio sube al incrementarse esa variable.")
            
        # Summary dataframe
        with st.expander("🔍 Ver tabla resumida de datos simulados (Muestra de 10 ejecuciones)"):
            format_dict = {
                'Profit': '${:,.2f} MXN',
                'Revenue': '${:,.2f} MXN',
                'COGS': '${:,.2f} MXN'
            }
            for col in df_sims.columns:
                if col in ['Raw_Material_Factor']:
                    format_dict[col] = '{:.2f}x'
                elif col in ['Low_Season_Demand', 'High_Season_Demand']:
                    format_dict[col] = '{:.2%}'
                    
            st.dataframe(df_sims.head(10).style.format(format_dict, na_rep="-"))
            
    else:
        st.warning("Haz clic en 'Ejecutar Simulación Monte Carlo' para generar y visualizar los resultados financieros.")

with tab3:
    st.markdown("### 🤖 Consulta Estratégica Multiagente (CrewAI)")
    st.write(
        "Este módulo utiliza a tus agentes consultores (`Analista de Datos` y `Estratega Comercial`) "
        "para examinar los números de la simulación de Monte Carlo e idear el plan de mitigación personalizado."
    )
    
    if not has_crewai:
        st.error("⚠️ El módulo de Agentes de CrewAI no está disponible en este entorno.")
        st.markdown(
            "Esto ocurre generalmente si la biblioteca `crewai` no pudo ser instalada debido a limitaciones "
            "de compilación de dependencias (como `chromadb`) en el servidor de Streamlit Cloud o si "
            "la versión de Python seleccionada en la nube es inferior a 3.10.\n\n"
            "**Cómo solucionarlo en Streamlit Cloud:**\n"
            "1. Abre la configuración de tu aplicación en el dashboard de Streamlit Community Cloud (clic en *Manage app* y luego en los tres puntos de configuración).\n"
            "2. Cambia la versión de Python a **3.10** o **3.11** (las versiones recomendadas para CrewAI).\n"
            "3. En tu archivo `requirements.txt`, puedes intentar remover cualquier especificación de versión conflictiva o forzar una reinstalación limpia del proyecto en la nube.\n"
            "4. Mientras tanto, puedes usar la pestaña de **Simulación Monte Carlo** para ejecutar el análisis cuantitativo y la pestaña de **Google Colab** para descargar el cuaderno interactivo y ejecutar la parte de agentes en Colab de forma totalmente gratuita."
        )
        with st.expander("🔍 Ver el detalle del error de importación"):
            st.exception(crewai_import_error)
    elif 'sim_data' not in st.session_state:
        st.warning("Primero debes correr la simulación en la pestaña anterior para poder analizarla con los agentes.")
    else:
        active_key = openai_key if api_provider == "OpenAI" else api_key
        provider_name = "OpenAI" if api_provider == "OpenAI" else "Gemini"
        
        if not active_key:
            st.error(f"⚠️ Falta la API Key de {provider_name}. Por favor, ingrésala en la barra lateral para poder ejecutar los agentes.")
        else:
            if st.button("🤖 Iniciar Ejecución del Equipo de Agentes", type="primary", width="stretch"):
                sd = st.session_state['sim_data']
                
                st.markdown("#### 📺 Consola de Razonamiento de Agentes (Live Log)")
                log_placeholder = st.empty()
                log_placeholder.code("Inicializando el entorno de agentes CrewAI...")
                
                stdout_redirect = StreamlitRedirect(log_placeholder)
                old_stdout = sys.stdout
                sys.stdout = stdout_redirect
                
                try:
                    report_result = run_strategic_crew(
                        api_key=api_key,
                        openai_key=openai_key,
                        scenario_name=sd['scenario_title'],
                        scenario_description=sd['scenario_description'],
                        metrics=sd['metrics'],
                        sensitivity=sd['sensitivity'],
                        model_name=model_option
                    )
                    
                    sys.stdout = old_stdout
                    
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
                
                report_content = str(st.session_state['crew_report'])
                st.markdown(report_content)
                
                st.download_button(
                    label="💾 Descargar Reporte en Markdown",
                    data=report_content,
                    file_name=f"reporte_estrategico_{scenario_title.lower().replace(' ', '_')}.md",
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
    
    if not colab_params:
        st.info("Configura un escenario en la barra lateral para habilitar la descarga del cuaderno de Colab.")
    else:
        notebook_json = generate_colab_notebook(scenario_type, colab_params)
        
        st.download_button(
            label="📥 Descargar Cuaderno Jupyter (.ipynb) Personalizado",
            data=notebook_json,
            file_name="analisis_riesgo_personalizado_colab.ipynb",
            mime="application/x-ipynb+json",
            width="stretch",
            type="primary"
        )
        
        st.success("¡Cuaderno Jupyter listo para descargar! Incluye tus configuraciones actuales.")
        
        with st.expander("📄 Ver estructura interna del Notebook generado (JSON)"):
            st.code(notebook_json[:1500] + "\n\n... [Contenido truncado para visualización] ...")
