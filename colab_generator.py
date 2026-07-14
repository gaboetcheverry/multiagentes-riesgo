import json

def generate_colab_notebook(scenario_type, params):
    """
    Genera un archivo Jupyter Notebook (.ipynb) como string formateado en JSON.
    El cuaderno contiene:
      1. Instalación de dependencias.
      2. Configuración segura de la clave API de Gemini.
      3. El motor de simulación de Monte Carlo genérico.
      4. Código para graficar con Plotly.
      5. Configuración y ejecución de los agentes de CrewAI.
    """
    
    # 1. Celda de título
    title_md = f"""# Análisis de Riesgo y Decisiones bajo Incertidumbre: {params.get('scenario_title', 'Caso de Estudio Personalizado')}
Este cuaderno interactivo fue generado automáticamente y está listo para ejecutarse en **Google Colab**. 

### Metodología:
1. **Simulación Cuantitativa de Monte Carlo (10,000 iteraciones)**: Evalúa el impacto de la volatilidad y riesgos concurrentes en los márgenes y utilidades del negocio, calculando métricas de riesgo financiero clave:
   * **Value at Risk (VaR 95%)**: El beneficio neto mínimo esperado con un 95% de nivel de confianza.
   * **Conditional Value at Risk (CVaR 95%)**: El beneficio promedio en el peor 5% de los escenarios.
   * **Análisis de Sensibilidad (Tornado)**: Correlación entre variables de incertidumbre y ganancias.
2. **Planificación Estratégica Multiagente (CrewAI + Gemini)**: Un equipo de agentes de Inteligencia Artificial (Analista de Datos y Estratega de Negocios) interpreta los resultados y diseña planes de acción tácticos y de mitigación.

---
### Ejecución paso a paso:
1. Ejecuta la celda de **Instalación de Dependencias**.
2. Ejecuta la celda de **Configuración de API Key** para ingresar tu clave de Gemini.
3. Ejecuta el **Motor de Simulación** y los **Gráficos Interactivos**.
4. Inicia la **Consulta Multiagente** para obtener tu reporte estratégico.
"""

    # 2. Celda para instalar dependencias
    install_code = """# Instalación de librerías necesarias en Google Colab
!pip install -q crewai crewai-tools plotly numpy pandas
print("¡Dependencias instaladas con éxito!")"""

    # 3. Configuración segura de la clave API
    api_key_code = """import os
import getpass

if "GEMINI_API_KEY" not in os.environ:
    print("Por favor, ingresa tu clave API de Gemini:")
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API Key: ")
    
print("Clave de API configurada.")"""

    # 4. Celda de parámetros del escenario
    params_code = f"# Parámetros del Escenario Personalizado\n"
    for k, v in params.items():
        if isinstance(v, str):
            params_code += f"{k} = '{v}'\n"
        else:
            params_code += f"{k} = {v}\n"

    # 5. Código del motor de simulación genérico
    engine_code = """import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def run_simulation(
    baseline_revenue,
    baseline_cogs,
    raw_material_share,
    fixed_costs,
    raw_material_risk_prob,
    raw_material_risk_increase,
    low_season_contraction_mean,
    low_season_contraction_std,
    high_season_spike_mean,
    high_season_spike_std,
    num_simulations=10000,
    seed=42
):
    np.random.seed(seed)
    
    profits = np.zeros(num_simulations)
    revenues = np.zeros(num_simulations)
    cogs_totals = np.zeros(num_simulations)
    
    raw_material_factors = np.zeros(num_simulations)
    low_season_demands = np.zeros(num_simulations)
    high_season_demands = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        is_crisis = np.random.rand() < raw_material_risk_prob
        rm_factor = 1.0 + raw_material_risk_increase if is_crisis else 1.0
        raw_material_factors[i] = rm_factor
        
        effective_cogs_multiplier = (1.0 - raw_material_share) + (raw_material_share * rm_factor)
        
        low_season_contraction = np.random.normal(low_season_contraction_mean, low_season_contraction_std)
        low_season_contraction = np.clip(low_season_contraction, 0.0, 1.0)
        low_season_demands[i] = 1.0 - low_season_contraction
        
        high_season_spike = np.random.normal(high_season_spike_mean, high_season_spike_std)
        high_season_spike = max(0.0, high_season_spike)
        high_season_demands[i] = 1.0 + high_season_spike
        
        # 3 meses temporada baja + 1 mes temporada alta
        rev_low = baseline_revenue * (1.0 - low_season_contraction) * 3
        cogs_low = baseline_cogs * effective_cogs_multiplier * (1.0 - low_season_contraction) * 3
        fixed_low = fixed_costs * 3
        
        rev_high = baseline_revenue * (1.0 + high_season_spike)
        cogs_high = baseline_cogs * effective_cogs_multiplier * (1.0 + high_season_spike)
        fixed_high = fixed_costs
        
        total_rev = rev_low + rev_high
        total_cogs = cogs_low + cogs_high
        total_fixed = fixed_low + fixed_high
        
        revenues[i] = total_rev
        cogs_totals[i] = total_cogs
        profits[i] = total_rev - total_cogs - total_fixed
        
    df_sims = pd.DataFrame({
        'Profit': profits,
        'Revenue': revenues,
        'COGS': cogs_totals,
        'Raw_Material_Factor': raw_material_factors,
        'Low_Season_Demand': low_season_demands,
        'High_Season_Demand': high_season_demands
    })
    
    sensitivity = {
        'Factor de Costo de Insumo Crítico': df_sims['Raw_Material_Factor'].corr(df_sims['Profit']),
        'Factor de Demanda en Temporada Baja': df_sims['Low_Season_Demand'].corr(df_sims['Profit']),
        'Factor de Demanda en Temporada Alta': df_sims['High_Season_Demand'].corr(df_sims['Profit'])
    }
    
    metrics = {
        'mean_profit': float(np.mean(profits)),
        'median_profit': float(np.median(profits)),
        'min_profit': float(np.min(profits)),
        'max_profit': float(np.max(profits)),
        'var_95': float(np.percentile(profits, 5)),
        'cvar_95': float(np.mean(profits[profits <= np.percentile(profits, 5)])),
        'prob_loss': float(np.mean(profits < 0.0) * 100),
        'mean_revenue': float(np.mean(revenues)),
        'mean_cogs': float(np.mean(cogs_totals)),
        'expected_margin': float((np.mean(profits) / np.mean(revenues)) * 100 if np.mean(revenues) > 0 else 0)
    }
    
    return df_sims, metrics, sensitivity
"""

    # 6. Ejecución de simulación y gráficos
    run_sim_call = """# Ejecutar simulación
df_sims, metrics, sensitivity = run_simulation(
    baseline_revenue=baseline_revenue,
    baseline_cogs=baseline_cogs,
    raw_material_share=raw_material_share,
    fixed_costs=fixed_costs,
    raw_material_risk_prob=raw_material_risk_prob,
    raw_material_risk_increase=raw_material_risk_increase,
    low_season_contraction_mean=low_season_contraction_mean,
    low_season_contraction_std=low_season_contraction_std,
    high_season_spike_mean=high_season_spike_mean,
    high_season_spike_std=high_season_spike_std,
    num_simulations=10000
)"""

    plotting_code = run_sim_call + """

# Mostrar resultados clave
print("====== MÉTRICAS DE RIESGO DE MONTE CARLO ======")
print(f"Ganancia Neta Promedio: ${metrics['mean_profit']:,.2f} MXN")
print(f"Margen Esperado: {metrics['expected_margin']:.2f}%")
print(f"Probabilidad de Pérdida: {metrics['prob_loss']:.2f}%")
print(f"Value at Risk (VaR 95%): ${metrics['var_95']:,.2f} MXN")
print(f"Conditional Value at Risk (CVaR 95%): ${metrics['cvar_95']:,.2f} MXN")
print("===============================================")

# 1. Histograma de distribución de utilidades
fig_hist = px.histogram(
    df_sims, x='Profit', nbins=50,
    title='Distribución de la Ganancia Neta Acumulada (4 Meses)',
    labels={'Profit': 'Ganancia Neta (MXN)'},
    color_discrete_sequence=['#4F46E5']
)
fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Punto de Equilibrio (0)")
fig_hist.add_vline(x=metrics['var_95'], line_dash="solid", line_color="orange", annotation_text=f"VaR 95% (${metrics['var_95']:,.0f})")
fig_hist.show()

# 2. Gráfico de Tornado para Análisis de Sensibilidad
sens_df = pd.DataFrame(list(sensitivity.items()), columns=['Variable', 'Correlación'])
sens_df = sens_df.sort_values(by='Correlación', key=abs, ascending=True)

fig_sens = px.bar(
    sens_df, y='Variable', x='Correlación', orientation='h',
    title='Análisis de Sensibilidad (Correlación con la Ganancia Neta)',
    labels={'Correlación': 'Coeficiente de Correlación'},
    color='Correlación',
    color_continuous_scale=px.colors.diverging.RdBu
)
fig_sens.show()
"""

    # 7. Celda para CrewAI
    crewai_code = """import os
from crewai import Agent, Task, Crew, Process

try:
    from crewai import LLM
    has_llm = True
except ImportError:
    has_llm = False

# Asegurar que el LLM esté configurado con la clave de Gemini
llm_model = "gemini/gemini-1.5-flash"
if has_llm:
    llm_inst = LLM(model=llm_model, api_key=os.environ.get("GEMINI_API_KEY"))
else:
    llm_inst = llm_model

# Formatear sensibilidades
sens_lines = []
for var, corr in sensitivity.items():
    impact = "Positivo" if corr > 0 else "Negativo"
    strength = "Fuerte" if abs(corr) > 0.7 else ("Moderado" if abs(corr) > 0.3 else "Débil")
    sens_lines.append(f"- **{var}**: Correlación de {corr:.2f} ({impact} {strength})")
sens_str = "\\n".join(sens_lines)

# Definir Agente Analista
analista_datos = Agent(
    role='Analista Senior de Datos y Simulación de Riesgo',
    goal='Interpretar escenarios estadísticos e identificar las variables críticas de incertidumbre que afectan al negocio.',
    backstory=(
        'Eres un experto en análisis cuantitativo, finanzas corporativas y modelos predictivos. '
        'Te especializas en desglosar simulaciones numéricas complejas (como Monte Carlo, VaR y CVaR) '
        'para identificar los cuellos de botella y vulnerabilidades financieras más graves de una empresa.'
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm_inst
)

# Definir Agente Estratega
estratega_negocios = Agent(
    role='Consultor de Estrategia Comercial y Mitigación de Riesgos',
    goal='Traducir análisis numéricos y de riesgo en planes de contingencia operativos y comerciales completamente accionables.',
    backstory=(
        'Eres un asesor empresarial experimentado que destaca por diseñar planes tácticos y estratégicos '
        'para proteger el flujo de caja, la continuidad operativa y la cadena de suministro en entornos '
        'de alta incertidumbre. Te adaptas al contexto geográfico e industrial del negocio.'
    ),
    verbose=True,
    allow_delegation=True,
    llm=llm_inst
)

# Tarea de Análisis
tarea_analisis = Task(
    description=f\"\"\"
    Realiza un análisis profundo del escenario: **{scenario_title}**
    
    **Descripción del Contexto**:
    {scenario_description}
    
    **Resultados Cuantitativos de la Simulación de Monte Carlo**:
    - Ingresos Promedio Totales: ${metrics['mean_revenue']:,.2f} MXN
    - Costo de Ventas (COGS) Promedio: ${metrics['mean_cogs']:,.2f} MXN
    - Margen de Ganancia Neto Promedio: {metrics['expected_margin']:.2f}%
    - Ganancia Neta Promedio del Período: ${metrics['mean_profit']:,.2f} MXN
    - Valor en Riesgo (VaR al 95%): ${metrics['var_95']:,.2f} MXN
    - Valor en Riesgo Condicional (CVaR al 95%): ${metrics['cvar_95']:,.2f} MXN
    - Probabilidad de Pérdida Neta (Beneficio < 0): {metrics['prob_loss']:.2f}%
    
    **Sensibilidad de Variables**:
    {sens_str}
    
    Analiza la combinación de estos factores cuantitativos, identifica los puntos de quiebre financiero y operativo, 
    y detalla cómo se interconectan los riesgos.
    \"\"\",
    expected_output="Un informe estructurado con los 3 riesgos cuantitativos más importantes y su impacto estimado en el margen de ganancia del negocio, explicando el significado del VaR y CVaR para la toma de decisiones.",
    agent=analista_datos
)

# Tarea de Estrategia
tarea_estrategia = Task(
    description=f\"\"\"
    Utiliza el informe de riesgos generado por el Analista de Datos para diseñar un plan de mitigación integral 
    para la empresa en el escenario de **{scenario_title}**.
    
    El plan debe ser altamente estratégico y operativo, considerando el contexto del negocio, y debe contener:
    1. Estrategias de cobertura, diversificación de proveedores o inventario anticipado para las materias primas críticas.
    2. Tácticas comerciales orientadas al mercado objetivo del escenario para sostener el flujo de caja e ingresos en periodos de contracción.
    3. Plan de preparación logística y de capacidad de producción/operación para reaccionar ante picos de demanda o desabasto.
    \"\"\",
    expected_output="Un plan de acción ejecutivo en español dividido en 3 horizontes de tiempo (Corto plazo: 1-3 meses, Mediano plazo: 3-6 meses y Largo plazo: más de 6 meses) con soluciones tácticas detalladas y aplicables al mercado geográfico correspondiente.",
    agent=estratega_negocios
)

# Crear Crew
equipo_consultoria = Crew(
    agents=[analista_datos, estratega_negocios],
    tasks=[tarea_analisis, tarea_estrategia],
    process=Process.sequential,
    verbose=True
)

# Ejecutar agentes
print("Iniciando el proceso de los agentes CrewAI...")
resultado_final = equipo_consultoria.kickoff()
"""

    show_result_code = """print("\\n\\n#########################################")
print("## REPORTE ESTRATÉGICO MULTIAGENTE FINAL ##")
print("#########################################\\n")
print(resultado_final)"""

    # Ensamblar la estructura JSON del cuaderno Jupyter (.ipynb)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in title_md.split("\n")]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Paso 1: Instalar Dependencias\n",
                    "Ejecuta la siguiente celda para instalar `crewai` y `plotly` en la sesión de Google Colab."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in install_code.split("\n")]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Paso 2: Configurar Clave de API de Gemini\n",
                    "Introduce tu clave de API de Google Gemini para permitir que los agentes ejecuten su análisis."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in api_key_code.split("\n")]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Paso 3: Definir Parámetros del Escenario y Variables\n",
                    "Aquí se configuran los mismos parámetros financieros y probabilidades que definiste en la interfaz web."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in params_code.split("\n")]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Paso 4: Cargar Motor de Simulación\n",
                    "Esta celda carga las funciones de simulación de Monte Carlo genéricas."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in engine_code.split("\n")]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Paso 5: Correr Simulación y Ver Gráficos de Riesgo\n",
                    "Ejecuta las simulaciones y genera el histograma de distribución de beneficios y el análisis de sensibilidad de Tornado."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in plotting_code.split("\n")]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Paso 6: Configurar y Ejecutar el Equipo de Agentes (CrewAI)\n",
                    "Aquí se crean los agentes `Analista de Datos` y `Estratega de Negocios`, y se les alimentan los datos del análisis de riesgo cuantitativo de la celda anterior."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in crewai_code.split("\n")]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Paso 7: Mostrar Reporte Estratégico Final\n",
                    "Imprime la recomendación estratégica del estratega en formato legible."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in show_result_code.split("\n")]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    return json.dumps(notebook, indent=2, ensure_ascii=False)
