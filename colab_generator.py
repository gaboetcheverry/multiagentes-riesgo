import json

def generate_colab_notebook(scenario_type, params):
    """
    Generates a Jupyter Notebook (.ipynb) as a JSON-formatted string.
    The notebook contains:
      1. Dependency installation
      2. Secure API key configuration
      3. The Monte Carlo risk engine code (self-contained)
      4. Plotly charting code
      5. CrewAI agent configurations and execution
    """
    
    # 1. Title cell
    title_md = f"""# Análisis de Riesgo y Decisiones bajo Incertidumbre: {params.get('scenario_title', 'Caso de Estudio')}
Este cuaderno interactivo fue generado automáticamente y está listo para ejecutarse en **Google Colab**. 

### Metodología:
1. **Simulación Cuantitativa de Monte Carlo (10,000 iteraciones)**: Evalúa el impacto de la volatilidad y riesgos concurrentes en los márgenes y utilidades del negocio, calculando métricas de riesgo financiero clave:
   * **Value at Risk (VaR 95%)**: La pérdida máxima esperada con un 95% de nivel de confianza.
   * **Conditional Value at Risk (CVaR 95%)**: El impacto promedio en el peor 5% de los escenarios.
   * **Análisis de Sensibilidad**: Correlación entre variables de incertidumbre y ganancias.
2. **Planificación Estratégica Multiagente (CrewAI + Gemini)**: Un equipo de agentes de Inteligencia Artificial (Analista de Datos y Estratega de Negocios) interpreta los resultados y diseña planes de acción tácticos y de mitigación.

---
### Ejecución paso a paso:
1. Ejecuta la celda de **Instalación de Dependencias**.
2. Ejecuta la celda de **Configuración de API Key** para ingresar tu clave de Gemini.
3. Ejecuta el **Motor de Simulación** y los **Gráficos Interactivos**.
4. Inicia la **Consulta Multiagente** para obtener tu reporte estratégico.
"""

    # 2. Install dependencies cell
    install_code = """# Instalación de librerías necesarias en Google Colab
!pip install -q crewai crewai-tools plotly numpy pandas
print("¡Dependencias instaladas con éxito!")"""

    # 3. Secure API key configuration
    api_key_code = """import os
import getpass

if "GEMINI_API_KEY" not in os.environ:
    print("Por favor, ingresa tu clave API de Gemini:")
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API Key: ")
    
print("Clave de API configurada.")"""

    # 4. Scenario parameters cell
    params_code = f"# Parámetros del Escenario Seleccionado\n"
    for k, v in params.items():
        if isinstance(v, str):
            params_code += f"{k} = '{v}'\n"
        else:
            params_code += f"{k} = {v}\n"

    # 5. Core Simulation Engine code
    if scenario_type == "salsa":
        engine_code = """import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def run_simulation(
    baseline_revenue,
    baseline_cogs,
    chile_share,
    fixed_costs,
    chile_risk_prob,
    chile_risk_increase,
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
    
    chile_factors = np.zeros(num_simulations)
    low_season_demands = np.zeros(num_simulations)
    high_season_demands = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        is_chile_crisis = np.random.rand() < chile_risk_prob
        chile_factor = 1.0 + chile_risk_increase if is_chile_crisis else 1.0
        chile_factors[i] = chile_factor
        
        effective_cogs_multiplier = (1.0 - chile_share) + (chile_share * chile_factor)
        
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
        'Chile_Factor': chile_factors,
        'Low_Season_Demand': low_season_demands,
        'High_Season_Demand': high_season_demands
    })
    
    sensitivity = {
        'Chile Jalapeño Price Factor': df_sims['Chile_Factor'].corr(df_sims['Profit']),
        'Low Season Demand Factor': df_sims['Low_Season_Demand'].corr(df_sims['Profit']),
        'High Season Demand Factor': df_sims['High_Season_Demand'].corr(df_sims['Profit'])
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
        'expected_margin': float((np.mean(profits) / np.mean(revenues)) * 100)
    }
    
    return df_sims, metrics, sensitivity
"""
    elif scenario_type == "coffee":
        engine_code = """import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def run_simulation(
    baseline_revenue,
    baseline_cogs,
    fixed_costs,
    exchange_rate_volatility,
    coffee_drop_prob,
    coffee_drop_pct,
    logistics_hike_prob,
    logistics_hike_pct,
    num_simulations=10000,
    seed=42
):
    np.random.seed(seed)
    
    profits = np.zeros(num_simulations)
    revenues = np.zeros(num_simulations)
    cogs_totals = np.zeros(num_simulations)
    
    fx_rates = np.zeros(num_simulations)
    coffee_prices = np.zeros(num_simulations)
    logistics_costs = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        fx_factor = np.random.normal(1.0, exchange_rate_volatility)
        fx_rates[i] = fx_factor
        
        has_coffee_drop = np.random.rand() < coffee_drop_prob
        coffee_factor = 1.0 - coffee_drop_pct if has_coffee_drop else 1.0
        coffee_prices[i] = coffee_factor
        
        has_logistics_hike = np.random.rand() < logistics_hike_prob
        logistics_factor = 1.0 + logistics_hike_pct if has_logistics_hike else 1.0
        logistics_costs[i] = logistics_factor
        
        # 4 meses
        rev_monthly = baseline_revenue * fx_factor * coffee_factor
        cogs_monthly = baseline_cogs * logistics_factor
        
        total_rev = rev_monthly * 4
        total_cogs = cogs_monthly * 4
        total_fixed = fixed_costs * 4
        
        revenues[i] = total_rev
        cogs_totals[i] = total_cogs
        profits[i] = total_rev - total_cogs - total_fixed
        
    df_sims = pd.DataFrame({
        'Profit': profits,
        'Revenue': revenues,
        'COGS': cogs_totals,
        'FX_Factor': fx_rates,
        'Coffee_Price_Factor': coffee_prices,
        'Logistics_Factor': logistics_costs
    })
    
    sensitivity = {
        'Exchange Rate Factor (USD/MXN)': df_sims['FX_Factor'].corr(df_sims['Profit']),
        'International Coffee Price Factor': df_sims['Coffee_Price_Factor'].corr(df_sims['Profit']),
        'Logistics Cost Factor': df_sims['Logistics_Factor'].corr(df_sims['Profit'])
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
        'expected_margin': float((np.mean(profits) / np.mean(revenues)) * 100)
    }
    
    return df_sims, metrics, sensitivity
"""
    else: # brewery
        engine_code = """import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def run_simulation(
    baseline_revenue,
    baseline_cogs,
    fixed_costs,
    water_drought_prob,
    water_cost_multiplier,
    barley_hike_min,
    barley_hike_max,
    tourism_mean,
    tourism_std,
    num_simulations=10000,
    seed=42
):
    np.random.seed(seed)
    
    profits = np.zeros(num_simulations)
    revenues = np.zeros(num_simulations)
    cogs_totals = np.zeros(num_simulations)
    
    water_factors = np.zeros(num_simulations)
    barley_factors = np.zeros(num_simulations)
    tourism_factors = np.zeros(num_simulations)
    
    water_share = 0.15
    barley_share = 0.25
    
    for i in range(num_simulations):
        has_drought = np.random.rand() < water_drought_prob
        water_factor = water_cost_multiplier if has_drought else 1.0
        water_factors[i] = water_factor
        
        barley_hike = np.random.triangular(barley_hike_min, (barley_hike_min + barley_hike_max)/2, barley_hike_max)
        barley_factor = 1.0 + barley_hike
        barley_factors[i] = barley_factor
        
        tourism_shift = np.random.normal(tourism_mean, tourism_std)
        demand_factor = 1.0 + tourism_shift
        tourism_factors[i] = demand_factor
        
        effective_cogs_multiplier = (1.0 - water_share - barley_share) + (water_share * water_factor) + (barley_share * barley_factor)
        
        rev_total = baseline_revenue * demand_factor * 4
        cogs_total = baseline_cogs * effective_cogs_multiplier * demand_factor * 4
        fixed_total = fixed_costs * 4
        
        revenues[i] = rev_total
        cogs_totals[i] = cogs_total
        profits[i] = rev_total - cogs_total - fixed_total
        
    df_sims = pd.DataFrame({
        'Profit': profits,
        'Revenue': revenues,
        'COGS': cogs_totals,
        'Water_Factor': water_factors,
        'Barley_Factor': barley_factors,
        'Tourism_Factor': tourism_factors
    })
    
    sensitivity = {
        'Water Cost Factor (Drought)': df_sims['Water_Factor'].corr(df_sims['Profit']),
        'Barley Cost Factor': df_sims['Barley_Factor'].corr(df_sims['Profit']),
        'Tourism Demand Factor': df_sims['Tourism_Factor'].corr(df_sims['Profit'])
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
        'expected_margin': float((np.mean(profits) / np.mean(revenues)) * 100)
    }
    
    return df_sims, metrics, sensitivity
"""

    is_custom_scenario = (scenario_type == "salsa" and "Salsa" not in params.get('scenario_title', ''))

    # 6. Run simulation call and plotting
    if scenario_type == "salsa":
        run_sim_call = """# Ejecutar simulación
df_sims, metrics, sensitivity = run_simulation(
    baseline_revenue=baseline_revenue,
    baseline_cogs=baseline_cogs,
    chile_share=chile_share,
    fixed_costs=fixed_costs,
    chile_risk_prob=chile_risk_prob,
    chile_risk_increase=chile_risk_increase,
    low_season_contraction_mean=low_season_contraction_mean,
    low_season_contraction_std=low_season_contraction_std,
    high_season_spike_mean=high_season_spike_mean,
    high_season_spike_std=high_season_spike_std,
    num_simulations=10000
)"""
        if is_custom_scenario:
            run_sim_call += """

# Renombrar variables para escenario personalizado (cualquier negocio)
sensitivity = {
    'Costo de Materia Prima Crítica': sensitivity['Chile Jalapeño Price Factor'],
    'Contracción de Demanda': sensitivity['Low Season Demand Factor'],
    'Expansión de Demanda': sensitivity['High Season Demand Factor']
}
df_sims = df_sims.rename(columns={
    'Chile_Factor': 'Factor_Costo_Materia_Prima',
    'Low_Season_Demand': 'Factor_Demanda_Baja',
    'High_Season_Demand': 'Factor_Demanda_Alta'
})"""
    elif scenario_type == "coffee":
        run_sim_call = """# Ejecutar simulación
df_sims, metrics, sensitivity = run_simulation(
    baseline_revenue=baseline_revenue,
    baseline_cogs=baseline_cogs,
    fixed_costs=fixed_costs,
    exchange_rate_volatility=exchange_rate_volatility,
    coffee_drop_prob=coffee_drop_prob,
    coffee_drop_pct=coffee_drop_pct,
    logistics_hike_prob=logistics_hike_prob,
    logistics_hike_pct=logistics_hike_pct,
    num_simulations=10000
)"""
    else: # brewery
        run_sim_call = """# Ejecutar simulación
df_sims, metrics, sensitivity = run_simulation(
    baseline_revenue=baseline_revenue,
    baseline_cogs=baseline_cogs,
    fixed_costs=fixed_costs,
    water_drought_prob=water_drought_prob,
    water_cost_multiplier=water_cost_multiplier,
    barley_hike_min=barley_hike_min,
    barley_hike_max=barley_hike_max,
    tourism_mean=tourism_mean,
    tourism_std=tourism_std,
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

    # 7. CrewAI agent code cell
    crewai_code = """import os
from crewai import Agent, Task, Crew, Process

# Asegurar que el LLM esté configurado
llm_model = "gemini/gemini-1.5-flash"

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
    llm=llm_model
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
    llm=llm_model
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

    # Assemble the Jupyter Notebook JSON
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
                    "Esta celda carga las funciones de simulación de Monte Carlo diseñadas específicamente para el tipo de escenario."
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
