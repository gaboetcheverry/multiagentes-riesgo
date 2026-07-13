import os
import sys
from crewai import Agent, Task, Crew, Process

def run_strategic_crew(api_key, scenario_name, scenario_description, metrics, sensitivity, model_name="gemini/gemini-1.5-flash", openai_key=None):
    """
    Sets up and runs the CrewAI agents to perform strategic risk analysis.
    Ingests the quantitative outputs from the Monte Carlo simulation.
    """
    # Set the keys in the environment if provided
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        
    # Check if we are running an OpenAI model or Gemini model
    is_openai = (model_name.startswith("gpt-") or model_name.startswith("openai/") or model_name.startswith("o1-") or model_name.startswith("o3-"))
    
    if is_openai:
        if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
            raise ValueError("OpenAI API Key is not set. Please provide it in the sidebar or set the OPENAI_API_KEY environment variable.")
    else:
        if "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
            raise ValueError("Gemini API Key is not set. Please provide it in the sidebar or set the GEMINI_API_KEY environment variable.")
        
    # Configure LLM model
    llm_model = model_name
    
    # Format sensitivity results into a readable string
    sens_lines = []
    for var, corr in sensitivity.items():
        impact = "Positivo" if corr > 0 else "Negativo"
        strength = "Fuerte" if abs(corr) > 0.7 else ("Moderado" if abs(corr) > 0.3 else "Débil")
        sens_lines.append(f"- **{var}**: Correlación de {corr:.2f} ({impact} {strength})")
    sens_str = "\n".join(sens_lines)

    # 1. Agent definition
    analista_datos = Agent(
        role='Analista Senior de Datos y Simulación de Riesgo',
        goal='Interpretar escenarios estadísticos e identificar las variables críticas de incertidumbre que afectan al negocio.',
        backstory=(
            'Eres un experto en análisis cuantitativo, finanzas corporativas y modelos predictivos. '
            'Te especializas en desglosar simulaciones numéricas complejas (como Monte Carlo, VaR y CVaR) '
            'para identificar los cuellos de botella y vulnerabilidades financieras más graves de una empresa. '
            'Traduces los números en explicaciones lógicas sobre dónde se encuentra el mayor riesgo.'
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm_model
    )

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

    # 2. Dynamic Task definition using simulation metrics
    description_analisis = f"""
    Realiza un análisis profundo del escenario: **{scenario_name}**
    
    **Descripción del Contexto**:
    {scenario_description}
    
    **Resultados Cuantitativos de la Simulación de Monte Carlo (10,000 iteraciones)**:
    - Ingresos Promedio Totales: ${metrics['mean_revenue']:,.2f} MXN
    - Costo de Ventas (COGS) Promedio: ${metrics['mean_cogs']:,.2f} MXN
    - Margen de Ganancia Neto Promedio: {metrics['expected_margin']:.2f}%
    - Ganancia Neta Promedio del Período: ${metrics['mean_profit']:,.2f} MXN
    - Valor en Riesgo (VaR al 95%): ${metrics['var_95']:,.2f} MXN (Representa el beneficio neto mínimo esperado con un 95% de confianza; si es negativo, representa pérdidas).
    - Valor en Riesgo Condicional (CVaR al 95%): ${metrics['cvar_95']:,.2f} MXN (Representa la media del peor 5% de los escenarios).
    - Probabilidad de Pérdida Neta (Beneficio < 0): {metrics['prob_loss']:.2f}%
    
    **Sensibilidad de Variables (Impacto en la Ganancia)**:
    {sens_str}
    
    Analiza la combinación de estos factores cuantitativos, identifica los puntos de quiebre financiero y operativo, 
    y detalla cómo se interconectan los riesgos (por ejemplo, el impacto combinado de un aumento de costos con una caída de la demanda).
    """

    tarea_analisis = Task(
        description=description_analisis,
        expected_output="Un informe estructurado en español con los 3 riesgos cuantitativos más importantes y su impacto estimado en el margen de ganancia del negocio, explicando el significado del VaR y CVaR para la toma de decisiones.",
        agent=analista_datos
    )

    description_estrategia = f"""
    Utiliza el informe de riesgos generado por el Analista de Datos para diseñar un plan de mitigación integral 
    para la empresa en el escenario de **{scenario_name}**.
    
    El plan debe ser altamente estratégico y operativo, considerando el contexto del negocio, y debe contener:
    1. Estrategias de cobertura, diversificación de proveedores o inventario anticipado para las materias primas críticas.
    2. Tácticas comerciales orientadas al mercado objetivo del escenario para sostener el flujo de caja e ingresos en periodos de contracción.
    3. Plan de preparación logística y de capacidad de producción/operación para reaccionar ante picos de demanda o desabasto.
    """

    tarea_estrategia = Task(
        description=description_estrategia,
        expected_output="Un plan de acción ejecutivo en español dividido en 3 horizontes de tiempo (Corto plazo: 1-3 meses, Mediano plazo: 3-6 meses y Largo plazo: más de 6 meses) con soluciones tácticas detalladas y aplicables al mercado geográfico correspondiente.",
        agent=estratega_negocios
    )

    # 3. Crew instantiation
    equipo_consultoria = Crew(
        agents=[analista_datos, estratega_negocios],
        tasks=[tarea_analisis, tarea_estrategia],
        process=Process.sequential,
        verbose=True
    )

    # Execute the crew sequential process
    resultado_final = equipo_consultoria.kickoff()
    
    return resultado_final
