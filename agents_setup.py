import os
import sys
import logging

# Disable telemetry to prevent environment encoding issues on Streamlit Cloud
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Force UTF-8 encoding for subprocess safety (CrewAI internal logging uses emojis)
os.environ["PYTHONIOENCODING"] = "utf-8"

from crewai import Agent, Task, Crew, Process

try:
    from crewai import LLM
    has_llm_class = True
except ImportError:
    has_llm_class = False

def clean_to_ascii(text):
    import unicodedata
    if not text:
        return ""
    # Normalize unicode to separate characters from their accents (NFD)
    nfd_form = unicodedata.normalize('NFD', text)
    only_ascii = "".join([c for c in nfd_form if not unicodedata.combining(c)])
    
    # Manual replacement for Spanish characters
    replacements = {
        'ñ': 'n', 'Ñ': 'N',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ü': 'u', 'Ü': 'U'
    }
    for orig, rep in replacements.items():
        only_ascii = only_ascii.replace(orig, rep)
        
    return only_ascii.encode('ascii', errors='ignore').decode('ascii')


class _SafeStream:
    """Wraps a stream to silently handle encoding errors from CrewAI's emoji-heavy logging."""
    def __init__(self, stream):
        self._stream = stream
    
    def write(self, text):
        try:
            self._stream.write(text)
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Strip non-ASCII characters and retry
            safe_text = text.encode('ascii', errors='replace').decode('ascii')
            try:
                self._stream.write(safe_text)
            except Exception:
                pass
        except Exception:
            pass
    
    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
    
    def __getattr__(self, name):
        return getattr(self._stream, name)


def run_strategic_crew(openai_key, scenario_name, scenario_description, metrics, sensitivity, model_name="gpt-4o-mini"):
    """
    Configura y ejecuta los agentes de CrewAI para realizar el análisis de riesgo estratégico.
    Ingesta los resultados cuantitativos de la simulación de Monte Carlo.
    """
    # Limpiar espacios en blanco de las claves y asegurar ASCII
    if isinstance(openai_key, str):
        openai_key = clean_to_ascii(openai_key).strip()

    # Establecer la clave en el entorno y limpiar residuos de Gemini
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    else:
        raise ValueError("La API Key de OpenAI no está configurada. Por favor, ingrésala en la barra lateral.")
        
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

    # Instanciar LLM de manera explícita para evitar fallbacks incorrectos
    if has_llm_class:
        llm_inst = LLM(model=model_name, api_key=openai_key)
    else:
        llm_inst = model_name
    
    # Limpiar inputs a ASCII para evitar errores de codificación en entornos con locale restringido (como Streamlit Cloud)
    scenario_name = clean_to_ascii(scenario_name)
    scenario_description = clean_to_ascii(scenario_description)
    
    # Formatear resultados de sensibilidad en texto legible y limpio de acentos
    sens_lines = []
    for var, corr in sensitivity.items():
        var_clean = clean_to_ascii(var)
        impact = "Positivo" if corr > 0 else "Negativo"
        strength = "Fuerte" if abs(corr) > 0.7 else ("Moderado" if abs(corr) > 0.3 else "Debil")
        sens_lines.append(f"- **{var_clean}**: Correlacion de {corr:.2f} ({impact} {strength})")
    sens_str = "\n".join(sens_lines)

    # 1. Definición de Agentes
    analista_datos = Agent(
        role=clean_to_ascii('Analista Senior de Datos y Simulación de Riesgo'),
        goal=clean_to_ascii('Interpretar escenarios estadísticos e identificar las variables críticas de incertidumbre que afectan al negocio.'),
        backstory=clean_to_ascii(
            'Eres un experto en análisis cuantitativo, finanzas corporativas y modelos predictivos. '
            'Te especializas en desglosar simulaciones numéricas complejas (como Monte Carlo, VaR y CVaR) '
            'para identificar los cuellos de botella y vulnerabilidades financieras más graves de una empresa. '
            'Traduces los números en explicaciones lógicas sobre dónde se encuentra el mayor riesgo.'
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm_inst
    )

    estratega_negocios = Agent(
        role=clean_to_ascii('Consultor de Estrategia Comercial y Mitigación de Riesgos'),
        goal=clean_to_ascii('Traducir análisis numéricos y de riesgo en planes de contingencia operativos y comerciales completamente accionables.'),
        backstory=clean_to_ascii(
            'Eres un asesor empresarial experimentado que destaca por diseñar planes tácticos y estratégicos '
            'para proteger el flujo de caja, la continuidad operativa y la cadena de suministro en entornos '
            'de alta incertidumbre. Te adaptas al contexto geográfico e industrial del negocio.'
        ),
        verbose=True,
        allow_delegation=True,
        llm=llm_inst
    )

    # 2. Definición de tareas dinámicas con métricas de simulación
    description_analisis = f"""
    Realiza un analisis profundo del escenario: **{scenario_name}**
    
    **Descripcion del Contexto**:
    {scenario_description}
    
    **Resultados Cuantitativos de la Simulacion de Monte Carlo (10,000 iteraciones)**:
    - Ingresos Promedio Totales: ${metrics['mean_revenue']:,.2f} MXN
    - Costo de Ventas (COGS) Promedio: ${metrics['mean_cogs']:,.2f} MXN
    - Margen de Ganancia Neto Promedio: {metrics['expected_margin']:.2f}%
    - Ganancia Neta Promedio del Periodo: ${metrics['mean_profit']:,.2f} MXN
    - Valor en Riesgo (VaR al 95%): ${metrics['var_95']:,.2f} MXN (Representa el beneficio neto minimo esperado con un 95% de confianza; si es negativo, representa perdidas).
    - Valor en Riesgo Condicional (CVaR al 95%): ${metrics['cvar_95']:,.2f} MXN (Representa la media del peor 5% de los escenarios).
    - Probabilidad de Perdida Neta (Beneficio < 0): {metrics['prob_loss']:.2f}%
    
    **Sensibilidad de Variables (Impacto en la Ganancia)**:
    {sens_str}
    
    Analiza la combinacion de estos factores cuantitativos, identifica los puntos de quiebre financiero y operativo, 
    y detalla como se interconectan los riesgos (por ejemplo, el impacto combinado de un aumento de costos con una caida de la demanda).
    """

    tarea_analisis = Task(
        description=clean_to_ascii(description_analisis),
        expected_output=clean_to_ascii("Un informe estructurado en espanol con los 3 riesgos cuantitativos mas importantes y su impacto estimado en el margen de ganancia del negocio, explicando el significado del VaR y CVaR para la toma de decisiones."),
        agent=analista_datos
    )

    description_estrategia = f"""
    Utiliza el informe de riesgos generado por el Analista de Datos para disenar un plan de mitigacion integral 
    para la empresa en el escenario de **{scenario_name}**.
    
    El plan debe ser altamente estrategico y operativo, considerando el contexto del negocio, y debe contener:
    1. Estrategias de cobertura, diversificacion de proveedores o inventario anticipado para las materias primas criticas.
    2. Tacticas comerciales orientadas al mercado objetivo del escenario para sostener el flujo de caja e ingresos en periodos de contraccion.
    3. Plan de preparacion logistica y de capacidad de produccion/operacion para reaccionar ante picos de demanda o desabasto.
    """

    tarea_estrategia = Task(
        description=clean_to_ascii(description_estrategia),
        expected_output=clean_to_ascii("Un plan de accion ejecutivo en espanol dividido en 3 horizontes de tiempo (Corto plazo: 1-3 meses, Mediano plazo: 3-6 meses y Largo plazo: mas de 6 meses) con soluciones tacticas detalladas y aplicables al mercado geografico correspondiente."),
        agent=estratega_negocios
    )

    # Configurar el embedder de OpenAI explícitamente (usa model_name, no model)
    embedder_config = {
        "provider": "openai",
        "config": {
            "model_name": "text-embedding-3-small",
            "api_key": openai_key
        }
    }

    # 3. Instanciar la tripulación (Crew)
    equipo_consultoria = Crew(
        agents=[analista_datos, estratega_negocios],
        tasks=[tarea_analisis, tarea_estrategia],
        process=Process.sequential,
        verbose=True,
        embedder=embedder_config
    )

    # 4. Protect stdout/stderr from CrewAI's emoji-rich event bus crashing on Windows charmap
    #    CrewAI internally prints 🚀 🤖 etc. which fail on 'charmap' codec
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Suppress noisy CrewAI event bus loggers that emit emojis
    for logger_name in ['crewai.utilities.events', 'crewai.flow.runtime', 'crewai']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)

    # Wrap streams only if they don't support UTF-8 natively
    stdout_needs_wrap = True
    stderr_needs_wrap = True
    try:
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower():
            stdout_needs_wrap = False
    except Exception:
        pass
    try:
        if hasattr(sys.stderr, 'encoding') and sys.stderr.encoding and 'utf' in sys.stderr.encoding.lower():
            stderr_needs_wrap = False
    except Exception:
        pass
    
    if stdout_needs_wrap:
        sys.stdout = _SafeStream(original_stdout)
    if stderr_needs_wrap:
        sys.stderr = _SafeStream(original_stderr)

    try:
        # Ejecución
        resultado_final = equipo_consultoria.kickoff()
    finally:
        # Always restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    return resultado_final
