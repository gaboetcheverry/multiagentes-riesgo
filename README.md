# Multi-Agent Risk Analysis & Decision Support System (Monte Carlo & CrewAI)

Este proyecto es una plataforma interactiva premium diseñada para realizar **análisis cuantitativo de riesgo** y **planificación estratégica calificada** utilizando simulaciones de Monte Carlo y un equipo de agentes de Inteligencia Artificial (CrewAI) impulsados por Google Gemini.

## 🌟 Características clave

1. **Simulador de Monte Carlo**: Ejecuta hasta 15,000 simulaciones de flujo de caja para evaluar el impacto financiero combinado de múltiples variables bajo riesgo e incertidumbre.
2. **Métricas de Riesgo Financiero**: Calcula de forma automática:
   * **Value at Risk (VaR 95%)**: El beneficio mínimo esperado con un 95% de confianza.
   * **Conditional Value at Risk (CVaR 95%)**: El promedio del peor 5% de los escenarios.
   * **Probabilidad de pérdida**: Porcentaje de escenarios que resultan en saldo negativo.
3. **Análisis de Sensibilidad de Tornado**: Gráficos interactivos de correlación para identificar qué variable tiene el impacto más crítico sobre las utilidades.
4. **Equipo Multiagente (CrewAI)**:
   * **Analista de Datos**: Interpreta los resultados numéricos de Monte Carlo y detecta puntos de quiebre operativos.
   * **Estratega de Negocios**: Diseña un plan de acción ejecutivo dividido en tres horizontes de tiempo (Corto, Mediano y Largo Plazo).
5. **Integración con Google Colab**: Descarga directa de un cuaderno Jupyter (`.ipynb`) pre-configurado con los mismos parámetros para correr todo el flujo en la nube sin instalaciones locales.

---

## 🛠️ Instalación y Configuración Local

### Prerrequisitos
* Python 3.10 o superior (Probado en Python 3.13)
* Clave de API de Gemini (puedes obtenerla gratis en [Google AI Studio](https://aistudio.google.com/))

### Pasos

1. **Clonar o descargar el proyecto** en tu máquina local.
2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ejecutar la aplicación Streamlit**:
   ```bash
   streamlit run app.py
   ```
4. **Abrir en el navegador**:
   Por defecto, Streamlit se abrirá en `http://localhost:8501`.

---

## ☁️ Integración con Google Colab

Si prefieres no instalar librerías pesadas en tu computadora o deseas compartir el análisis con tu equipo en un entorno en la nube, el sistema permite la exportación directa:

1. Ve a la pestaña **Integración Google Colab** en el panel de control web.
2. Haz clic en **Descargar Cuaderno Jupyter (.ipynb) Personalizado**.
3. Abre [Google Colab](https://colab.research.google.com/).
4. Sube el archivo `.ipynb` descargado.
5. Ejecuta las celdas secuencialmente (el cuaderno incluye guías paso a paso e instalación automática de dependencias).

---

## 📂 Estructura del Código

* [app.py](file:///c:/Users/User/OneDrive/Desktop/Multiagentes%20riesgo/app.py): Interfaz web interactiva construida con Streamlit.
* [risk_engine.py](file:///c:/Users/User/OneDrive/Desktop/Multiagentes%20riesgo/risk_engine.py): Motor matemático para las simulaciones de Monte Carlo y correlaciones de sensibilidad.
* [agents_setup.py](file:///c:/Users/User/OneDrive/Desktop/Multiagentes%20riesgo/agents_setup.py): Definición de los agentes CrewAI, objetivos, backstories y orquestación de tareas.
* [colab_generator.py](file:///c:/Users/User/OneDrive/Desktop/Multiagentes%20riesgo/colab_generator.py): Compilador dinámico de cuadernos Jupyter para Google Colab.
* [test_app.py](file:///c:/Users/User/OneDrive/Desktop/Multiagentes%20riesgo/test_app.py): Suite de pruebas unitarias automatizadas para los módulos de simulación y exportación.
