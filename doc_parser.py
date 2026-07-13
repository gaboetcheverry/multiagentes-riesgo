import io
import json
import pandas as pd
import fitz  # PyMuPDF
import docx  # python-docx
import google.generativeai as genai

def extract_text_from_pdf(file_bytes):
    """Extracts all text from a PDF file using PyMuPDF."""
    text = ""
    # Open PDF from memory stream
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text() + "\n"
    return text

def extract_text_from_docx(file_bytes):
    """Extracts all text from a Word (.docx) file using python-docx."""
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs_text = [p.text for p in doc.paragraphs]
    
    # Also extract text from tables
    table_text = []
    for table in doc.tables:
        for row in table.rows:
            row_cells = [cell.text for cell in row.cells]
            table_text.append(" | ".join(row_cells))
            
    return "\n".join(paragraphs_text + table_text)

def extract_text_from_excel_csv(file_bytes, filename):
    """Reads Excel/CSV files and converts the top contents to a text representation."""
    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
        
    # Get shape and basic details
    summary = f"Archivo Excel/CSV: {filename}\n"
    summary += f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas\n"
    summary += f"Columnas: {', '.join(df.columns.tolist())}\n"
    summary += "Primeras 50 filas de datos:\n"
    # Convert first 50 rows to markdown or CSV table format
    summary += df.head(50).to_csv(index=False)
    return summary

def parse_business_file(file_bytes, filename):
    """Dispatches parsing depending on file extension."""
    ext = filename.split('.')[-1].lower()
    
    if ext == 'pdf':
        return extract_text_from_pdf(file_bytes)
    elif ext in ['doc', 'docx']:
        return extract_text_from_docx(file_bytes)
    elif ext in ['csv', 'xlsx', 'xls']:
        return extract_text_from_excel_csv(file_bytes, filename)
    else:
        # Fallback to plain text reading
        try:
            return file_bytes.decode('utf-8')
        except Exception:
            raise ValueError(f"Extensión de archivo .{ext} no soportada.")

def extract_parameters_via_gemini(document_text, filename, api_key):
    """
    Calls the Gemini API to extract financial parameters from raw text.
    Ensures response conforms to a clean, flat JSON schema matching the simulation requirements.
    """
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    prompt = f"""
    Eres un analista financiero experto en gestión de riesgos y simulación de Monte Carlo. 
    Se te proporciona el contenido extraído de un archivo de negocio real ({filename}).
    Tu tarea es analizar la información y extraer los parámetros base y de incertidumbre para una simulación financiera de 4 meses.
    
    El modelo de simulación de 4 meses asume:
    - 3 meses de temporada baja (sujetos a contracción de demanda)
    - 1 mes de temporada alta (sujeto a un pico o repunte de demanda)
    - Riesgo de aumento en el costo de un insumo o materia prima crítica (que representa un % del costo de ventas o COGS).
    
    Por favor, lee el documento y extrae u optimiza los siguientes parámetros. Si el documento no especifica alguno de los valores financieros, utiliza tu conocimiento sectorial y el contexto de la empresa para ESTIMAR montos y desviaciones lógicas (especifica la justificación de estas estimaciones en la descripción).
    
    Debes devolver un objeto JSON plano que contenga EXACTAMENTE los siguientes campos:
    
    1. "scenario_title": Un nombre corto y descriptivo del negocio/proyecto (máx 50 caracteres).
    2. "scenario_description": Un resumen ejecutivo (máx 300 caracteres) que describa de qué trata la empresa, qué insumo crítico es el volátil y el contexto del mercado geográfico o sectorial.
    3. "baseline_revenue": Ingresos mensuales promedio base del negocio en condiciones normales. Debe ser un número float.
    4. "baseline_cogs": Costo mensual de ventas base (COGS) promedio del negocio. Debe ser un número float.
    5. "chile_share": Proporción de la materia prima/insumo crítico dentro del costo de ventas total (ej. 0.35 si representa el 35% del COGS). Debe ser un float entre 0.05 y 0.90.
    6. "fixed_costs": Gastos fijos mensuales promedio (renta, nómina, etc.). Debe ser un número float.
    7. "chile_risk_prob": Probabilidad de que el precio del insumo crítico aumente bruscamente debido a factores de riesgo (clima, logística, aranceles) (ej. 0.30 para un 30% de probabilidad). Debe ser un float entre 0.0 y 1.0.
    8. "chile_risk_increase": Porcentaje estimado de aumento en el costo del insumo en caso de crisis (ej. 0.50 si sube un 50% de precio). Debe ser un float entre 0.0 y 2.0.
    9. "low_season_contraction_mean": Contracción promedio esperada de la demanda general en los 3 meses de temporada baja (ej. 0.20 para una reducción del 20% en ventas). Debe ser un float entre 0.0 y 1.0.
    10. "low_season_contraction_std": Volatilidad o desviación estándar de la contracción de la demanda en temporada baja. Valor sugerido entre 0.02 y 0.15 (por defecto 0.05).
    11. "high_season_spike_mean": Repunte o incremento promedio esperado de la demanda en el mes de temporada alta (ej. 0.60 para un incremento de 60%). Debe ser un float entre 0.0 y 2.0.
    12. "high_season_spike_std": Volatilidad o desviación estándar del incremento en temporada alta. Valor sugerido entre 0.05 y 0.25 (por defecto 0.10).
    
    Contenido del documento:
    ---
    {document_text}
    ---
    
    Devuelve ÚNICAMENTE el objeto JSON plano solicitado. No agregues formatos de código como ```json ... ```, solo la cadena JSON limpia.
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Request JSON response type natively
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    
    try:
        data = json.loads(response.text)
        return data
    except Exception as e:
        # Fallback to cleaning markdown blocks if any
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)
