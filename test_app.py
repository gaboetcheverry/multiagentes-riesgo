import unittest
import json
import pandas as pd
import numpy as np

from risk_engine import run_monte_carlo_generic
from colab_generator import generate_colab_notebook

class TestRiskAppModules(unittest.TestCase):
    
    def test_run_monte_carlo_generic(self):
        """Prueba que el motor de simulación de Monte Carlo genérico retorne salidas y dimensiones correctas."""
        df_sims, metrics, sensitivity = run_monte_carlo_generic(
            baseline_revenue=100000.0,
            baseline_cogs=40000.0,
            raw_material_share=0.30,
            fixed_costs=30000.0,
            raw_material_risk_prob=0.35,
            raw_material_risk_increase=0.45,
            low_season_contraction_mean=0.20,
            low_season_contraction_std=0.05,
            high_season_spike_mean=0.60,
            high_season_spike_std=0.10,
            num_simulations=100, # corridas pequeñas para mayor velocidad
            seed=42
        )
        
        # Verificar tipos de datos
        self.assertIsInstance(df_sims, pd.DataFrame)
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(sensitivity, dict)
        
        # Verificar dimensiones
        self.assertEqual(len(df_sims), 100)
        
        # Verificar que existan las métricas clave
        self.assertIn('mean_profit', metrics)
        self.assertIn('var_95', metrics)
        self.assertIn('cvar_95', metrics)
        self.assertIn('prob_loss', metrics)
        self.assertIn('expected_margin', metrics)
        
        # Verificar las variables de sensibilidad genéricas
        self.assertIn('Factor de Costo de Insumo Crítico', sensitivity)
        self.assertIn('Factor de Demanda en Temporada Baja', sensitivity)
        self.assertIn('Factor de Demanda en Temporada Alta', sensitivity)

    def test_colab_generator(self):
        """Prueba que el exportador de Google Colab genere una estructura JSON (.ipynb) válida."""
        params = {
            'scenario_title': "Escenario de Prueba Genérico",
            'scenario_description': "Descripción de prueba para verificar variables genéricas.",
            'baseline_revenue': 100000.0,
            'baseline_cogs': 40000.0,
            'raw_material_share': 0.30,
            'fixed_costs': 30000.0,
            'raw_material_risk_prob': 0.35,
            'raw_material_risk_increase': 0.45,
            'low_season_contraction_mean': 0.20,
            'low_season_contraction_std': 0.05,
            'high_season_spike_mean': 0.60,
            'high_season_spike_std': 0.10
        }
        
        notebook_str = generate_colab_notebook("generic", params)
        self.assertIsInstance(notebook_str, str)
        
        # Debe ser un JSON válido
        notebook_json = json.loads(notebook_str)
        self.assertIn('cells', notebook_json)
        self.assertIn('metadata', notebook_json)
        self.assertEqual(notebook_json['nbformat'], 4)
        
        # Verificar que tenga celdas de tipo código y markdown
        cells = notebook_json['cells']
        self.assertGreater(len(cells), 0)
        
        cell_types = [c['cell_type'] for c in cells]
        self.assertIn('code', cell_types)
        self.assertIn('markdown', cell_types)

    def test_csv_parser(self):
        """Prueba que doc_parser lea correctamente un archivo CSV desde bytes."""
        from doc_parser import parse_business_file
        
        csv_data = "Mes,Ingresos,Costos\nEne,120000,50000\nFeb,130000,52000"
        file_bytes = csv_data.encode('utf-8')
        
        result_text = parse_business_file(file_bytes, "test.csv")
        self.assertIn("Archivo Excel/CSV: test.csv", result_text)
        self.assertIn("Mes,Ingresos,Costos", result_text)
        self.assertIn("Ene,120000,50000", result_text)

    def test_text_fallback_parser(self):
        """Prueba el parser fallback de texto."""
        from doc_parser import parse_business_file
        text = "Esta es una descripción comercial simple de prueba."
        file_bytes = text.encode('utf-8')
        
        result = parse_business_file(file_bytes, "info.txt")
        self.assertEqual(result, text)

if __name__ == '__main__':
    unittest.main()
