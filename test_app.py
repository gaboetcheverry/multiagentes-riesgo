import unittest
import json
import pandas as pd
import numpy as np

from risk_engine import run_monte_carlo_salsa, run_monte_carlo_coffee, run_monte_carlo_brewery
from colab_generator import generate_colab_notebook

class TestRiskAppModules(unittest.TestCase):
    
    def test_run_monte_carlo_salsa(self):
        """Test the Salsa scenario Monte Carlo engine returns correct outputs and shapes."""
        df_sims, metrics, sensitivity = run_monte_carlo_salsa(
            baseline_revenue=100000.0,
            baseline_cogs=40000.0,
            chile_share=0.30,
            fixed_costs=30000.0,
            chile_risk_prob=0.35,
            chile_risk_increase=0.45,
            low_season_contraction_mean=0.20,
            low_season_contraction_std=0.05,
            high_season_spike_mean=0.60,
            high_season_spike_std=0.10,
            num_simulations=100, # small runs for speed
            seed=42
        )
        
        # Verify types
        self.assertIsInstance(df_sims, pd.DataFrame)
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(sensitivity, dict)
        
        # Verify sizes
        self.assertEqual(len(df_sims), 100)
        
        # Verify key metrics exist
        self.assertIn('mean_profit', metrics)
        self.assertIn('var_95', metrics)
        self.assertIn('cvar_95', metrics)
        self.assertIn('prob_loss', metrics)
        self.assertIn('expected_margin', metrics)
        
        # Verify sensitivity variables
        self.assertIn('Chile Jalapeño Price Factor', sensitivity)
        self.assertIn('Low Season Demand Factor', sensitivity)
        self.assertIn('High Season Demand Factor', sensitivity)

    def test_run_monte_carlo_coffee(self):
        """Test the Coffee exporter scenario simulation engine."""
        df_sims, metrics, sensitivity = run_monte_carlo_coffee(
            baseline_revenue=150000.0,
            baseline_cogs=70000.0,
            fixed_costs=40000.0,
            exchange_rate_volatility=0.08,
            coffee_drop_prob=0.40,
            coffee_drop_pct=0.25,
            logistics_hike_prob=0.15,
            logistics_hike_pct=0.30,
            num_simulations=50,
            seed=42
        )
        
        self.assertEqual(len(df_sims), 50)
        self.assertIn('Exchange Rate Factor (USD/MXN)', sensitivity)
        self.assertIn('var_95', metrics)

    def test_run_monte_carlo_brewery(self):
        """Test the Brewery scenario simulation engine."""
        df_sims, metrics, sensitivity = run_monte_carlo_brewery(
            baseline_revenue=80000.0,
            baseline_cogs=30000.0,
            fixed_costs=25000.0,
            water_drought_prob=0.20,
            water_cost_multiplier=3.0,
            barley_hike_min=0.10,
            barley_hike_max=0.30,
            tourism_mean=0.10,
            tourism_std=0.15,
            num_simulations=50,
            seed=42
        )
        
        self.assertEqual(len(df_sims), 50)
        self.assertIn('Water Cost Factor (Drought)', sensitivity)
        self.assertIn('cvar_95', metrics)

    def test_colab_generator(self):
        """Test that the Google Colab Notebook exporter creates a valid ipynb JSON structure."""
        params = {
            'scenario_title': "Test Scenario",
            'scenario_description': "Test Description",
            'baseline_revenue': 100000.0,
            'baseline_cogs': 40000.0,
            'chile_share': 0.30,
            'fixed_costs': 30000.0,
            'chile_risk_prob': 0.35,
            'chile_risk_increase': 0.45,
            'low_season_contraction_mean': 0.20,
            'low_season_contraction_std': 0.05,
            'high_season_spike_mean': 0.60,
            'high_season_spike_std': 0.10
        }
        
        notebook_str = generate_colab_notebook("salsa", params)
        self.assertIsInstance(notebook_str, str)
        
        # Should be valid JSON
        notebook_json = json.loads(notebook_str)
        self.assertIn('cells', notebook_json)
        self.assertIn('metadata', notebook_json)
        self.assertEqual(notebook_json['nbformat'], 4)
        
        # Verify it has cells of type code and markdown
        cells = notebook_json['cells']
        self.assertGreater(len(cells), 0)
        
        cell_types = [c['cell_type'] for c in cells]
        self.assertIn('code', cell_types)
        self.assertIn('markdown', cell_types)

    def test_csv_parser(self):
        """Test that doc_parser can read a CSV from bytes."""
        from doc_parser import parse_business_file
        
        # Create a simple CSV bytes buffer
        csv_data = "Mes,Ingresos,Costos\nEne,120000,50000\nFeb,130000,52000"
        file_bytes = csv_data.encode('utf-8')
        
        result_text = parse_business_file(file_bytes, "test.csv")
        self.assertIn("Archivo Excel/CSV: test.csv", result_text)
        self.assertIn("Mes,Ingresos,Costos", result_text)
        self.assertIn("Ene,120000,50000", result_text)

    def test_text_fallback_parser(self):
        """Test text fallback parsing."""
        from doc_parser import parse_business_file
        text = "This is a simple business description."
        file_bytes = text.encode('utf-8')
        
        result = parse_business_file(file_bytes, "info.txt")
        self.assertEqual(result, text)

if __name__ == '__main__':
    unittest.main()
