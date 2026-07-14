import numpy as np
import pandas as pd

def run_monte_carlo_generic(
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
    num_simulations=10000,
    seed=42
):
    """
    Ejecuta una simulación de Monte Carlo genérica para un negocio sobre un período de 4 meses:
    - 3 meses de temporada baja (sujetos a contracción de demanda)
    - 1 mes de temporada alta (sujeto a un pico o repunte de demanda)
    - Riesgo de incremento en el costo de una materia prima o insumo crítico (que representa un % del COGS)
    """
    np.random.seed(seed)
    
    # Pre-asignar arreglos para guardar entradas y salidas
    profits = np.zeros(num_simulations)
    revenues = np.zeros(num_simulations)
    cogs_totals = np.zeros(num_simulations)
    
    # Guardar variables aleatorias para análisis de sensibilidad
    raw_material_factors = np.zeros(num_simulations)
    low_season_demands = np.zeros(num_simulations)
    high_season_demands = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        # 1. Factor de incremento de costo de materia prima crítica
        is_crisis = np.random.rand() < raw_material_risk_prob
        rm_factor = 1.0 + raw_material_risk_increase if is_crisis else 1.0
        raw_material_factors[i] = rm_factor
        
        # Multiplicador de COGS efectivo basado en la proporción del insumo
        effective_cogs_multiplier = (1.0 - raw_material_share) + (raw_material_share * rm_factor)
        
        # 2. Temporada baja (3 meses)
        low_season_contraction = np.random.normal(low_season_contraction_mean, low_season_contraction_std)
        # Limitar la contracción a rangos lógicos [0, 1]
        low_season_contraction = np.clip(low_season_contraction, 0.0, 1.0)
        low_season_demands[i] = 1.0 - low_season_contraction
        
        # 3. Temporada alta (1 mes)
        high_season_spike = np.random.normal(high_season_spike_mean, high_season_spike_std)
        # Limitar el pico a ser positivo
        high_season_spike = max(0.0, high_season_spike)
        high_season_demands[i] = 1.0 + high_season_spike
        
        # Calcular estados financieros del período de temporada baja (3 meses)
        rev_low = baseline_revenue * (1.0 - low_season_contraction) * 3
        cogs_low = baseline_cogs * effective_cogs_multiplier * (1.0 - low_season_contraction) * 3
        fixed_low = fixed_costs * 3
        
        # Calcular estados financieros de temporada alta (1 mes)
        rev_high = baseline_revenue * (1.0 + high_season_spike)
        cogs_high = baseline_cogs * effective_cogs_multiplier * (1.0 + high_season_spike)
        fixed_high = fixed_costs
        
        # Totales del período de 4 meses
        total_rev = rev_low + rev_high
        total_cogs = cogs_low + cogs_high
        total_fixed = fixed_low + fixed_high
        total_profit = total_rev - total_cogs - total_fixed
        
        revenues[i] = total_rev
        cogs_totals[i] = total_cogs
        profits[i] = total_profit

    # Métricas estadísticas clave
    mean_profit = np.mean(profits)
    median_profit = np.median(profits)
    min_profit = np.min(profits)
    max_profit = np.max(profits)
    
    # Value at Risk (VaR) 95% - El percentil 5 de la distribución de ganancias
    var_95 = np.percentile(profits, 5)
    
    # Conditional Value at Risk (CVaR) 95% - Promedio de las ganancias por debajo del 95% VaR
    cvar_95 = np.mean(profits[profits <= var_95])
    
    # Probabilidad de pérdida neta (ganancia < 0)
    prob_loss = np.mean(profits < 0.0) * 100
    
    # Márgenes esperados
    mean_rev = np.mean(revenues)
    mean_cogs = np.mean(cogs_totals)
    expected_margin = (mean_profit / mean_rev) * 100 if mean_rev > 0 else 0
    
    # Crear DataFrame para análisis gráfico
    df_sims = pd.DataFrame({
        'Profit': profits,
        'Revenue': revenues,
        'COGS': cogs_totals,
        'Raw_Material_Factor': raw_material_factors,
        'Low_Season_Demand': low_season_demands,
        'High_Season_Demand': high_season_demands
    })
    
    # Correlación de sensibilidad
    sensitivity = {
        'Factor de Costo de Insumo Crítico': df_sims['Raw_Material_Factor'].corr(df_sims['Profit']),
        'Factor de Demanda en Temporada Baja': df_sims['Low_Season_Demand'].corr(df_sims['Profit']),
        'Factor de Demanda en Temporada Alta': df_sims['High_Season_Demand'].corr(df_sims['Profit'])
    }
    
    metrics = {
        'mean_profit': float(mean_profit),
        'median_profit': float(median_profit),
        'min_profit': float(min_profit),
        'max_profit': float(max_profit),
        'var_95': float(var_95),
        'cvar_95': float(cvar_95),
        'prob_loss': float(prob_loss),
        'mean_revenue': float(mean_rev),
        'mean_cogs': float(mean_cogs),
        'expected_margin': float(expected_margin)
    }
    
    return df_sims, metrics, sensitivity
