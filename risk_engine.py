import numpy as np
import pandas as pd

def run_monte_carlo_salsa(
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
    num_simulations=10000,
    seed=42
):
    """
    Runs a Monte Carlo simulation for the Cancun Salsa Business over a 4-month period:
    - 3 months of low season (September, October, November)
    - 1 month of high season (December)
    """
    np.random.seed(seed)
    
    # Pre-allocate arrays to store inputs and outputs
    profits = np.zeros(num_simulations)
    revenues = np.zeros(num_simulations)
    cogs_totals = np.zeros(num_simulations)
    
    # Store random variables for sensitivity analysis
    chile_factors = np.zeros(num_simulations)
    low_season_demands = np.zeros(num_simulations)
    high_season_demands = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        # 1. Chile Jalapeño cost factor (drawn once for the season)
        is_chile_crisis = np.random.rand() < chile_risk_prob
        chile_factor = 1.0 + chile_risk_increase if is_chile_crisis else 1.0
        chile_factors[i] = chile_factor
        
        # Effective cost factor for COGS based on Chile Jalapeño share
        # cogs_factor = (1 - share) * 1.0 + share * chile_factor
        effective_cogs_multiplier = (1.0 - chile_share) + (chile_share * chile_factor)
        
        # 2. Low season (September, October, November) - 3 months
        # Draw contraction factor (e.g. mean 20%, std 5%)
        # Note: contraction is expressed as a positive percentage (e.g. 0.20), so demand factor is (1 - contraction)
        low_season_contraction = np.random.normal(low_season_contraction_mean, low_season_contraction_std)
        # Clip contraction to stay within logical bounds [0, 1]
        low_season_contraction = np.clip(low_season_contraction, 0.0, 1.0)
        low_season_demands[i] = 1.0 - low_season_contraction
        
        # 3. High season (December) - 1 month
        # Draw spike factor (e.g. mean 60%, std 10%)
        high_season_spike = np.random.normal(high_season_spike_mean, high_season_spike_std)
        # Clip spike to be positive
        high_season_spike = max(0.0, high_season_spike)
        high_season_demands[i] = 1.0 + high_season_spike
        
        # Calculate financials
        # September-November (3 months)
        rev_low = baseline_revenue * (1.0 - low_season_contraction) * 3
        cogs_low = baseline_cogs * effective_cogs_multiplier * (1.0 - low_season_contraction) * 3
        fixed_low = fixed_costs * 3
        
        # December (1 month)
        rev_high = baseline_revenue * (1.0 + high_season_spike)
        cogs_high = baseline_cogs * effective_cogs_multiplier * (1.0 + high_season_spike)
        fixed_high = fixed_costs
        
        # Totals
        total_rev = rev_low + rev_high
        total_cogs = cogs_low + cogs_high
        total_fixed = fixed_low + fixed_high
        total_profit = total_rev - total_cogs - total_fixed
        
        revenues[i] = total_rev
        cogs_totals[i] = total_cogs
        profits[i] = total_profit

    # Calculate key metrics
    mean_profit = np.mean(profits)
    median_profit = np.median(profits)
    min_profit = np.min(profits)
    max_profit = np.max(profits)
    
    # Value at Risk (VaR) 95% - The 5th percentile of profit distribution
    # This means there is a 5% chance the profit will be below this value
    var_95 = np.percentile(profits, 5)
    
    # Conditional Value at Risk (CVaR) 95% - Average of profits below the 95% VaR
    cvar_95 = np.mean(profits[profits <= var_95])
    
    # Probability of loss (profit < 0)
    prob_loss = np.mean(profits < 0.0) * 100
    
    # Expected margins
    mean_rev = np.mean(revenues)
    mean_cogs = np.mean(cogs_totals)
    expected_margin = (mean_profit / mean_rev) * 100 if mean_rev > 0 else 0
    
    # Create DataFrame for analysis
    df_sims = pd.DataFrame({
        'Profit': profits,
        'Revenue': revenues,
        'COGS': cogs_totals,
        'Chile_Factor': chile_factors,
        'Low_Season_Demand': low_season_demands,
        'High_Season_Demand': high_season_demands
    })
    
    # Sensitivity (correlation with profit)
    sensitivity = {
        'Chile Jalapeño Price Factor': df_sims['Chile_Factor'].corr(df_sims['Profit']),
        'Low Season Demand Factor': df_sims['Low_Season_Demand'].corr(df_sims['Profit']),
        'High Season Demand Factor': df_sims['High_Season_Demand'].corr(df_sims['Profit'])
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


def run_monte_carlo_coffee(
    baseline_revenue=150000.0,
    baseline_cogs=70000.0,
    fixed_costs=40000.0,
    exchange_rate_volatility=0.08, # standard deviation of exchange rate (USD/MXN) change
    coffee_drop_prob=0.40,
    coffee_drop_pct=0.25,
    logistics_hike_prob=0.15,
    logistics_hike_pct=0.30,
    num_simulations=10000,
    seed=42
):
    """
    Runs a Monte Carlo simulation for the Veracruz Coffee Exporter over a 4-month period:
    - Income is USD-denominated (subject to Exchange Rate)
    - Coffee price changes directly impact revenue
    - Logistics risks impact COGS
    """
    np.random.seed(seed)
    
    profits = np.zeros(num_simulations)
    revenues = np.zeros(num_simulations)
    cogs_totals = np.zeros(num_simulations)
    
    fx_rates = np.zeros(num_simulations)
    coffee_prices = np.zeros(num_simulations)
    logistics_costs = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        # 1. Exchange rate factor (drawn as normal centered around 1.0)
        # Represents USD/MXN strength: if MXN strengthens, we get less MXN for USD revenue
        fx_factor = np.random.normal(1.0, exchange_rate_volatility)
        fx_rates[i] = fx_factor
        
        # 2. Coffee international price drop (40% probability of 25% drop)
        has_coffee_drop = np.random.rand() < coffee_drop_prob
        coffee_factor = 1.0 - coffee_drop_pct if has_coffee_drop else 1.0
        coffee_prices[i] = coffee_factor
        
        # 3. Logistics cost increase (15% probability of 30% increase in COGS)
        has_logistics_hike = np.random.rand() < logistics_hike_prob
        logistics_factor = 1.0 + logistics_hike_pct if has_logistics_hike else 1.0
        logistics_costs[i] = logistics_factor
        
        # Calculate monthly financials over 4 months
        # Revenue is affected by exchange rate and coffee price (since we export coffee)
        rev_monthly = baseline_revenue * fx_factor * coffee_factor
        cogs_monthly = baseline_cogs * logistics_factor
        
        total_rev = rev_monthly * 4
        total_cogs = cogs_monthly * 4
        total_fixed = fixed_costs * 4
        total_profit = total_rev - total_cogs - total_fixed
        
        revenues[i] = total_rev
        cogs_totals[i] = total_cogs
        profits[i] = total_profit
        
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
        'expected_margin': float((np.mean(profits) / np.mean(revenues)) * 100 if np.mean(revenues) > 0 else 0)
    }
    
    return df_sims, metrics, sensitivity


def run_monte_carlo_brewery(
    baseline_revenue=80000.0,
    baseline_cogs=30000.0,
    fixed_costs=25000.0,
    water_drought_prob=0.20,
    water_cost_multiplier=3.0,
    barley_hike_min=0.10,
    barley_hike_max=0.30,
    tourism_mean=0.10,
    tourism_std=0.15,
    num_simulations=10000,
    seed=42
):
    """
    Runs a Monte Carlo simulation for the Craft Brewery in Baja California over a 4-month period:
    - Water scarcity spikes water cost in COGS
    - Barley cost fluctuates (triangular distribution)
    - Tourism seasonal variation affects sales (normal distribution)
    """
    np.random.seed(seed)
    
    profits = np.zeros(num_simulations)
    revenues = np.zeros(num_simulations)
    cogs_totals = np.zeros(num_simulations)
    
    water_factors = np.zeros(num_simulations)
    barley_factors = np.zeros(num_simulations)
    tourism_factors = np.zeros(num_simulations)
    
    # Assume water represents 15% of COGS, barley represents 25% of COGS, other costs 60%
    water_share = 0.15
    barley_share = 0.25
    
    for i in range(num_simulations):
        # 1. Water cost surge (drought)
        has_drought = np.random.rand() < water_drought_prob
        water_factor = water_cost_multiplier if has_drought else 1.0
        water_factors[i] = water_factor
        
        # 2. Barley price increase (triangular distribution)
        barley_hike = np.random.triangular(barley_hike_min, (barley_hike_min + barley_hike_max)/2, barley_hike_max)
        barley_factor = 1.0 + barley_hike
        barley_factors[i] = barley_factor
        
        # 3. Tourism variation
        tourism_shift = np.random.normal(tourism_mean, tourism_std)
        demand_factor = 1.0 + tourism_shift
        tourism_factors[i] = demand_factor
        
        # Compute effective COGS factor:
        effective_cogs_multiplier = (1.0 - water_share - barley_share) + (water_share * water_factor) + (barley_share * barley_factor)
        
        # Over 4 months
        rev_total = baseline_revenue * demand_factor * 4
        cogs_total = baseline_cogs * effective_cogs_multiplier * demand_factor * 4
        fixed_total = fixed_costs * 4
        
        total_profit = rev_total - cogs_total - fixed_total
        
        revenues[i] = rev_total
        cogs_totals[i] = cogs_total
        profits[i] = total_profit
        
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
        'expected_margin': float((np.mean(profits) / np.mean(revenues)) * 100 if np.mean(revenues) > 0 else 0)
    }
    
    return df_sims, metrics, sensitivity
