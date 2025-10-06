"""
SSUAP: Small-Sample Uncertainty Assessment Protocol
Implementation of sample-size adjusted confidence thresholds for multi-criteria
environmental assessment under institutional sampling constraints (n<20).

Reference: Fatima Zahra Kherazi (2025). SSUAP: A Transferable Protocol for Small-Sample
Uncertainty Assessment in Multi-Criteria Environmental Evaluation. Environmental
Modelling & Software [in peer review].

License: MIT
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

def calculate_thresholds(n: int) -> Tuple[float, float]:
    """
    Calculate sample-size adjusted confidence thresholds.
    
    Args:
        n: Sample size (number of spatial units, e.g., basins)
    
    Returns:
        Tuple of (cv_high, cv_moderate) thresholds
    
    Example:
        >>> cv_high, cv_mod = calculate_thresholds(n=5)
        >>> print(f"HIGH: {cv_high:.3f}, MODERATE: {cv_mod:.3f}")
        HIGH: 0.172, MODERATE: 0.345
    """
    cv_high = 0.15 + 0.05 / np.sqrt(n)
    cv_moderate = 0.30 + 0.10 / np.sqrt(n)
    return cv_high, cv_moderate


def calculate_weight_bounds(n: int) -> Tuple[float, float]:
    """
    Calculate realistic weight uncertainty bounds based on sample size.
    
    Args:
        n: Sample size
    
    Returns:
        Tuple of (lower_multiplier, upper_multiplier)
        E.g., (0.75, 1.25) means ±25% bounds
    
    Example:
        >>> lower, upper = calculate_weight_bounds(n=5)
        >>> print(f"Bounds: ±{(upper-1)*100:.1f}%")
        Bounds: ±22.4%
    """
    bound_percentage = 0.20 + 0.05 / np.sqrt(n)
    lower_mult = 1 - bound_percentage
    upper_mult = 1 + bound_percentage
    return lower_mult, upper_mult


def generate_ensemble_scenarios(base_weights: Dict[str, float], 
                                n: int) -> Dict[str, Dict[str, float]]:
    """
    Generate six-scenario ensemble spanning decision-relevant weight space.
    
    Args:
        base_weights: Dictionary of component weights (e.g., {'climate': 0.4, 'land': 0.3})
        n: Sample size for bound calculation
    
    Returns:
        Dictionary of six weight scenarios: Empirical, Equal, two emphasis scenarios,
        and two conservative bound scenarios
    
    Example:
        >>> weights = {'climate': 0.35, 'land_use': 0.40, 'erosion': 0.25}
        >>> scenarios = generate_ensemble_scenarios(weights, n=5)
        >>> print(len(scenarios))
        6
    """
    components = list(base_weights.keys())
    n_components = len(components)
    lower_mult, upper_mult = calculate_weight_bounds(n)
    
    scenarios = {}
    
    # Scenario 1: Empirical (data-driven baseline)
    scenarios['Empirical'] = base_weights.copy()
    
    # Scenario 2: Equal (neutral perspective)
    equal_weight = 1.0 / n_components
    scenarios['Equal'] = {comp: equal_weight for comp in components}
    
    # Scenario 3 & 4: Emphasis scenarios (amplify first two components by 30%)
    for i, comp in enumerate(components[:2]):  # First two components
        emphasis_weights = base_weights.copy()
        emphasis_weights[comp] *= 1.3
        total = sum(emphasis_weights.values())
        scenarios[f'{comp.title()}_Emphasis'] = {k: v/total for k, v in emphasis_weights.items()}
    
    # Scenario 5: Conservative lower bound
    conservative_lower = {k: v * lower_mult for k, v in base_weights.items()}
    total = sum(conservative_lower.values())
    scenarios['Conservative_Lower'] = {k: v/total for k, v in conservative_lower.items()}
    
    # Scenario 6: Conservative upper bound
    conservative_upper = {k: v * upper_mult for k, v in base_weights.items()}
    total = sum(conservative_upper.values())
    scenarios['Conservative_Upper'] = {k: v/total for k, v in conservative_upper.items()}
    
    return scenarios


def calculate_composite_scores(component_scores: pd.DataFrame,
                               weights: Dict[str, float]) -> pd.Series:
    """
    Calculate weighted composite vulnerability scores.
    
    Args:
        component_scores: DataFrame with rows=basins, columns=components (normalized 0-1)
        weights: Dictionary of component weights
    
    Returns:
        Series of composite scores per basin
    """
    composite = pd.Series(0.0, index=component_scores.index)
    for component, weight in weights.items():
        if component in component_scores.columns:
            composite += weight * component_scores[component]
    return composite


def classify_vulnerability(composite_scores: pd.Series, 
                          method: str = 'median_split') -> pd.Series:
    """
    Binary classification into HIGH vs MODERATE priority.
    
    Args:
        composite_scores: Series of composite vulnerability scores
        method: 'median_split' (default) or 'mean_split'
    
    Returns:
        Series of classifications ('HIGH' or 'MODERATE')
    """
    if method == 'median_split':
        threshold = composite_scores.median()
    else:
        threshold = composite_scores.mean()
    
    classifications = pd.Series('MODERATE', index=composite_scores.index)
    classifications[composite_scores >= threshold] = 'HIGH'
    return classifications


def assess_ensemble_agreement(ensemble_classifications: pd.DataFrame) -> pd.DataFrame:
    """
    Assess classification confidence based on ensemble agreement.
    
    Args:
        ensemble_classifications: DataFrame with rows=basins, columns=scenarios,
                                 values='HIGH' or 'MODERATE'
    
    Returns:
        DataFrame with columns: Consensus_Class, Agreement_Rate, N_HIGH, N_MODERATE
    """
    results = []
    for basin in ensemble_classifications.index:
        basin_classes = ensemble_classifications.loc[basin]
        n_high = (basin_classes == 'HIGH').sum()
        n_total = len(basin_classes)
        agreement_rate = max(n_high, n_total - n_high) / n_total
        consensus = 'HIGH' if n_high > n_total/2 else 'MODERATE'
        
        results.append({
            'Basin': basin,
            'Consensus_Class': consensus,
            'Agreement_Rate': agreement_rate,
            'N_HIGH': n_high,
            'N_MODERATE': n_total - n_high
        })
    
    return pd.DataFrame(results).set_index('Basin')


def calculate_cv(ensemble_scores: pd.DataFrame) -> pd.Series:
    """
    Calculate coefficient of variation across ensemble scenarios.
    
    Args:
        ensemble_scores: DataFrame with rows=basins, columns=scenarios,
                        values=composite scores
    
    Returns:
        Series of CV values per basin
    """
    means = ensemble_scores.mean(axis=1)
    stds = ensemble_scores.std(axis=1)
    cv = stds / means
    return cv


def assign_confidence_levels(cv_values: pd.Series, n: int) -> pd.Series:
    """
    Assign HIGH/MODERATE/LOW confidence based on CV and sample size.
    
    Args:
        cv_values: Series of coefficient of variation per basin
        n: Sample size
    
    Returns:
        Series of confidence levels ('HIGH', 'MODERATE', or 'LOW')
    """
    cv_high, cv_moderate = calculate_thresholds(n)
    
    confidence = pd.Series('LOW', index=cv_values.index)
    confidence[cv_values <= cv_moderate] = 'MODERATE'
    confidence[cv_values <= cv_high] = 'HIGH'
    
    return confidence


def run_ssuap_analysis(component_scores: pd.DataFrame,
                       base_weights: Dict[str, float]) -> Dict:
    """
    Complete SSUAP analysis workflow.
    
    Args:
        component_scores: DataFrame with normalized component scores (0-1)
        base_weights: Dictionary of base weights for components
    
    Returns:
        Dictionary containing:
            - 'thresholds': (cv_high, cv_moderate)
            - 'scenarios': Dictionary of weight scenarios
            - 'composite_scores': DataFrame of composite scores per scenario
            - 'classifications': DataFrame of classifications per scenario
            - 'ensemble_agreement': DataFrame with consensus and agreement rates
            - 'cv_values': Series of CV per basin
            - 'confidence_levels': Series of confidence levels per basin
    """
    n = len(component_scores)
    
    # Calculate thresholds
    thresholds = calculate_thresholds(n)
    
    # Generate scenarios
    scenarios = generate_ensemble_scenarios(base_weights, n)
    
    # Run analysis for each scenario
    composite_scores = pd.DataFrame()
    classifications = pd.DataFrame()
    
    for scenario_name, weights in scenarios.items():
        scores = calculate_composite_scores(component_scores, weights)
        composite_scores[scenario_name] = scores
        
        class_result = classify_vulnerability(scores)
        classifications[scenario_name] = class_result
    
    # Assess ensemble
    ensemble_agreement = assess_ensemble_agreement(classifications)
    cv_values = calculate_cv(composite_scores)
    confidence_levels = assign_confidence_levels(cv_values, n)
    
    return {
        'thresholds': thresholds,
        'scenarios': scenarios,
        'composite_scores': composite_scores,
        'classifications': classifications,
        'ensemble_agreement': ensemble_agreement,
        'cv_values': cv_values,
        'confidence_levels': confidence_levels
    }


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("SSUAP Example with Synthetic Data")
    print("="*60)
    
    # Create synthetic 5-basin dataset
    basins = ['Basin_A', 'Basin_B', 'Basin_C', 'Basin_D', 'Basin_E']
    component_data = {
        'climate_vulnerability': [0.65, 0.45, 0.80, 0.55, 0.70],
        'land_use_change': [0.55, 0.70, 0.40, 0.85, 0.60],
        'erosion_risk': [0.70, 0.60, 0.75, 0.50, 0.65]
    }
    component_scores = pd.DataFrame(component_data, index=basins)
    
    # Define base weights
    base_weights = {
        'climate_vulnerability': 0.35,
        'land_use_change': 0.40,
        'erosion_risk': 0.25
    }
    
    # Run complete analysis
    results = run_ssuap_analysis(component_scores, base_weights)
    
    # Display results
    print(f"\nSample size: {len(basins)}")
    print(f"CV thresholds: HIGH={results['thresholds'][0]:.3f}, MODERATE={results['thresholds'][1]:.3f}")
    print(f"\nGenerated {len(results['scenarios'])} scenarios")
    
    print("\n" + "="*60)
    print("ENSEMBLE AGREEMENT")
    print("="*60)
    print(results['ensemble_agreement'])
    
    print("\n" + "="*60)
    print("CONFIDENCE ASSESSMENT")
    print("="*60)
    print(pd.DataFrame({
        'CV': results['cv_values'],
        'Confidence': results['confidence_levels']
    }))
