import numpy as np
from typing import Dict, Any

def analyze_bias(results: Dict[str, Dict[str, Any]], task_type: str = '3class') -> None:
    """
    Analyze bias in model predictions across different target groups
    
    Args:
        results: Dictionary with model results
        task_type: Type of classification task ('binary' or '3class')
    """
    print(f"\n=== Bias Analysis for {task_type} Classification ===\n")
    
    for model_name, result in results[task_type].items():
        print(f"\nModel: {model_name}")
        
        if not result['target_metrics']:
            print("Target metrics not available for bias analysis")
            continue
        
        # Calculate standard deviation of performance across target groups
        target_values = list(result['target_metrics'].values())
        target_std = np.std(target_values)
        target_mean = np.mean(target_values)
        target_min = np.min(target_values)
        target_max = np.max(target_values)
        
        print(f"Performance across target groups:")
        print(f"Mean accuracy: {target_mean:.4f}")
        print(f"Standard deviation: {target_std:.4f}")
        print(f"Min accuracy: {target_min:.4f}")
        print(f"Max accuracy: {target_max:.4f}")
        print(f"Max-Min difference: {target_max - target_min:.4f}")
        
        # Identify target groups with worst and best performance
        worst_target = min(result['target_metrics'].items(), key=lambda x: x[1])
        best_target = max(result['target_metrics'].items(), key=lambda x: x[1])
        
        print(f"Target group with worst performance: {worst_target[0]} (accuracy: {worst_target[1]:.4f})")
        print(f"Target group with best performance: {best_target[0]} (accuracy: {best_target[1]:.4f})")
        
        # Calculate bias score (normalized standard deviation)
        bias_score = target_std / target_mean if target_mean > 0 else float('inf')
        print(f"Bias score (normalized std): {bias_score:.4f}")

def plot_target_group_performance(results: Dict[str, Dict[str, Any]], task_type: str = '3class') -> None:
    """
    Plot performance by target group
    
    Args:
        results: Dictionary with model results
        task_type: Type of classification task ('binary' or '3class')
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    target_groups = set()
    for model_result in results[task_type].values():
        if model_result['target_metrics']:
            for target in model_result['target_metrics'].keys():
                target_groups.add(target)
    
    target_groups = sorted(target_groups)
    
    # Create data for plotting
    data = []
    for model_name, model_result in results[task_type].items():
        model_name_short = model_name.split('/')[-1]  # Get just the model name without path
        
        if model_result['target_metrics']:
            model_data = [model_result['target_metrics'].get(target, float('nan')) for target in target_groups]
            data.append((model_name_short, model_data))
    
    # Create plot
    x = np.arange(len(target_groups))
    width = 0.8 / len(data)  # Width of the bars
    
    for i, (model_name, model_data) in enumerate(data):
        offset = width * i - width * len(data) / 2 + width / 2
        plt.bar(x + offset, model_data, width, label=model_name)
    
    plt.xlabel('Target Group')
    plt.ylabel('Accuracy')
    plt.title(f'Model Performance by Target Group ({task_type} Classification)')
    plt.xticks(x, target_groups, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.show()