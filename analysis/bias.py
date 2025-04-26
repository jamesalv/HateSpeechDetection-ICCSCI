import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import pandas as pd

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
        
        # Print GMB metrics if they're available
        if 'bias_auc_metrics' in result:
            print("\nGMB Metrics:")
            for metric, value in result['bias_auc_metrics'].items():
                print(f"{metric}: {value:.4f}")

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

def calculate_gmb_metrics(
    predictions: np.ndarray, 
    probabilities: np.ndarray, 
    true_labels: np.ndarray, 
    test_df: pd.DataFrame, 
    target_groups: List[str]
) -> Dict[str, float]:
    """
    Calculate GMB (Generalized Mean of Bias) AUC metrics from model predictions
    
    Args:
        predictions: Model's class predictions
        probabilities: Model's probability outputs
        true_labels: Ground truth labels
        test_df: DataFrame with test data including target groups
        target_groups: List of target groups to evaluate
        
    Returns:
        Dictionary with GMB metrics
    """
    # Create mappings from post_id to predictions and ground truth
    prediction_scores = {}
    ground_truth = {}
    
    for i, (_, row) in enumerate(test_df.iterrows()):
        post_id = row['post_id'] if 'post_id' in row else i
        
        # For binary classification, use probability of positive class
        # For multiclass, use probability of predicted class
        if probabilities.shape[1] == 2:  # Binary
            prediction_scores[post_id] = probabilities[i, 1]  # Score for positive class
        else:  # Multi-class
            prediction_scores[post_id] = probabilities[i, predictions[i]]
            
        # Convert to binary for bias evaluation (toxic vs non-toxic)
        if 'final_label' in row:
            if row['final_label'] in ['toxic', 'hatespeech', 'offensive']:
                ground_truth[post_id] = 1
            else:
                ground_truth[post_id] = 0
        else:
            # Use true_labels directly if final_label not in DataFrame
            ground_truth[post_id] = 1 if true_labels[i] > 0 else 0
    
    # Calculate metrics for each target group and method
    bias_metrics = defaultdict(lambda: defaultdict(dict))
    methods = ['subgroup', 'bpsn', 'bnsp']
    
    for method in methods:
        for group in target_groups:
            # Get positive and negative samples based on the method
            positive_ids, negative_ids = get_bias_evaluation_samples(test_df, method, group)
            
            if len(positive_ids) == 0 or len(negative_ids) == 0:
                continue  # Skip if no samples for this group/method
                
            # Collect ground truth and predictions
            y_true = []
            y_score = []
            
            for post_id in positive_ids:
                if post_id in ground_truth and post_id in prediction_scores:
                    y_true.append(ground_truth[post_id])
                    y_score.append(prediction_scores[post_id])
                
            for post_id in negative_ids:
                if post_id in ground_truth and post_id in prediction_scores:
                    y_true.append(ground_truth[post_id])
                    y_score.append(prediction_scores[post_id])
            
            # Calculate AUC if we have enough samples with both classes
            if len(y_true) > 10 and len(set(y_true)) > 1:
                try:
                    auc = roc_auc_score(y_true, y_score)
                    bias_metrics[method][group] = auc
                except ValueError:
                    # Skip if there's an issue with ROC AUC calculation
                    pass
    
    # Calculate GMB for each method
    gmb_metrics = {}
    power = -5  # Power parameter for generalized mean
    
    for method in methods:
        if not bias_metrics[method]:
            continue
            
        scores = list(bias_metrics[method].values())
        if not scores:
            continue
            
        # Calculate generalized mean with p=-5
        power_mean = np.mean([score ** power for score in scores]) ** (1/power)
        gmb_metrics[f'GMB-{method.upper()}-AUC'] = power_mean
    
    # Calculate a combined GMB score that includes all methods
    all_scores = []
    for method in methods:
        all_scores.extend(list(bias_metrics[method].values()))
    
    if all_scores:
        gmb_metrics['GMB-COMBINED-AUC'] = np.mean([score ** power for score in all_scores]) ** (1/power)
    
    return gmb_metrics

def get_bias_evaluation_samples(data, method, group):
    """
    Get positive and negative sample IDs for bias evaluation based on method and group
    
    Args:
        data: DataFrame with test data
        method: Bias evaluation method ('subgroup', 'bpsn', or 'bnsp')
        group: Target group to evaluate
        
    Returns:
        Tuple of (positive_ids, negative_ids)
    """
    positive_ids = []
    negative_ids = []
    
    for _, row in data.iterrows():
        # Skip if no target_groups column or no post_id
        if 'target_groups' not in row or 'post_id' not in row:
            continue
            
        target_groups = row['target_groups']
        if target_groups is None:
            continue
            
        post_id = row['post_id']
        is_in_group = group in target_groups
        
        # Convert various label formats to binary toxic/non-toxic
        if 'final_label' in row:
            is_toxic = row['final_label'] in ['toxic', 'hatespeech', 'offensive']
        else:
            continue
        
        if method == 'subgroup':
            # Only consider samples mentioning the group
            if is_in_group:
                if is_toxic:
                    positive_ids.append(post_id)
                else:
                    negative_ids.append(post_id)
                    
        elif method == 'bpsn':
            # Compare non-toxic posts mentioning the group with toxic posts NOT mentioning the group
            if is_in_group and not is_toxic:
                negative_ids.append(post_id)
            elif not is_in_group and is_toxic:
                positive_ids.append(post_id)
                
        elif method == 'bnsp':
            # Compare toxic posts mentioning the group with non-toxic posts NOT mentioning the group
            if is_in_group and is_toxic:
                positive_ids.append(post_id)
            elif not is_in_group and not is_toxic:
                negative_ids.append(post_id)
    
    return positive_ids, negative_ids