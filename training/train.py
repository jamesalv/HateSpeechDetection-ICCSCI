import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Any, Union, Optional

from models.classifier import TransformerClassifier
from data.dataset import prepare_data_loaders

def train_and_evaluate_model(
    model_name: str, 
    data_df: pd.DataFrame, 
    num_classes: int, 
    batch_size: int = 16, 
    epochs: int = 4,
    max_length: int = 128,
    learning_rate: float = 2e-5,
    warmup_steps: int = 0,
    model_dir: str = "saved_models",
    auto_weighted: bool = False
) -> Dict[str, Any]:
    """
    Train and evaluate a model with the given parameters
    
    Args:
        model_name: Name of the Hugging Face transformer model
        data_df: DataFrame with preprocessed data
        num_classes: Number of classes (2 or 3)
        batch_size: Batch size for training
        epochs: Number of training epochs
        max_length: Maximum sequence length for tokenization
        model_dir: Directory to save models
        auto_weighted: Whether to use automatic weighting for classes
        
    Returns:
        Dictionary with results
    """
    # Initialize model
    classifier = TransformerClassifier(model_name, num_classes)
    
    # Prepare data loaders
    train_dataloader, val_dataloader, test_dataloader, label_map, test_df, label_to_weight = prepare_data_loaders(
        data_df, 
        classifier.tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        auto_weighted=auto_weighted,
    )
    
    # Invert label map for later use
    inv_label_map = {v: k for k, v in label_map.items()}
    
    class_weights = {label_map[label]: weight for label, weight in label_to_weight.items()}
    print(f"Class weights: {class_weights}")
    ordered_weights = np.array([class_weights[i] for i in range(len(class_weights))])
    
    # Train model
    history = classifier.train(
        train_dataloader, 
        val_dataloader, 
        class_weights=ordered_weights,
        epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps
    )
    
    # Evaluate on test set
    test_loss, test_accuracy, test_f1 = classifier.evaluate(test_dataloader)
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1 Score: {test_f1:.4f}")
    
    # Get detailed metrics
    predictions, true_labels, probabilities = classifier.predict(test_dataloader)
    
    # Convert numeric labels back to text
    text_preds = [inv_label_map[pred] for pred in predictions]
    text_true = [inv_label_map[label] for label in true_labels]
    
    # Calculate additional metrics
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    
    # For AUROC, we need to handle multiclass case
    if num_classes == 2:
        auroc = roc_auc_score(true_labels, probabilities[:, 1])
    else:
        # One-vs-Rest approach for multiclass
        auroc = roc_auc_score(
            np.eye(num_classes)[true_labels], 
            probabilities, 
            average='macro', 
            multi_class='ovr'
        )
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_map.keys()),
                yticklabels=list(label_map.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.show()
    
    # Save model with descriptive name
    model_type = "binary" if num_classes == 2 else "3class"
    model_save_path = os.path.join(model_dir, f"{model_name.replace('/', '_')}_{model_type}")
    classifier.save_model(model_save_path)
    
    # Prepare target group analysis
    if 'target_groups' in test_df.columns:
        # Analyze performance by target group
        test_df['predicted'] = text_preds
        test_df['true'] = text_true
        test_df['correct'] = test_df['predicted'] == test_df['true']
        
        # Extract target groups to analyze
        all_targets = []
        for targets in test_df['target_groups']:
            all_targets.extend(targets)
        
        from collections import Counter
        top_targets = Counter(all_targets).most_common(10)
        print("\nPerformance by Target Group:")
        
        target_metrics = {}
        for target, _ in top_targets:
            # Filter rows containing this target
            target_df = test_df[test_df['target_groups'].apply(lambda x: target in x)]
            if len(target_df) < 10:  # Skip if too few samples
                continue
                
            target_acc = target_df['correct'].mean()
            print(f"{target}: Accuracy = {target_acc:.4f} (n={len(target_df)})")
            target_metrics[target] = target_acc
    else:
        target_metrics = None
    
    # Return comprehensive results
    results = {
        'model_name': model_name,
        'num_classes': num_classes,
        'history': history,
        'metrics': {
            'accuracy': test_accuracy,
            'f1_score': test_f1,
            'precision': precision,
            'recall': recall,
            'auroc': auroc,
            'loss': test_loss
        },
        'confusion_matrix': cm,
        'label_map': label_map,
        'target_metrics': target_metrics
    }
    
    return results

def run_model_comparison(
    models_to_compare: List[str], 
    data_3class: pd.DataFrame, 
    data_2class: pd.DataFrame, 
    batch_size: int = 16, 
    epochs: int = 3,
    auto_weighted: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Run comparison of multiple models on both binary and 3-class tasks
    
    Args:
        models_to_compare: List of model names to compare
        data_3class: DataFrame with 3-class data
        data_2class: DataFrame with binary data
        batch_size: Batch size for training
        epochs: Number of training epochs
        
    Returns:
        Dictionary with all results
    """
    results = {
        'binary': {},
        '3class': {}
    }
    
    # First run 3-class models
    print("\n=== Running 3-Class Classification Models ===\n")
    for model_name in models_to_compare:
        print(f"\nTraining {model_name} for 3-class classification")
        results['3class'][model_name] = train_and_evaluate_model(
            model_name, 
            data_3class, 
            num_classes=3,
            batch_size=batch_size,
            epochs=epochs,
            auto_weighted=auto_weighted
        )
    
    # Then run binary models
    print("\n=== Running Binary Classification Models ===\n")
    for model_name in models_to_compare:
        print(f"\nTraining {model_name} for binary classification")
        results['binary'][model_name] = train_and_evaluate_model(
            model_name, 
            data_2class, 
            num_classes=2,
            batch_size=batch_size,
            epochs=epochs,
            auto_weighted=auto_weighted
        )
    
    return results