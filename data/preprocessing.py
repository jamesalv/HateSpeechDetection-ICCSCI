import json
from typing import Dict, List, Any
import pandas as pd
from collections import Counter
from tqdm import tqdm

def load_raw_data(file_path: str) -> Dict[str, Any]:
    """
    Load the raw JSON data from file.
    
    Args:
        file_path: Path to the JSON dataset file
        
    Returns:
        Loaded JSON data
    """
    with open(file_path, "r") as file:
        return json.load(file)
    
def preprocess_dataset(data: Dict[str, Any], num_classes: int = 3) -> pd.DataFrame:
    """
    Preprocess the entire dataset.
    
    Args:
        data: Raw dataset
        num_classes: Number of classes for classification (2 or 3)
        
    Returns:
        Preprocessed dataframe
    """
    all_data = []
    count_confused = 0
    
    # Process each entry with progress bar
    for key, value in tqdm(data.items(), desc="Processing entries", unit="entry"):
        processed_entry = {}
        processed_entry["post_id"] = key
        
        # Combine post_tokens to a single string
        processed_entry["text"] = " ".join(value["post_tokens"])
        
        # Extract labels and target groups
        labels = [annot["label"] for annot in value["annotators"]]
        target_groups = []
        for annot in value["annotators"]:
            target_groups.extend(annot["target"])
        
        # Skip entries where all annotators disagree
        if len(set(labels)) == 3:
            count_confused += 1
            continue
        
        # Determine final label
        processed_entry["final_label"] = Counter(labels).most_common()[0][0]
        
        # Convert to binary classification if needed
        if num_classes == 2:
            if processed_entry['final_label'] in ('hatespeech', 'offensive'):
                processed_entry['final_label'] = 'toxic'
            else:
                processed_entry['final_label'] = 'non-toxic'
        
        # Process target groups (remove duplicates)
        processed_entry["target_groups"] = list(set(target_groups))
        
        all_data.append(processed_entry)

    # Print statistics
    print(f"Initial data: {len(data)}")
    print(f"Uncertain data: {count_confused}")
    print(f"Total final data count: {len(all_data)}")

    # Convert to DataFrame
    return pd.DataFrame(all_data)
