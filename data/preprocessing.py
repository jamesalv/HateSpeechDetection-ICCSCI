import json
from typing import Dict, List, Any, Tuple
import pandas as pd
from collections import Counter
from tqdm import tqdm
import re
import spacy

# Import ekphrasis components
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize ekphrasis text processor
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=[
        "url",
        "email",
        "percent",
        "money",
        "phone",
        "user",
        "time",
        "date",
        "number",
    ],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated", "emphasis", "censored"},
    segmenter="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=True,
    spell_correction=True,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons],
)


def clean_html(text):
    """Remove HTML tags from text"""
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", text)
    return cleantext

def preprocess_text(text):
    """
    Preprocess text using ekphrasis and additional cleaning
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Preprocessed text as a string
    """
    # First clean any HTML
    text = clean_html(text)
    
    # Process with ekphrasis
    word_list = text_processor.pre_process_doc(text)
    
    # Remove annotation markers
    remove_words = [
        "<allcaps>", "</allcaps>", 
        "<hashtag>", "</hashtag>", 
        "<elongated>", "</elongated>",
        "<emphasis>", "</emphasis>", 
        "<repeated>", "</repeated>"
    ]
    word_list = [word for word in word_list if word not in remove_words]
    
    # Remove angle brackets from annotated words
    processed_text = " ".join(word_list)
    processed_text = re.sub(r"[<\*>]", "", processed_text)
    
    # Use spaCy for additional normalization (without tokenization)
    doc = nlp(processed_text)
    # Create a normalized string, preserving meaningful whitespace
    normalized_text = " ".join([token.text for token in doc])
    
    return normalized_text

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
    
def process_raw_entries(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process raw data entries without classification-specific steps.
    
    Args:
        data: Raw dataset
        
    Returns:
        Tuple of (processed_entries, count_confused)
    """
    processed_entries = []
    count_confused = 0
    
    # Process each entry with progress bar
    for key, value in tqdm(data.items(), desc="Processing entries", unit="entry"):
        processed_entry = {}
        processed_entry["post_id"] = key
        
        # Combine post_tokens to a single string and preprocess
        raw_text = " ".join(value["post_tokens"])
        processed_entry["text"] = raw_text  # Store original text
        processed_entry["processed_text"] = preprocess_text(raw_text)  # Store preprocessed text
        
        # Extract labels and target groups
        # Only select target groups which at least selected by twon annotator
        labels = [annot["label"] for annot in value["annotators"]]
        target_groups = []
        for annot in value["annotators"]:
            target_groups.extend(annot["target"])
        counter_groups = Counter(target_groups)
        target_groups = [group for group, count in counter_groups.items() if count > 1]
        
        # Skip entries where all annotators disagree
        if len(set(labels)) == 3:
            count_confused += 1
            continue
        
        # Store labels for later classification
        processed_entry["labels"] = labels
        
        # Process target groups (remove duplicates)
        processed_entry["target_groups"] = list(set(target_groups))
        
        processed_entries.append(processed_entry)
    
    return processed_entries, count_confused

def create_dataset_with_labels(processed_entries: List[Dict[str, Any]], num_classes: int = 3) -> pd.DataFrame:
    """
    Create final dataset with appropriate labels based on the number of classes.
    
    Args:
        processed_entries: List of processed data entries
        num_classes: Number of classes for classification (2 or 3)
        
    Returns:
        DataFrame with appropriate labels
    """
    dataset_entries = []
    
    for entry in processed_entries:
        # Create a new entry with all the existing fields
        new_entry = entry.copy()
        
        # Determine final label
        new_entry["final_label"] = Counter(entry["labels"]).most_common()[0][0]
        
        # Convert to binary classification if needed
        if num_classes == 2:
            if new_entry['final_label'] in ('hatespeech', 'offensive'):
                new_entry['final_label'] = 'toxic'
            else:
                new_entry['final_label'] = 'non-toxic'
        
        # Remove the temporary labels field
        del new_entry["labels"]
        
        dataset_entries.append(new_entry)
    
    return pd.DataFrame(dataset_entries)

def preprocess_datasets(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess data, creating both 2-class and 3-class versions efficiently.
    
    Args:
        data_path: Path to raw dataset file
        
    Returns:
        Tuple of (data_3class, data_2class)
    """
    print("Loading raw data...")
    raw_data = load_raw_data(data_path)
    
    print("Processing and preprocessing entries...")
    processed_entries, count_confused = process_raw_entries(raw_data)
    
    # Print statistics
    print(f"Initial data: {len(raw_data)}")
    print(f"Uncertain data: {count_confused}")
    print(f"Total processed entries: {len(processed_entries)}")
    
    print("Creating 3-class dataset...")
    data_3class = create_dataset_with_labels(processed_entries, num_classes=3)
    
    print("Creating 2-class dataset...")
    data_2class = create_dataset_with_labels(processed_entries, num_classes=2)
    
    return data_3class, data_2class