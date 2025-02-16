"""
This script prepares the dataset for training the CodeT5 model.
It loads the dataset, preprocesses the data, and tokenizes the data for training.

Example input-output pair:
 - input_text = "Summarize code: def add(a, b): return a + b TL;DR:"
 - output_text = "Add two numbers"
"""

# Import the necessary libraries
import os
import json

from transformers import AutoTokenizer
from datasets import Dataset
from huggingface_hub import snapshot_download, login
from typing import Tuple

from constants import CS_INPUT_MAX_LENGTH, CS_OUTPUT_MAX_LENGTH

TOKENIZER = None
PRESUFFIX = False


def load_jsonl(file_path) -> list:
    """
    Load a JSONL (JSON Lines) file and return its contents as a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file to be loaded.

    Returns:
        list: A list of dictionaries, each representing a JSON object from a line in the file.
    """
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def preprocess_data(data_point) -> Tuple[str, str]:
    """
    Preprocess a data point by converting code and docstring tokens into strings and formatting them for the CodeT5 model.

    Args:
        data_point (dict): A dictionary containing 'code_tokens' and 'docstring_tokens'.

    Returns:
        tuple: A tuple containing the formatted input text and the target summary text.
    """
    # Convert code tokens into a string (space-separated)
    code_text = " ".join(data_point['code_tokens'])

    # Convert docstring tokens into a string (space-separated)
    docstring_text = " ".join(data_point['docstring_tokens'])

    if PRESUFFIX:
        input_text = f"Summarize code: {code_text} TL;DR:"
    else:
        input_text = code_text

    return input_text, docstring_text


def tokenize_data(data_point) -> dict:
    """
    Tokenizes the input data point into model inputs and labels.

    Args:
        data_point (tuple): A tuple containing the input text and target summary.

    Returns:
        dict: A dictionary containing tokenized input text and labels with padding and truncation applied.
    """
    input_text, target_summary = preprocess_data(data_point)

    # Tokenize input code and output summary
    model_inputs = TOKENIZER(input_text, max_length=CS_INPUT_MAX_LENGTH, truncation=True, padding='max_length')
    labels = TOKENIZER(target_summary, max_length=CS_OUTPUT_MAX_LENGTH, truncation=True, padding='max_length')

    # Add labels
    model_inputs['labels'] = labels['input_ids']

    return model_inputs


def prepare_dataset(tokenizer_name, presufix=True) -> Tuple[Dataset, Dataset, Dataset, AutoTokenizer]:
    """
    Prepares the dataset for training and validation.

    This function performs the following steps:
    1. Downloads the dataset from the Hugging Face repository.
    2. Loads the training and validation data from JSONL files.
    3. Converts the loaded data into Dataset objects.
    4. Tokenizes the data using a specified tokenizer.

    Returns:
        tuple: A tuple containing the tokenized training dataset, validation dataset, and the tokenizer.
    """
    # Load the tokenizer
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)

    # Set the global variable for prefix/suffix
    global PRESUFFIX
    PRESUFFIX = presufix

    # Download the dataset
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    snapshot_download(repo_id="auphong2707/dl4se-code-review-dataset", 
                      repo_type="dataset",
                      local_dir=".",
                      allow_patterns="*.jsonl"
                     )

    # Load data
    train_data = load_jsonl('data/Review-Comment-Generation/trans_data/train.jsonl')
    val_data = load_jsonl('data/Review-Comment-Generation/trans_data/val.jsonl')
    test_data = load_jsonl('data/Review-Comment-Generation/trans_data/test.jsonl')

    # Convert data into a Dataset object
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)

    # Tokenize the data
    train_dataset = train_dataset.map(tokenize_data, batched=False, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(tokenize_data, batched=False, remove_columns=val_dataset.column_names)
    test_dataset = test_dataset.map(tokenize_data, batched=False, remove_columns=test_dataset.column_names)

    return train_dataset, val_dataset, test_dataset, TOKENIZER


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, TOKENIZER = prepare_dataset()
    print("Train example\n", train_dataset[0])
    print("Validation example \n", val_dataset[0])
    print("Test example \n", test_dataset[0])