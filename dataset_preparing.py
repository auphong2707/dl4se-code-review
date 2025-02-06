import os
import json

from transformers import RobertaTokenizer
from datasets import Dataset
from huggingface_hub import snapshot_download, login

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

# Example input-output pair
# input_text = "Summarize: ~Code~"
# output_text = "Sumarized Content"

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def preprocess_data(data_point):
    # Convert code tokens into a string (space-separated)
    code_text = " ".join(data_point['code_tokens'])

    # Convert docstring tokens into a string (space-separated)
    docstring_text = " ".join(data_point['docstring_tokens'])

    # Add task prefix for CodeT5
    input_text = f"Summarize code: {code_text} TL;DR:"

    return input_text, docstring_text

def tokenize_data(data_point):
    input_text, target_summary = preprocess_data(data_point)

    # Tokenize input code and output summary
    model_inputs = tokenizer(input_text, max_length=200, truncation=True, padding='max_length')
    labels = tokenizer(target_summary, max_length=100, truncation=True, padding='max_length')

    # Add labels
    model_inputs['labels'] = labels['input_ids']

    return model_inputs


def prepare_dataset():
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

    # Convert data into a Dataset object
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Tokenize the data
    train_dataset = train_dataset.map(tokenize_data, batched=False, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(tokenize_data, batched=False, remove_columns=val_dataset.column_names)

    return train_dataset, val_dataset, tokenizer

if __name__ == "__main__":
    train_dataset, val_dataset, tokenizer = prepare_dataset()
    print(train_dataset[0])
    print(val_dataset[0])