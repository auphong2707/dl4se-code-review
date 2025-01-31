from transformers import T5Tokenizer
from datasets import Dataset

tokenizer = T5Tokenizer.from_pretrained("Salesforce/codet5-base")

# Example input-output pair
# input_text = "Summarize: ~Code~"
# output_text = "Sumarized Content"

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
    model_inputs = tokenizer(input_text, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(target_summary, max_length=256, truncation=True, padding='max_length')

    # Add labels
    model_inputs['labels'] = labels['input_ids']

    return model_inputs


def prepare_dataset(train_data, val_data):
    # Convert data into a Dataset object
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Tokenize the data
    train_dataset = train_dataset.map(tokenize_data, batched=True)
    val_dataset = val_dataset.map(tokenize_data, batched=True)

    return train_dataset, val_dataset, tokenizer