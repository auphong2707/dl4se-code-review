from transformers import TrainingArguments, Trainer, T5ForConditionalGeneration
from dataset_preparing import prepare_dataset
import json

# Prepare dataset
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

train_data = load_jsonl('data/Review-Comment-Generation/trans_data/train.jsonl')
val_data = load_jsonl('data/Review-Comment-Generation/trans_data/train.jsonl')

train_dataset, val_dataset, tokenizer = prepare_dataset(train_data, val_data)

# Create training arguments
training_args = TrainingArguments(
    output_dir='./results/code-summarization/code-t5-base',
    evaluation_strategy='epochs',
    save_strategy='epoch',
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=100,
    weight_decay=0.01,
    logging_dir='./logs/code-summarization/code-t5-base',
    logging_steps=10,
    save_total_limit=2,
    fp16=True, # Enable mixed precision training
)

# Load model
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save Fined-Tuned Model
model.save_pretrained('./results/code-summarization/code-t5-base')
tokenizer.save_pretrained('./results/code-summarization/code-t5-base')