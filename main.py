from transformers import TrainingArguments, Trainer, T5ForConditionalGeneration
from dataset_preparing import prepare_dataset
import wandb, os
import evaluate

# Prepare dataset
train_dataset, val_dataset, tokenizer = prepare_dataset()

# Define compute_metrics function
metric = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode the predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate BLEU score
    bleu_score = metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

    return bleu_score

# Login wandb
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Create training arguments
training_args = TrainingArguments(
    output_dir='./results/code-summarization/code-t5-small',
    run_name='test-codet5-small',
    report_to="wandb",
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_accumulation_steps=256,
    torch_empty_cache_steps=128,
    logging_dir='./logs/code-summarization/code-t5-small',
    logging_steps=100,
    save_total_limit=2,
    fp16=True
)

# Load model
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
model.generation_config.max_length = 100
model.generation_config.num_beams = 3

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    preprocess_logits_for_metrics=(lambda logits, labels: logits[0].argmax(dim=-1)),
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save Fined-Tuned Model
model.save_pretrained('./results/code-summarization/code-t5-small')
tokenizer.save_pretrained('./results/code-summarization/code-t5-small')