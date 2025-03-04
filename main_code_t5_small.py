from constants import *
from helper import set_seed
set_seed(SEED)

import wandb, huggingface_hub, os
import evaluate

from transformers import TrainingArguments, Trainer, T5ForConditionalGeneration

from dataset_preparing import prepare_dataset

# [PREPARING DATASET AND FUNCTIONS]
# Login wandb & huggingface
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Prepare the dataset and tokenizer
train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(TOKENIZER_CS_CT5B)

# Define compute_metrics function
metric = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode the predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate BLEU score
    bleu_score = metric.compute(predictions=decoded_preds, 
                                references=[[label] for label in decoded_labels],
                                max_order=4
                            )

    return bleu_score


# [SETTING UP MODEL AND TRAINING ARGUMENTS]
# Set experiment name
EXPERIMENT_NAME = "experiment-1"
EXPERIMENT_RESULTS_DIR = RESULTS_CS_DIR_CT5S + EXPERIMENT_NAME
os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)

# Load model
def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR)
if checkpoint:
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
else:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CS_CT5S)
    model.generation_config.max_length = CS_OUTPUT_MAX_LENGTH
    model.generation_config.num_beams = CS_NUM_BEAMS

# Create training arguments
training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME,
    report_to="wandb",
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=LR_CS_CT5S,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_CS_CT5S,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_CS_CT5S,
    num_train_epochs=NUM_TRAIN_EPOCHS_CS_CT5S,
    weight_decay=WEIGHT_DECAY_CS_CT5S,
    eval_accumulation_steps=EAS_CS_CT5S,
    torch_empty_cache_steps=TEC_CS_CT5S,
    output_dir=EXPERIMENT_RESULTS_DIR,
    logging_dir=EXPERIMENT_RESULTS_DIR + "/logs",
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    seed=SEED
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    preprocess_logits_for_metrics=(lambda logits, labels: logits[0].argmax(dim=-1)),
    compute_metrics=compute_metrics,
)

# [TRAINING]
if checkpoint:
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()

# [EVALUATING]
test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

# [SAVING THINGS]
# Save the model and tokenizer
model.save_pretrained(EXPERIMENT_RESULTS_DIR)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR)

# Save the training arguments
with open(EXPERIMENT_RESULTS_DIR + "/training_args.txt", "w") as f:
    f.write(str(training_args))

# Save the test results
with open(EXPERIMENT_RESULTS_DIR + "/test_results.txt", "w") as f:
    f.write(str(test_results))

# Upload to HuggingFace
api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=RESULTS_DIR,
    repo_id="auphong2707/dl4se-code-review",
    repo_type="model",
    private=True
)