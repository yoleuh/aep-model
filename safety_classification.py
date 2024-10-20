"""
Safety Classification Project

Created by:
- Reid Ammer (ammer.5@osu.edu)
- Brian Tan (tan.1220@osu.edu)

For the AEP Safety Observation Challenge at HackOHI/O 2024

Description:
This script trains a model to classify safety observations into high priority
and standard priority categories using the DistilBERT model.
"""

import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

# Configurable variables
INPUT_FILE = "data/data.csv"  # Path to the input CSV file
MAX_STEPS = 500  # Maximum number of training steps
NUM_TRAIN_EPOCHS = 3  # Number of training epochs
TRAIN_BATCH_SIZE = 16  # Training batch size
EVAL_BATCH_SIZE = 64  # Evaluation batch size
WARMUP_STEPS = 500  # Number of warmup steps for learning rate scheduler
WEIGHT_DECAY = 0.01  # Weight decay for regularization
USE_CPU = True  # Force CPU usage instead of GPU

# Force CPU usage if specified (Do not change)
if USE_CPU:
    torch.cuda.is_available = lambda: False
device = torch.device("cpu" if USE_CPU else "cuda")

# Load data
df = pd.read_csv(INPUT_FILE)


# Define classification function
# You can add new keywords here
def classify_priority(comment):
    high_priority_keywords = [
        "fall",
        "hazard",
        "injury",
        "danger",
        "unsafe",
        "risk",
        "accident",
        "emergency",
        "critical",
        "severe",
        "urgent",
        "immediate",
        "life-threatening",
        "serious",
        "fatal",
        "suspended load",
        "elevation",
        "mobile equipment",
        "vehicle",
        "rotating equipment",
        "high temperature",
        "steam",
        "fire",
        "explosion",
        "excavation",
        "trench",
        "electrical",
        "arc flash",
        "toxic",
        "radiation",
    ]
    return (
        1
        if any(keyword in str(comment).lower() for keyword in high_priority_keywords)
        else 0
    )


# Apply classification to dataset
df["label"] = df["PNT_ATRISKNOTES_TX"].apply(classify_priority)

# Split data
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Dataset objects
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Load tokenizer and model
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(device)


# Tokenize data
def tokenize_function(examples):
    """Tokenize the input text data."""
    return tokenizer(
        examples["PNT_ATRISKNOTES_TX"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


# Tokenize and prepare datasets
def tokenize_and_prepare(dataset):
    tokenized = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )
    tokenized = tokenized.add_column("label", dataset["label"])
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized


tokenized_train = tokenize_and_prepare(train_dataset)
tokenized_eval = tokenize_and_prepare(eval_dataset)


# Define metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="no",
    # Max Steps
    max_steps=MAX_STEPS,
    use_cpu=USE_CPU,
)

# Create and run trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")

# Save the model and tokenizer
model_save_path = "./saved_model"
tokenizer_save_path = "./saved_tokenizer"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)
print(f"Model saved to {model_save_path}")
print(f"Tokenizer saved to {tokenizer_save_path}")


# Function to load the model and tokenizer
def load_model_and_tokenizer(model_path, tokenizer_path):
    loaded_model = DistilBertForSequenceClassification.from_pretrained(model_path).to(
        device
    )
    loaded_tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    return loaded_model, loaded_tokenizer


# Function to classify new inputs
def classify_new_input(input_text, model, tokenizer):
    inputs = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return "High Priority" if predicted_class == 1 else "Standard Priority"


# Example of how to use the saved model for future inputs
if __name__ == "__main__":
    # Check if saved model exists
    if os.path.exists(model_save_path) and os.path.exists(tokenizer_save_path):
        # Load the saved model and tokenizer
        loaded_model, loaded_tokenizer = load_model_and_tokenizer(
            model_save_path, tokenizer_save_path
        )

        # Classify all comments in the original dataset
        df["predicted_priority"] = df["PNT_ATRISKNOTES_TX"].apply(
            lambda x: classify_new_input(x, loaded_model, loaded_tokenizer)
        )

        # Save the results to a new CSV file
        output_path = "results/output_data.csv"
        df.to_csv(output_path, index=False)
        print("Results saved to {output_path}")

    else:
        print("Saved model not found. Please run the training script first.")
