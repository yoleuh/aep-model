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

# Force CPU usage
torch.cuda.is_available = lambda: False
device = torch.device("cpu")

# Load data
df = pd.read_csv("content/data_cleaned.csv")


# Define classification function
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
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_steps=20,
    evaluation_strategy="steps",
    save_strategy="no",
    max_steps=3000,
    no_cuda=True,  # Force CPU usage
)

# Create and run trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

trainer.train()


# Define model-based classification function
def classify_comment_model(comment):
    inputs = tokenizer(
        comment, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to CPU
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return "High Priority" if predicted_class == 1 else "Standard Priority"


# Test the classifier
test_comments = [
    "Worker noticed a loose handrail near a 20-foot drop.",
    "The coffee machine in the break room is out of order.",
    "Exposed electrical wires found in the construction area.",
    "Employee reported feeling dizzy after working in a confined space.",
]

for comment in test_comments:
    result = classify_comment_model(comment)
    print(f"Comment: {comment}")
    print(f"Classification: {result}\n")

# Evaluate the model on the test set
test_results = trainer.evaluate()
print("Test set results:", test_results)
