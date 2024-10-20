"""
Safety Classification Project

Created by:
- Reid Ammer (ammer.5@osu.edu)
- Brian Tan (tan.1220@osu.edu)

For the AEP Safety Observation Challenge at HackOHI/O 2024

Description:
This script loads the safety classification model previously fine-tuned by
safety_classification.py and allows users to interactively input new safety
comments for classification. It provides real-time predictions, categorizing
each input as either "High Priority" or "Standard Priority" along with a
confidence score.

Usage: Run the script and enter safety comments when prompted. Type 'quit' to exit.
"""

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Configurable variables
USE_CPU = True  # Force CPU usage instead of GPU

# Force CPU usage if specified (Do not change)
if USE_CPU:
    torch.cuda.is_available = lambda: False
device = torch.device("cpu" if USE_CPU else "cuda")

# Paths to saved model and tokenizer
MODEL_PATH = "./saved_model"
TOKENIZER_PATH = "./saved_tokenizer"


def load_model_and_tokenizer(model_path, tokenizer_path):
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    return model, tokenizer


def classify_comment(comment, model, tokenizer):
    inputs = tokenizer(
        comment, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    return "High Priority" if predicted_class == 1 else "Standard Priority", confidence


def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)
    print("Model and tokenizer loaded successfully.")

    while True:
        user_input = input("\nEnter a safety comment (or 'quit' to exit): ")
        if user_input.lower() == "quit":
            break

        classification, confidence = classify_comment(user_input, model, tokenizer)
        print(f"Classification: {classification}")
        print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
