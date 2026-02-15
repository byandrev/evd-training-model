import os

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main():
    # 1. Load Data
    print("Loading data...")
    try:
        df = pd.read_csv("comentarios.csv")
    except FileNotFoundError:
        print("Error: 'comentarios.csv' not found.")
        return

    # 2. Preprocessing
    print("Preprocessing data...")
    # Map labels to integers
    label_map = {"normal": 0, "critico": 1, "muy_critico": 2}

    # Verify all labels correspond to the map
    if not set(df["label"].unique()).issubset(set(label_map.keys())):
        print(
            f"Warning: Found unknown labels. Expected {list(label_map.keys())}, found {df['label'].unique()}"
        )
        # Optional: Filter or error out. For now, let's filter to be safe or just map and dropna
        df = df[df["label"].isin(label_map.keys())].copy()

    df["label"] = df["label"].map(label_map)

    # Split dataset
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 3. Tokenizer
    model_name = "pysentimiento/robertuito-base-uncased"
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128
        )

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Set format for pytorch
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    # 4. Model Setup
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 5. Metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,  # You can adjust this
        per_device_train_batch_size=8,  # Adjust based on GPU memory
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 8. Train
    print("Starting training...")
    trainer.train()

    # 9. Evaluate
    print("Evaluating...")
    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")

    # 10. Save Model
    save_path = "./modelo_fine_tuned"
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Done!")


if __name__ == "__main__":
    main()
