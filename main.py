from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from collections import Counter

MODEL_NAME = "pysentimiento/robertuito-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

label2id = {"normal": 0, "critico": 1, "muy_critico": 2}
id2label = {v: k for k, v in label2id.items()}


dataset = load_dataset("csv", data_files="comentarios.csv")
dataset = dataset["train"].train_test_split(test_size=0.2)


def tokenize(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=128
    )


def compute_metrics(eval_pred):
    # logits, labels = eval_pred
    # preds = logits.argmax(axis=1)

    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     labels, preds, average="weighted"
    # )

    # acc = accuracy_score(labels, preds)

    # return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    report = classification_report(labels, preds, output_dict=True)

    return {
        "f1_muy_critico": report["2"]["f1-score"],
        "recall_muy_critico": report["2"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }


dataset = dataset.map(tokenize, batched=True)
dataset = dataset.map(lambda x: {"labels": label2id[x["label"]]})
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id
)


class WeightedTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 3.0, 6.0]).to(logits.device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="./modelo_set",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.05,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./modelo_set_riesgo")
tokenizer.save_pretrained("./modelo_set_riesgo")
