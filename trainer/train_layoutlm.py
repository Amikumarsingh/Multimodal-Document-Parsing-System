import os
from transformers import (
    LayoutLMv3ForTokenClassification, 
    LayoutLMv3Processor, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset, load_metric
import numpy as np
import torch

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataloader import get_labels

def train_layoutlm(model_id="microsoft/layoutlmv3-base", dataset_name="nielsr/funsd", output_dir="./results"):
    dataset = load_dataset(dataset_name)
    labels = get_labels()
    label2id = {label: i for i, label in label2id.items()} if 'label2id' in locals() else {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    processor = LayoutLMv3Processor.from_pretrained(model_id, apply_ocr=False)

    def prepare_dataset(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        words = examples["words"]
        boxes = examples["bboxes"]
        word_labels = examples["ner_tags"]

        encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
                             truncation=True, padding="max_length")
        return encoding

    train_dataset = dataset["train"].map(prepare_dataset, batched=True, remove_columns=dataset["train"].column_names)
    test_dataset = dataset["test"].map(prepare_dataset, batched=True, remove_columns=dataset["test"].column_names)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_id, 
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label
    )

    metric = load_metric("seqeval", trust_remote_code=True)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=1000,
        per_device_train_batch_size=2,
        learning_rate=1e-5,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForTokenClassification(processor.tokenizer),
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model"))

if __name__ == "__main__":
    train_layoutlm()
