import torch
from sklearn.metrics import f1_score
from transformers import Trainer, TrainingArguments
import wandb
from dataloader import dev_dataset, train_dataset, test_dataset
from model import BertClassifier, LlamaClassifier

def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(-1).flatten()
    f1_null, f1_negative, f1_positive = f1_score(labels, preds, average=None)
    f1_all_macro = f1_score(labels, preds, average="macro")

    return {
        "f1_null": f1_null,
        "f1_negative": f1_negative,
        "f1_positive": f1_positive,
        "f1_all": f1_all_macro,
    }

def main():
    wandb.init(project="negative-replica", entity="kahinek1999")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LlamaClassifier(n_claims=14).to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        learning_rate=3e-5,
        per_device_train_batch_size=1,  # in paper it is 16
        per_device_eval_batch_size=1,  # in paper it is 16
        gradient_accumulation_steps=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_negative",
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
    )  # adamw_torch is the default optimizer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Validation Results: {eval_results}")

    wandb.finish()


if __name__ == "__main__":
    main()