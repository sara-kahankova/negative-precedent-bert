import torch
import wandb
from dataloader import bert_train_dataset, bert_test_dataset, bert_dev_dataset
from model import BertClassifier
from sklearn.metrics import f1_score
from transformers import Trainer, TrainingArguments


def compute_metrics(pred):
  # computes F1 score breakdown
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

    use_lora = False # applies LoRA to BERT
    use_two_berts = True # MonoBERT or BERT replica
    wandb.init(project="negative-replica", entity="###redacted")
    wandb.config.update(
        {
            "use_lora": use_lora,
            "use_two_berts": use_two_berts,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BertClassifier(
        n_claims=14, use_lora=use_lora, use_two_berts=use_two_berts
    ).to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        learning_rate=3e-5,
        per_device_train_batch_size=16,  # in Valvoda paper it is 16
        per_device_eval_batch_size=16,  # in Valvoda paper it is 16
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_negative",
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",  # Report metrics to W&B
    )  # not including optimiser as adamw_torch is the default

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=bert_train_dataset,
        eval_dataset=bert_dev_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate(eval_dataset=bert_test_dataset)
    print(f"Validation Results: {eval_results}")

    wandb.finish()


if __name__ == "__main__":
    main()
