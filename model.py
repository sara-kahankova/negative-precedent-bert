
import torch
import torch.nn as nn
from jaxtyping import Float
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor
from transformers import BertForSequenceClassification, LlamaForSequenceClassification

class ReplicaClassifier(nn.Module):

    def __init__(self, n_claims):
        super().__init__()
        self.bert_claims = BertModel.from_pretrained("bert-base-uncased")
        self.bert_outcomes = BertModel.from_pretrained("bert-base-uncased")
        self.n_claims = n_claims
        self.mlp_claims = nn.Sequential(
            nn.Linear(768, 50), nn.ReLU(), nn.Dropout(0.2), nn.Linear(50, n_claims)
        )
        self.mlp_outcomes = nn.Sequential(
            nn.Linear(768, 50), nn.ReLU(), nn.Dropout(0.2), nn.Linear(50, n_claims)
        )
        self.loss_fn = nn.BCELoss(reduction="none")

    def forward(
        self,
        input_ids: Float[Tensor, "batch n_seq"],
        attention_mask: Float[Tensor, "batch n_seq"],
        labels: Float[Tensor, "batch n_claims"],
        ):

        claim_labels = (labels != 0).float()  # 0 = unclaimed
        outcome_labels = (labels == 2).float()  # 2 = positive

        bert_claims_output = self.bert_claims(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state_cls_claims = bert_claims_output.last_hidden_state[:, 0, :]
        mlp_claims_output = self.mlp_claims(last_hidden_state_cls_claims)
        probability_claims = torch.sigmoid(mlp_claims_output)
        loss_claims = self.loss_fn(probability_claims, claim_labels.float())
        bert_outcomes_output = self.bert_outcomes(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state_cls_outcome = bert_outcomes_output.last_hidden_state[:, 0, :]

        mlp_outcomes_output = self.mlp_outcomes(last_hidden_state_cls_outcome)
        probability_outcomes = torch.sigmoid(
            mlp_outcomes_output
        )
        loss_outcomes = self.loss_fn(probability_outcomes, outcome_labels.float())
        loss_outcomes[claim_labels == 0] = 0
        probability_unclaimed = (
            1 - probability_claims
        )
        probability_claimed_accepted = (
            probability_claims * probability_outcomes
        )
        probability_claimed_rejected = probability_claims * (
            1 - probability_outcomes
        )
        final_probs = torch.stack(
            (
                probability_unclaimed,
                probability_claimed_rejected,
                probability_claimed_accepted,
            ),
            dim=2,
        )

        loss_claims = torch.mean(loss_claims)
        loss_outcomes = torch.mean(loss_outcomes)
        loss = (loss_claims + loss_outcomes) / 2
        return {"loss": loss, "predictions": final_probs}

class BertClassifier(nn.Module):
    def __init__(self, n_claims, use_lora=False, use_two_berts=True):
        super().__init__()
        self.use_lora = use_lora
        self.use_two_berts = use_two_berts

        if use_two_berts:
            self.bert_claims = BertForSequenceClassification.from_pretrained("bert-base-uncased", problem_type = "multi_label_classification", num_labels=n_claims)
            self.bert_outcomes = BertForSequenceClassification.from_pretrained("bert-base-uncased", problem_type = "multi_label_classification", num_labels=n_claims)
            self.n_claims = n_claims
        else:
            self.bert_claim_outcome = BertForSequenceClassification.from_pretrained("bert-base-uncased", problem_type = "multi_label_classification", num_labels=n_claims*2)
            self.n_claims = n_claims

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=32,
                lora_alpha=64,
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
                target_modules=[
                    "query",
                    "value",
                    "key",
                ],
            )

            if use_two_berts:
                self.bert_claims = get_peft_model(
                self.bert_claims, lora_config
                )
                self.bert_outcomes = get_peft_model(
                self.bert_outcomes, lora_config
                )
            else:
                self.bert_claim_outcome = get_peft_model(
                self.bert_claim_outcome, lora_config
                )

        self.loss_fn = nn.BCELoss(reduction="none")

    def forward(
        self,
        input_ids: Float[Tensor, "batch n_seq"],
        attention_mask: Float[Tensor, "batch n_seq"],
        labels: Float[Tensor, "batch n_claims"],
        ):

        if self.use_two_berts:
            bert_claims_output = self.bert_claims(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            bert_outcomes_output = self.bert_outcomes(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        else:
            bert_claim_outcome_output = self.bert_claim_outcome(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            bert_claims_output = bert_claim_outcome_output[:, : self.n_claims]
            bert_outcomes_output = bert_claim_outcome_output[:, self.n_claims :]



        claim_labels = (labels != 0).float()  # 0 = unclaimed
        outcome_labels = (labels == 2).float()  # 2 = positive


        probability_claims = torch.sigmoid(bert_claims_output)

        loss_claims = self.loss_fn(probability_claims, claim_labels.float())

        probability_outcomes = torch.sigmoid(bert_outcomes_output)
        loss_outcomes = self.loss_fn(probability_outcomes, outcome_labels.float())
        loss_outcomes[claim_labels == 0] = 0
        probability_unclaimed = (
            1 - probability_claims
        )
        probability_claimed_accepted = (
            probability_claims * probability_outcomes
        )
        probability_claimed_rejected = probability_claims * (
            1 - probability_outcomes
        )
        final_probs = torch.stack(
            (
                probability_unclaimed,
                probability_claimed_rejected,
                probability_claimed_accepted,
            ),
            dim=2,
        )
        loss_claims = torch.mean(loss_claims)
        loss_outcomes = torch.mean(loss_outcomes)
        loss = (loss_claims + loss_outcomes) / 2
        return {"loss": loss, "predictions": final_probs}


class LlamaClassifier(nn.Module):
    def __init__(self, n_claims):
        super().__init__()

        self.llama_claim_outcome = LlamaForSequenceClassification.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            num_labels=n_claims * 2,
        )

        self.n_claims = n_claims
        self.loss_fn = nn.BCELoss(reduction="none")

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["key", "query", "value"],
        )

        self.llama_claim_outcome = get_peft_model(self.llama_claim_outcome, lora_config)

    def forward(
        self,
        input_ids: Float[Tensor, "batch n_seq"],
        attention_mask: Float[Tensor, "batch n_seq"],
        labels: Float[Tensor, "batch n_claims"],
    ):

        claim_labels = (labels != 0).float()  # 0 = unclaimed
        outcome_labels = (labels == 2).float()  # 2 = positive

        llama_claim_outcome_output = self.llama_claim_outcome(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        llama_claims_output = llama_claim_outcome_output[:, : self.n_claims]
        llama_outcomes_output = llama_claim_outcome_output[:, self.n_claims :]

        probability_claims = torch.sigmoid(llama_claims_output)
        loss_claims = self.loss_fn(probability_claims, claim_labels.float())

        probability_outcomes = torch.sigmoid(llama_outcomes_output)
        loss_outcomes = self.loss_fn(probability_outcomes, outcome_labels.float())
        loss_outcomes[claim_labels == 0] = 0


        loss_claims = self.loss_fn(probability_claims, claim_labels
                                   )
        loss_outcomes = self.loss_fn(
            probability_outcomes, outcome_labels
        )
        loss_outcomes[claim_labels == 0] = 0
        probability_unclaimed = (
            1 - probability_claims
        )
        probability_claimed_accepted = (
            probability_claims * probability_outcomes
        )
        probability_claimed_rejected = probability_claims * (
            1 - probability_outcomes
        )
        final_probs = torch.stack(
            (
                probability_unclaimed,
                probability_claimed_rejected,
                probability_claimed_accepted,
            ),
            dim=2,
        )

        loss_claims = torch.mean(loss_claims)
        loss_outcomes = torch.mean(loss_outcomes)
        loss = (loss_claims + loss_outcomes) / 2
        return {"loss": loss, "predictions": final_probs}
