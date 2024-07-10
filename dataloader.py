import pickle
import torch
from torch.utils.data import Dataset

class TokenizedDataset(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
            (
                self.tokens,
                self.masks,
                _,
                _,
                _,
                self.claim_labels,
                self.outcome_labels,
                _,
            ) = data

            # Original shape: (batch_size, 1, max_seq_len)
            self.tokens = self.tokens.squeeze(1)

            # Original shape: (batch_size, 1, max_seq_len)
            self.masks = self.masks.squeeze(1)

        # Combine claim_labels and outcome_labels into a single labels tensor
        self.labels = self.combine_labels(self.claim_labels, self.outcome_labels)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
      # Format output
        return {
            "input_ids": torch.tensor(self.tokens[idx]),
            "attention_mask": torch.tensor(self.masks[idx]),
            "labels": torch.tensor(self.labels[idx]),
        }

    def combine_labels(self, claim_labels, outcome_labels):
        unclaimed_is_positive = outcome_labels[claim_labels == 0].any()
        assert not unclaimed_is_positive, "Unclaimed claim has positive outcome"
        return claim_labels + outcome_labels

# Create formatted Datasets for LLaMA
llama_train_dataset = TokenizedDataset("./ECHR/Outcome/llama/tokenized_train.pkl")
llama_test_dataset = TokenizedDataset("./ECHR/Outcome/llama/tokenized_test.pkl")
llama_dev_dataset = TokenizedDataset("./ECHR/Outcome/llama/tokenized_dev.pkl")

# Create formatted Datasets for BERT
bert_train_dataset = TokenizedDataset("./ECHR/Outcome/bert/tokenized_train.pkl")
bert_test_dataset = TokenizedDataset("./ECHR/Outcome/bert/tokenized_test.pkl")
bert_dev_dataset = TokenizedDataset("./ECHR/Outcome/bert/tokenized_dev.pkl")
