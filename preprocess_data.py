import json
import os
import pickle  # used for saving and loading Python objects to and from disk in a binary format
import re  # regex
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MultiLabelBinarizer,  #  transforming lists of labels into binary label indicators
)
from tqdm import tqdm  # useful for keeping track of the progress of tasks
from transformers import BertTokenizer, AutoTokenizer


def fix_claims(all_facts, all_claims, all_outcomes, case_id, all_arguments):
    allowed = [
        "10",
        "11",
        "13",
        "14",
        "2",
        "3",
        "5",
        "6",
        "7",
        "8",
        "9",
        "P1-1",
        "P1-3",
        "P4-2",
    ]  # what are these? articles?

    new_claims, new_outcomes, new_ids, new_facts, new_arguments = [], [], [], [], []
    for claim, outcome, i, c_id, fact, argument in zip(
        all_claims,
        all_outcomes,
        range(len(all_claims)),
        case_id,
        all_facts,
        all_arguments,
    ):
        n_c = []
        for c in claim:
            if c in allowed:
                n_c.append(c)

        cnt = 0
        flag = True
        if len(n_c) > 0 and len(n_c) >= len(outcome):
            for x in outcome:
                if x not in n_c:
                    flag = False
            if flag:
                n_c.sort()
                outcome.sort()
                new_claims.append(n_c)
                new_outcomes.append(outcome)
                new_ids.append(c_id)
                new_facts.append(fact)
                new_arguments.append(argument)
            else:
                cnt += 1

    return new_facts, new_claims, new_outcomes, new_ids, new_arguments


def get_arguments(data):
    # Extracts arguments from the "LAW" section into [arguments] of each lines
    try:
        arguments = (
            data["text"]
            .split("THE LAW")[1]
            .split("FOR THESE REASONS, THE COURT UNANIMOUSLY")[0]
        )
        """
        The [1] index is used to select the second portion of the split text data.
        The code is discarding everything before the first occurrence of "THE LAW".
        The [0] index is used to select the first portion of the split text data.
        The code is extracting everything before the first occurrence of "FOR THESE REASONS, THE COURT UNANIMOUSLY"
        """
        arguments = arguments.split("\n")  # gets list of lines
        arguments = [
            a.strip() for a in arguments
        ]  # applies strip() to each element (a) in the list arguments
        arguments = list(filter(None, arguments))  # filters empty values
    except:
        return []
    return arguments


def get_data(pretokenized_dir, tokenizer, max_len):
    # prepares data for fixedclaims(), uses get_arguments applying preprocessing functions

    dataset_facts = []
    dataset_arguments = []
    dataset_claims = []
    dataset_outcomes = []
    dataset_ids = []

    paths = ["train", "dev", "test"]
    out_path = ["train_augmented", "dev_augmented", "test_augmented"]

    for case_path, out in zip(paths, out_path):
        all_facts = []
        all_arguments = []
        all_claims = []
        all_outcomes = []
        all_ids = []

        for item in tqdm(os.listdir("ECHR/Outcome/" + case_path)):
            if item.endswith(".json"):
                with open(
                    os.path.join("ECHR/Outcome/" + case_path, item), "r"
                ) as json_file:
                    data = json.load(json_file)
                    try:
                        alleged_arguments = (
                            data["text"]
                            .split("THE LAW")[1]
                            .split("FOR THESE REASONS, THE COURT UNANIMOUSLY")[0]
                            .lower()
                        )
                        # claims = list(set(re.findall("article\s(\d{1,2})\s", alleged_arguments)))
                        convention_claims = list(
                            set(
                                re.findall(
                                    "article\s(\d{1,2})\s.{0,15}convention",
                                    alleged_arguments,
                                )
                            )
                        )
                        # claims = [c for c in other_claims if int(c) <= 18]
                    except:
                        convention_claims = []

                    try:
                        alleged_arguments = (
                            data["text"]
                            .split("THE LAW")[1]
                            .split("FOR THESE REASONS, THE COURT UNANIMOUSLY")[0]
                            .lower()
                        )
                        # claims = list(set(re.findall("article\s(\d{1,2})\s", alleged_arguments)))
                        protocol_claims = list(
                            set(
                                re.findall(
                                    "article\s(\d{1,2})\s.{0,15}protocol.{0,15}(\d)",
                                    alleged_arguments,
                                )
                            )
                        )
                        protocol_claims = [
                            "P" + p[1] + "-" + p[0] for p in protocol_claims
                        ]
                    except:
                        protocol_claims = []

                    argument = get_arguments(data)
                    claims = list(set(convention_claims + protocol_claims))
                    data["claim"] = claims
                    data["arguments"] = argument

                directory = os.path.dirname(os.path.join("ECHR/Outcome/" + out, item))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                with open(os.path.join("ECHR/Outcome/" + out, item), "w") as out_file:
                    json.dump(data, out_file, indent=1)

                if len(claims) > 0:
                    all_facts.append(data["facts"])
                    all_claims.append(claims)
                    all_arguments.append(argument)
                    # print("claims", claims)
                    # print("outcomes", data["violated_articles"])
                    all_outcomes.append(data["violated_articles"])
                    case_id = str(data["case_no"])
                    all_ids.append(case_id)

        all_facts, all_claims, all_outcomes, all_ids, all_arguments = fix_claims(
            all_facts, all_claims, all_outcomes, all_ids, all_arguments
        )
        print(pretokenized_dir, len(all_facts))

        dataset_facts += all_facts
        dataset_claims += all_claims
        dataset_outcomes += all_outcomes
        dataset_arguments += all_arguments
        dataset_ids += all_ids

    split_dataset(
        pretokenized_dir,
        tokenizer,
        max_len,
        dataset_ids,
        dataset_facts,
        dataset_arguments,
        dataset_claims,
        dataset_outcomes,
    )


def get_stats(data):
    # Calculates statistics on the data
    data = np.array(data)
    stats = np.array(
        [0 for i in range(len(data[0]))]
    )  # len(data[0]) gets columns converted to a range 0..len, each set for 0
    cnt = 0  # counter
    for d in data:  # d represents a row in the dataset
        stats = stats + d
        if d.sum() > 0:
            cnt += 1

    return stats, cnt


def get_neg(claim_data, out_data):
    # Calculates the difference between claim data and outcome data
    cdata = np.array(claim_data)
    odata = np.array(out_data)
    c_stats, c_cnt = get_stats(claim_data)
    out_stats, out_cnt = get_stats(out_data)
    stats = c_stats - out_stats
    cnt = 0
    for c, o in zip(cdata, odata):
        n = c - o
        if n.sum() > 0:
            cnt += 1

    return stats, cnt


def data_stats(claims, outcomes, type):
    # Provides statistics on the data, including the number of claims, positives, and negatives

    c_stats, c_cnt = get_stats(claims)
    out_stats, out_cnt = get_stats(outcomes)
    neg_stats = c_stats - out_stats
    _, n_cnt = get_neg(claims, outcomes)

    print("-" * 40)
    print(f"{type:^9} | {c_cnt:^9} | {out_cnt:^9} | {n_cnt:^9}")

    return [c_stats, out_stats, neg_stats]


def split_dataset(
    pretokenized_dir,
    tokenizer,
    max_len,
    dataset_ids,
    dataset_facts,
    dataset_arguments,
    dataset_claims,
    dataset_outcomes,
):
    # splits dataset into training / validation / test

    train_ids, train_facts, train_arguments, train_claims, train_outcomes = (
        [],
        [],
        [],
        [],
        [],
    )
    val_ids, val_facts, val_arguments, val_claims, val_outcomes = [], [], [], [], []
    test_ids, test_facts, test_arguments, test_claims, test_outcomes = (
        [],
        [],
        [],
        [],
        [],
    )

    dataset_ids = [
        str(id) for id in dataset_ids
    ]  # done to ensure consistency in data types and handle cases where IDs are not strings

    r_s = 42
    X_ids, X_test_ids, y_outcome, y_test_outcome = train_test_split(
        dataset_ids, dataset_outcomes, test_size=0.10, random_state=r_s
    )
    X_train_ids, X_valid_ids, y_train_outcome, y_valid_outcome = train_test_split(
        X_ids, y_outcome, test_size=0.10, random_state=r_s
    )

    case_dic = dict(
        zip(
            dataset_ids,
            zip(dataset_facts, dataset_claims, dataset_outcomes, dataset_arguments),
        )
    )

    for id, value in tqdm(zip(case_dic.keys(), case_dic.values())):
        if id in X_train_ids:
            train_ids.append(id)
            train_facts.append(value[0])
            train_claims.append(value[1])
            train_outcomes.append(value[2])
            train_arguments.append(value[3])
        elif id in X_valid_ids:
            val_ids.append(id)
            val_facts.append(value[0])
            val_claims.append(value[1])
            val_outcomes.append(value[2])
            val_arguments.append(value[3])
        else:
            test_ids.append(id)
            test_facts.append(value[0])
            test_claims.append(value[1])
            test_outcomes.append(value[2])
            test_arguments.append(value[3])

    mlb = MultiLabelBinarizer()
    train_claims, train_outcomes = binarizer(train_claims, train_outcomes, mlb, True)
    test_claims, test_outcomes = binarizer(test_claims, test_outcomes, mlb)
    val_claims, val_outcomes = binarizer(val_claims, val_outcomes, mlb)

    print(f"{'split':^9} | {'claims':^9} | {'positives':^9} | {'negatives':^9}")
    training = data_stats(train_claims, train_outcomes, "train")
    validation = data_stats(val_claims, val_outcomes, "val")
    test = data_stats(test_claims, test_outcomes, "test")

    for i in [training, validation, test]:
        for j in i:
            print(j)

    print("Tokenizing data...")

    Path(pretokenized_dir).mkdir(parents=True, exist_ok=True)

    # train
    train_facts, train_masks = preprocessing_for_llama(
        train_facts, tokenizer, max=max_len
    )
    train_arguments, train_masks_arguments = preprocessing_for_llama(
        train_arguments, tokenizer, max=max_len
    )

    with open(pretokenized_dir + "/tokenized_train.pkl", "wb") as f:
        pickle.dump(
            [
                train_facts,
                train_masks,
                train_arguments,
                train_masks_arguments,
                train_ids,
                train_claims,
                train_outcomes,
                mlb,
            ],
            f,
            protocol=4,
        )

    # validation

    val_facts, val_masks = preprocessing_for_llama(val_facts, tokenizer, max=max_len)
    val_arguments, val_masks_arguments = preprocessing_for_llama(
        val_arguments, tokenizer, max=max_len
    )

    with open(pretokenized_dir + "/tokenized_dev.pkl", "wb") as f:
        pickle.dump(
            [
                val_facts,
                val_masks,
                val_arguments,
                val_masks_arguments,
                val_ids,
                val_claims,
                val_outcomes,
                mlb,
            ],
            f,
            protocol=4,
        )

    # test
    test_facts, test_masks = preprocessing_for_llama(test_facts, tokenizer, max=max_len)
    test_arguments, test_masks_arguments = preprocessing_for_llama(
        test_arguments, tokenizer, max=max_len
    )

    with open(pretokenized_dir + "/tokenized_test.pkl", "wb") as f:
        pickle.dump(
            [
                test_facts,
                test_masks,
                test_arguments,
                test_masks_arguments,
                test_ids,
                test_claims,
                test_outcomes,
                mlb,
            ],
            f,
            protocol=4,
        )

    return train_ids, train_facts, train_claims, train_outcomes


def binarizer(claims, outcomes, mlb, fit=False):
    # why is it needed
    if fit:
        claims = mlb.fit_transform(claims)
        outcomes = mlb.transform(outcomes)
    else:
        claims = mlb.transform(claims)
        outcomes = mlb.transform(outcomes)

    return claims, outcomes


def preprocessing_for_llama(data, tokenizer, max=512):
    # Prepares the text data for input to llama-based models by tokenizing and encoding the text

    """Perform required preprocessing steps for pretrained llama.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """

    # For every sentence...
    input_ids = []
    attention_masks = []

    for sent in tqdm(data):
        sent = " ".join(sent)
        sent = sent[
            :500000
        ]  # Speeds the process up for documents with a lot of precedent we would truncate anyway.
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=max,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,  # Return attention mask
            truncation=True,
        )

        # Add the outputs to the lists
        input_ids.append([encoded_sent.get("input_ids")])
        attention_masks.append([encoded_sent.get("attention_mask")])

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def text_preprocessing(text):
    # Preprocesses the text data by removing mentions and correcting errors

    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r"(@.*?)[\s]", " ", text)

    # Replace '&amp;' with '&'
    text = re.sub(r"&amp;", "&", text)

    # Remove trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def bert_preprocess():
    # Preprocesses the outcome data for BERT
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    get_data("ECHR/Outcome/bert", tokenizer, 512)

def llama_preprocess():
    # Preprocesses the outcome data for llama
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying alternative method...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=False)
        except Exception as e:
            print(f"Error loading tokenizer with use_fast=False: {e}")
            raise

    # Explicitly set padding token and strategy
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Print tokenizer info for debugging
    print("Tokenizer class:", tokenizer.__class__.__name__)
    print("Tokenizer vocab size:", len(tokenizer.vocab))
    print("Tokenizer pad token:", tokenizer.pad_token)
    print("Tokenizer pad token ID:", tokenizer.pad_token_id)
    print("Tokenizer eos token:", tokenizer.eos_token)
    print("Tokenizer eos token ID:", tokenizer.eos_token_id)

    get_data("ECHR/Outcome/llama", tokenizer, 512)


if __name__ == "__main__":
    llama_preprocess()
    # bert_preprocess()
