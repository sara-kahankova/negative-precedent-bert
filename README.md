# Thesis Project: Negative Precedent Prediction

Welcome to the repository for the Negative Precedent Prediction research project. 

## Introduction of the project

This project explores the use of Large Language Models ("LLMs") for predicting legal outcomes, focusing on cases from the European Court of Human Rights. Building on previous work by Valvoda et al. (2023), the study aims to improve the accuracy of negative precedent prediction, which has lagged behind positive precedent prediction. The research employs various LLM architectures, including BERT and Llama 3, to investigate whether increasing model size can enhance prediction performance.

## Get Started

Follow these steps to set up your environment and get started with the ECHR Outcome Corpus.

### Step 1: Install Dependencies

First, ensure you have Python installed on your machine. Then, install the necessary dependencies from the `requirements.txt` file. Open your terminal and run:

```bash
pip install -r requirements.txt
```

### Step 2: Set Up the Directory Structure
Create a new directory named ECHR and copy the Outcome corpus files into a sub-directory named Outcome.

``` bash
mkdir -p ECHR/Outcome
```

### Step 3: Download the Dataset

Navigate to the ECHR/Outcome directory and download the Outcome corpus dataset using the link provided:

Outcome corpus

If you are operating from the command line, you might need to install gdown and unzip packages to download and extract the dataset:

```bash
pip install gdown
gdown id=1znbSf0vLJD-CxqpyzslxFw-vEe4qXOxw
unzip outcome_corpus.zip -d ECHR/Outcome
```
### Step 4: Preprocess the Data
You can now run the preprocess_data.py script to create the tokenized files. If you want to run llama preprocessing, you need to first uncomment llama_preprocess() at the end of the file.

```bash
python preprocess_data.py
```
If you are using the Llama model, make sure to authenticate with Huggingface-cli to access the Llama model. Follow these instructions to authenticate:

```bash
huggingface-cli login
```
After preprocessing, you should see new bert and/or llama directories in ECHR/Outcome.

### Step 5: Train Your Model
Finally, run the appropriate training script based on the model you wish to train. For example:

``` bash
python train_bert.py
```
or

``` bash
python train_llama.py
```

## Conclusion of the project

The study found that while larger models like Llama 3 showed modest improvements in negative precedent prediction, the gains were less significant than anticipated. This suggests that simply increasing model size may not be sufficient to bridge the gap between positive and negative precedent prediction in legal contexts. Furthermore, the MonoBERT experiment demonstrated that a simplified architecture could achieve comparable results to the original parallel BERT implementation, with only a slight decrease in the F1 score for negative prediction. The research highlights the complexity of legal prediction tasks and the need for further investigation into factors such as dataset quality, task formulation, and the inherent capabilities of LLMs in legal reasoning.

MonoLlama ~ copper-cloud
MonoBERT ~ glorious-fog
BERT Replica ~ lucky-leaf

<img width="475" alt="image" src="https://github.com/user-attachments/assets/173697e2-4ddf-4948-a75b-5e5fa1f9c95e">

