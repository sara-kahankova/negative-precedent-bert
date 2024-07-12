# Research Project: Setting Up the ECHR Outcome Corpus

Welcome to the repository for the ECHR Outcome Corpus research project. This guide will walk you through the steps needed to set up the code and start working with the dataset.

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

###Step 3: Download the Dataset

Navigate to the ECHR/Outcome directory and download the Outcome corpus dataset using the link provided:

Outcome corpus

If you are operating from the command line, you might need to install gdown and unzip packages to download and extract the dataset:

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1znbSf0vLJD-CxqpyzslxFw-vEe4qXOxw
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
