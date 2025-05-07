# Enhancing Discriminative Representation in Similar Relation Clusters for Few-Shot Continual Relation Extraction (NAACL 2025)

This repository contains the implementation for the paper "Enhancing Discriminative Representation in Similar Relation Clusters for Few-Shot Continual Relation Extraction," NAACL 2025.

## General Requirements

* **Python:** Version 3.8 or higher.
* **Common Libraries:** These are required for both BERT and LLM2Vec setups.
    ```bash
    pip install transformers==4.40.0 torch==2.3.0 scikit-learn==1.4.2 nltk==3.8.1 retry==0.9.2
    ```

## BERT Implementation

### Setup

1.  Ensure the libraries listed under "General Requirements" are installed.

### Running Experiments

1.  Navigate to the BERT experiment directory:
    ```bash
    cd Bert/bash
    ```
2.  Run the experiment scripts:
    * For TACRED (5-shot):
        ```bash
        bash tacred_5shot.sh
        ```
    * For FewRel (5-shot):
        ```bash
        bash fewrel_5shot.sh
        ```

## LLM2Vec Implementation

### Setup

1.  **Install Common Libraries:** Make sure the libraries listed under "General Requirements" are installed.
2.  **Install LLM2Vec Specific Libraries:**
    ```bash
    pip install llm2vec==0.2.2
    pip install flash-attn --no-build-isolation
    ```
3.  **Hugging Face Login:** You'll need to log in to Hugging Face to download certain models. Replace `your_huggingface_token_to_access_model` with your actual Hugging Face access token.
    ```bash
    huggingface-cli login --token your_huggingface_token_to_access_model
    ```
4.  **OpenAI API Key (for CPL model):** If you intend to run experiments using the CPL model, you must provide your OpenAI API key. Add this key to the `config.ini` file.

### Running Experiments

1.  Change to the LLM experiment directory:
    ```bash
    cd LLM/bash
    ```
2.  Execute the experiment scripts:
    * For TACRED (5-shot):
        ```bash
        bash tacred_5shot.sh
        ```
    * For FewRel (5-shot):
        ```bash
        bash fewrel_5shot.sh
        ```

### Important Notes for LLM2Vec Experiments

* **CPL Model Precision (Llama2 & Mistral):** To ensure a fair comparison with the results reported in [https://arxiv.org/abs/2410.00334](https://arxiv.org/abs/2410.00334), experiments involving the CPL model with Llama2 and Mistral are conducted using `float32` precision.
* **Default Precision:** All other LLM2Vec experimental setups utilize `bf16` precision.
* **Troubleshooting Model Stalls:** In some LLM experiments, the model may occasionally get stuck on the first task. If you encounter this issue, consider the following troubleshooting steps:
    * Adjust the learning rate. (1e-5 or 1e-4)
    * Switch the precision from `bf16` to `float32`.