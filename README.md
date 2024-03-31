# Text Similarity Evaluation using TF-IDF and Word2Vec  

This repository contains a Python script for evaluating text similarity using TF-IDF (Term Frequency-Inverse Document Frequency) and Word2Vec techniques. The script computes similarity matrices, trains Word2Vec models, and evaluates the performance of the TF-IDF method.  

## Prerequisites  

Before running the script, ensure you have the following libraries installed:  

- NLTK 
- scikit-learn 
- Gensim 
- NumPy 
- Pandas 
- Matplotlib  

You can install these libraries via pip:  

```bash 
pip install nltk scikit-learn gensim numpy pandas matplotlib
```
or 
```bash
pip install -r requirements.txt
```

Additionally, NLTK resources such as the Brown corpus and tokenization models are required. You can download them by executing:
```python
import nltk 
nltk.download('brown') 
nltk.download('punkt')
```
## Usage

To use this script, follow these steps:

1. Ensure the prerequisites are met.
2. Execute the script by running the `VectorVsLexical.ipynb` function.
3. The script loads the Brown corpus, computes TF-IDF similarity, trains a Word2Vec model, and loads the SimLex-999 dataset for evaluation.
4. It performs transitivity analysis on the dataset.
5. The TF-IDF method is evaluated, and the results are saved as a JSON file and a graph.

## Functions Overview

* **Preprocessing**: Preprocesses the input corpus.
* **TF-IDF Similarity Calculation**: Computes TF-IDF similarity between documents.
* **Word2Vec Model Training**: Trains a Word2Vec model on the corpus.
* **Similar Word Retrieval with Word2Vec**: Retrieves top similar words using the trained Word2Vec model.
* **Evaluation**: Generates a dictionary format for evaluation and evaluates the performance using Pytrec_eval (TREC evaluation tools).
* **Result Saving**: Saves evaluation results to JSON and graph formats.
* **Golden Dataset Extraction**: Extracts relevant data from the SimLex-999 dataset.
* **Search Operation**: Performs a search operation using TF-IDF similarity scores.
* **Transitivity Analysis**: Analyzes transitivity in search results.

## Output

Evaluation results are saved as a JSON file containing NDCG scores and a graph visualizing the TF-IDF NDCG scores.
