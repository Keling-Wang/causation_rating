# causation_rating

🤗
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 
[![Release Version](https://img.shields.io/github/v/release/username/repository)](https://github.com/username/repository/releases)
[![Python Version](https://img.shields.io/badge/python-%3E%3D3.8-blue)](https://www.python.org/)
[![Transformers Version](https://img.shields.io/badge/transformers-4.45.1-orange.svg)](https://huggingface.co/docs/transformers/)
[![scikit-learn Version](https://img.shields.io/badge/scikit--learn-1.2.2-yellow)](https://scikit-learn.org/)
[![Optuna Version](https://img.shields.io/badge/optuna-4.0.0-blue)](https://optuna.org/)
[![nlpaug Version](https://img.shields.io/badge/nlpaug-1.1.11-purple)](https://github.com/makcedward/nlpaug)


## Description

This package contains all codes used in the training, evaluation, and prediction process of [`bert-causation-rating-dr1`](https://huggingface.co/kelingwang/bert-causation-rating-dr1) and [`bert-causation-rating-dr2`](https://huggingface.co/kelingwang/bert-causation-rating-dr2) models. 

These models are fine-tuned [biobert-base-cased-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) models on two small sets of manually annotated texts with causation labels. They are tasked with classifying a sentence into different levels of strength of causation expressed in this sentence.
Before tuning, the `biobert-base-cased-v1.2` model is fine-tuned on a dataset containing causation labels from a published paper. This model starts from pre-trained [`kelingwang/bert-causation-rating-pubmed`](https://huggingface.co/kelingwang/bert-causation-rating-pubmed). For more information please view the link.

The sentences in the dataset were rated independently by two researchers. The `dr1` version is tuned on the set of sentences with labels rated by Rater 1, and
the `dr2` version is tuned on the set of sentences with labels rated by Rater 2 and 3.

### Intended use and limitations of the model

This model is primarily used to rate for the strength of expressed causation in a sentence extracted from a clinical guideline in the field of diabetes mellitus management. 
This model predicts strength of causation (SoC) labels based on the text inputs as: 
 * -1: No correlation or variable relationships mentioned in the sentence.
 * 0: There is correlational relationships but not causation in the sentence.
 * 1: The sentence expresses weak causation.
 * 2: The sentence expresses moderate causation.
 * 3: The sentence expresses strong causation.
*NOTE:* The model output is five one-hot logits and will be 0-index based, and the labels will be 0 to 4. It is good to use [this `python` module](https://github.com/Keling-Wang/causation_rating/blob/main/tests/prediction_from_pretrained.py) if one wants to make predictions.

## Features

- Make predictions: based on pretrained `dr1` and `dr2` models, `causation_rating.predict_from_pretrained` allows one directly uses the two models maintained on 🤗 Hugging Face to make predictions based on new sentences.
- Reproduce training process: `causation_rating.bayesian_optimization` module allows one reproduces Bayesian hyperparameter tuning process to find best parameters, based on datasets maintained on 🤗 Hugging Face. `causation_rating.model_finaltraining` allows one reproduces final training process on the entire dataset after hyperparameter tuning.

## Installation

You can install the package via pip:

```bash
python -m pip install git+https://github.com/Keling-Wang/causation_rating.git
```
Alternatively, you can clone this repository and install manually:

```bash
git clone https://github.com/Keling-Wang/causation_rating.git
cd causation_rating
pip install -e .
```

## Usage

You can play with the code after importing this package and redefining some constants to your need.

```python
import causation_rating
import torch

# You always need to set constants before using the model. For details, see the `constants.py` file and `README.md`.
causation_rating.set_config(DEVICE = torch.device("cuda"), 
                            #BATCH_SIZE = 128,
                            MODEL_PATH_FINALUSE = 'kelingwang/bert-causation-rating-dr2',
                            )
# ... [Your code here]
```
