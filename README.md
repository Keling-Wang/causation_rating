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

This package contains all codes used in the training, evaluation, and prediction process of [`bert-causation-rating-drt`](https://huggingface.co/kelingwang/bert-causation-rating-drt) model. 

This model is fine-tuned [biobert-base-cased-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) model on a small set of manually annotated texts with causation labels. It is tasked with classifying a sentence into different levels of strength of causation expressed in this sentence.
Before tuning, the `biobert-base-cased-v1.2` model is fine-tuned on a dataset containing causation labels from a published paper. This model starts from pre-trained [`kelingwang/bert-causation-rating-pubmed`](https://huggingface.co/kelingwang/bert-causation-rating-pubmed). For more information please view the link.

The sentences in the dataset were rated independently by two researchers.

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

- Make predictions: based on pretrained `drt` models, `causation_rating.predict_from_pretrained` allows one directly uses the two models maintained on `🤗 Hugging Face` to make predictions based on new sentences.
- Reproduce training process: `causation_rating.bayesian_optimization` module allows one reproduces Bayesian hyperparameter tuning process to find best parameters, based on datasets maintained on `🤗 Hugging Face`. `causation_rating.model_finaltraining` allows one reproduces final training process on the entire dataset after hyperparameter tuning.

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
                            MODEL_PATH_FINALUSE = 'kelingwang/bert-causation-rating-drt',
                            )
# ... [Your code here]
```
## Constants defined and used in this code
Here you can find default values and descriptions for all the constants defined in the package. See also `constants.py`.

#### Basic raining arguments
 - `SEED`: seed used across all places where a seed for random initiation/splitting is required. Default to `114514`.
 - `DEVICE`: device used for model prediction and training. Default to `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
 - `EPOCH`: epoches for model training. Default to `8`. 
 - `BATCH_SIZE`: batch size for model training. Default to `128`.
 - `DIST_MATRIX` distance matrix used in the ordinal log loss function. Default to:
```python
torch.tensor([[0, 2, 3.5, 4.5, 5.5],
              [2, 0, 1.5, 2.5, 3.5],
              [3.5, 1.5, 0, 1, 2],
              [4.5, 2.5, 1, 0, 1],
              [5.5, 3.5, 2, 1, 0]])
```
Also see [this paper](https://aclanthology.org/2022.coling-1.407/) for details.

#### Hyperparameter tuning searching space
 - `LR_SEARCH`: a range of learning rate used in Bayesian hyperparameter optimization. Default to `[5e-6, 8e-5]`. 
 - `WEIGHT_DECAY_SEARCH`: a continuous range of weight decay used in Bayesian hyperparameter optimization. Default to `[0.035, 0.15]`.
 - `WARMUP_RATIO_SEARCH`: a continuous range of warm-up ratio used in Bayesian hyperparameter optimization. Default to `[0.30, 0.50]`.
 - `LR_SCHEDULER_POWER_SEARCH`: a continuous range of the power to the polynomial learning rate scheduler used in Bayesian hyperparameter optimization. Default to `[1.0, 3.0]`.
 - `OLL_POWER_SEARCH`: a set of vaules of the power to the error measures used in the ordinal log loss function. Default to `[2.0, 2.25, 2.5, 2.75, 3.0]`.

#### Paths
 - `MODEL_PATH_AUG`: path to the model used in data augmentation. Default to `'dmis-lab/biobert-base-cased-v1.2'`. This is a `🤗 Hugging Face` path.
 - `MODEL_PATH_PRE`: path to the pretrained model to be fine-tuned. Default to `'kelingwang/bert-causation-rating-pubmed'`. This is a `🤗 Hugging Face` path.
 - `MODEL_PATH_FINALUSE`: path to the model for final prediction. Default to `'kelingwang/bert-causation-rating-drt'`. This is a `🤗 Hugging Face` path.

 - `DATASET_PATH`： dataset used for training. Default to `'https://huggingface.co/datasets/kelingwang/causation_strength_rating/resolve/main/rating_drt.csv'`.
 - `IMPORTANT_WORD_PATH`: word list indicating important causal linking words. Default to `'https://huggingface.co/datasets/kelingwang/causation_strength_rating/resolve/main/linkingwords_complete.csv'`. 
 - `OUTPUT_PATH`: path for outputs. Default to `'./'`.
 - `TEST_DATA_NAME`: user-defined name for the test dataset that will be held out. Default to `'test_drt.csv'`.
