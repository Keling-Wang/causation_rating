from causation_rating.classifier import BERTClassifier
from causation_rating.custom_dataset import CausalDataset
from causation_rating.read_data import load_dataset, write_dataset
from causation_rating import constants
from causation_rating.oll_loss_trainer import OLLTrainer
from causation_rating.metrics import compute_metrics

import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments

class BERTClassifier_final(BERTClassifier):
    def __init__(self, model_path, num_labels=5, aug_factor=7, training_args=None, loss_err_power=3.0):
        super().__init__(model_path, num_labels, aug_factor, training_args, loss_err_power)

    def predict_final(self, pred_dataset):
        # Initialize model and tokenizer
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_path, num_labels=self.num_labels
        ) 
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model.dist_matrix = constants.DIST_MATRIX.to(self.model.device)
        # Make predictions
        self.trainer = OLLTrainer(err_power=self.loss_err_power,
                             model=self.model, 
                             args=TrainingArguments(output_dir="./tmp"),
                             tokenizer=self.tokenizer,
                             compute_metrics=compute_metrics)
        
        predictions = self.trainer.predict(pred_dataset)
        return predictions

def predict_fromPretrained(model_path, file_for_pred, write_csv = False, path_to_save = None):
    """
    Predict the labels of a list of texts using a pretrained model.
    Args:
        model_path (str): Path to the pretrained model.
        texts (list): List of texts to predict.
    Returns:
        list: List of predicted labels.
    """
    model = BERTClassifier_final(model_path=model_path)
    X_pred, _ = load_dataset(file_for_pred)
    pred_dataset = CausalDataset(X_pred, [0]*len(X_pred))

    predictions = model.predict_final(pred_dataset)
    pred_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    if write_csv:
        write_dataset(X_pred, pred_labels, path_to_save)
    
    return pd.DataFrame({'text': X_pred, 'label': pred_labels - 1})
