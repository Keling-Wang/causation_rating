import torch
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
from scipy.stats import kendalltau


def compute_metrics(EvalPrediction: tuple) -> dict:
    """
    Compute the evaluation metrics for the given predictions.
    
    Args:
        EvalPrediction: Tuple containing logits and labels.

    Returns:
        Dictionary containing the evaluation metrics.
    """

    def off_by_k_accuracy(predictions: torch.tensor, true_labels: torch.tensor, k: int) -> float:
    
        # Calculate the absolute difference between predicted and true class indices
        diff = torch.abs(predictions - true_labels)
    
        # Count how many differences are less than k
        correct_within_k = (diff < k).sum().item()
    
        # Calculate the Off-by-k accuracy as a percentage
        ob_k_accuracy = (correct_within_k / len(true_labels)) * 100
    
        return ob_k_accuracy

    
    logits, labels = EvalPrediction
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    preds = torch.argmax(logits,dim = -1)
    true_labels = torch.argmax(labels, dim=1)

    # off-by-k accuracy
    off_by_1_acc = off_by_k_accuracy(preds, true_labels, 1)
    off_by_2_acc = off_by_k_accuracy(preds, true_labels, 2)
    
    true_labels = true_labels.tolist()
    
    # Mean Squared Error for Classification
    mse = mean_squared_error(true_labels, preds)

    f1 = f1_score(true_labels, preds, average='weighted')

    # Kendall's Tau
    tau = kendalltau(true_labels, preds).statistic


    return {
        "off_by_1_acc": off_by_1_acc,
        "off_by_2_acc": off_by_2_acc,
        "mse": mse,
        #"auc": auc,
        "f1": f1,
        "Kendall's Tau": tau,
    }