import torch
from transformers import Trainer


class OLLTrainer(Trainer):

    def __init__(self, err_power, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.err_power = err_power

    def compute_loss(self, model: torch.nn.Module, inputs: dict, return_outputs=False):
        """
        Compute the loss for the given inputs:
        - Calculate the distance matrix between all classes.
        - Compute the probabilities for both sets of logits.
        - Compute the distances between the true labels and all classes.
        - Calculate the errors for both labels.
        - Return the mean of both errors as the final loss.

        Args:

            model (:obj:`torch.nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
            err_power (:obj:`float`, `optional`, defaults to 1.5):
                The power to which the distances are raised.
            return_outputs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return the outputs of the model.

        """
        err_power = self.err_power
        dist_matrix = model.dist_matrix
        labels = inputs["labels"]

        # Forward pass
        outputs = model(**inputs)

        # Compute probabilities for both sets of logits
        probas = torch.softmax(outputs.logits, dim=1)

        
        num_classes = outputs.logits.size(-1)  # Assuming both logits have the same number of classes

        true_labels = torch.argmax(labels, dim=1).tolist()

        # Create label IDs for all classes
        label_ids = [[k for k in range(num_classes)] for _ in range(len(labels))]

        # Compute distances
        distances = [[float(dist_matrix[true_labels[j]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]

        distances_tensor = torch.tensor(distances, device=outputs.logits.device, requires_grad=True)

        # Calculate errors for both labels
        err = -torch.log(1 - probas) * distances_tensor ** err_power

        # Final loss is the mean of both errors
        loss = torch.sum(err, axis=1).mean()

        return (loss, outputs) if return_outputs else loss