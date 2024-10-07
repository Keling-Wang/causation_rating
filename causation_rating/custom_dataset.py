from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import re

from causation_rating import constants


class CausalDataset(Dataset):
    """
    Custom dataset for text classification with BERT.

    Args:
        texts: List of input text samples.
        labels: List of integer labels (0 to num_classes-1).
        tokenizer: Pre-trained tokenizer.
        max_length: Maximum sequence length.
        num_classes: Number of classes for one-hot encoding (default 5).
        adjust: boolean. Whether to adjust attention mask based on important words.
        adjust_factor: float. Factor to adjust attention mask for important words.

    Returns:
        A PyTorch dataset for text classification.
    """

    def __init__(self, texts: list, labels: list, 
                 tokenizer=None, max_length=128, num_classes=5, 
                 adjust=True, ignore_ref = True):
        """
        Initialize the dataset with texts, labels, and the tokenizer.
        
        Args:
            texts: List of input text samples.
            labels: List of integer labels (0 to num_classes-1).
            tokenizer: Pre-trained tokenizer.
            max_length: Maximum sequence length.
            num_classes: Number of classes for one-hot encoding (default 5).
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained(constants.MODEL_PATH_PRE)
        self.max_length = max_length
        self.num_classes = num_classes
        self.adjust = adjust
        self.ignore_pattern = re.compile(r'[\d()\[\]\-–]+') if ignore_ref else re.compile(r'')
        # ignore_pattern ignore numbers, brackets, and dashes and set their attention mask to 0.
    def adjust_attention_mask(self, tokens, adjust):
        
        attention_mask = torch.ones(self.max_length)  # Default attention mask
        
        # Adjust based on important words and their direct derivations
        # update 01 Oct: Now only adjust for ignored tokens.
        if adjust:
            for idx, token in enumerate(tokens):
                if self.ignore_pattern.search(token) and idx < self.max_length:
                    attention_mask[idx] = 0
        return attention_mask

    def __len__(self):
        """
        Return the total number of text samples.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get the tokenized input, attention mask, and one-hot encoded label for the given index.
        
        Args:
            idx: Index of the sample.
        
        Returns:
            A dictionary containing:
                - input_ids: Tensor of input token IDs.
                - token_type_ids: Tensor of token type IDs.
                - attention_mask: Adjusted attention mask tensor.
                - label: One-hot encoded label tensor.
        """
        text = self.texts[idx]
        label = self.labels[idx]  # Assume this is an integer label (0 to 4)

        if isinstance(label, str):
            try:
                label = int(label)
            except ValueError:
                raise ValueError(f"Label at index {idx} is not a valid integer: '{label}'")
        
        if not isinstance(label, int) or label < 0 or label >= self.num_classes:
            raise ValueError(f"Invalid label at index {idx}: {label}. Labels must be integers from 0 to {self.num_classes-1}")
        
        
        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        encoding = {key:val.squeeze(0) for key, val in encoding.items()}
        
        # Convert the tokenized text into individual components
        tokens = self.tokenizer.tokenize(text)
        attention_mask = self.adjust_attention_mask(tokens, self.adjust).squeeze(0)
        
        # One-hot encode the label for 5 classes
        label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes).float().squeeze(0)

        return {
            'input_ids': encoding["input_ids"],
            'token_type_ids': encoding["token_type_ids"],
            'attention_mask': attention_mask,
            'labels': label_one_hot,  # One-hot encoded label
        }