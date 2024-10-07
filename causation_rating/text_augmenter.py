import nlpaug.augmenter.word as naw
from nlpaug.util import Doc, Action
from nlpaug.augmenter.word import WordAugmenter
import nlpaug.model.lang_models as nml
import string
from collections import Counter
import numpy as np
from causation_rating.custom_dataset import CausalDataset
from causation_rating import constants
from get_important_words import get_important_words

class ProtectedWordContextualWordEmbsAug(naw.ContextualWordEmbsAug):
    # We modify the original ContextualWordEmbsAug class for data augmentation to allow us protect ``important word list'' from being augmented.
    def __init__(self, protected_words, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protected_words = set(protected_words)
    
    def substitute(self, data, protected_words=None):
        protected_words = self.protected_words

        if not data:
            return data
        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data
            all_data = [data]

        # If length of input is larger than max allowed input, only augment heading part
        split_results = [] # head_text, tail_text, head_tokens, tail_tokens
        reserved_stopwords = []
        for d in all_data:
            split_result, reserved_stopword = self.split_text(d)
            split_results.append(split_result)
            reserved_stopwords.append(reserved_stopword)

        change_seq = 0
        # Pick target word for augmentation
        for i, (split_result, reserved_stopword_tokens) in enumerate(zip(split_results, reserved_stopwords)):
            head_text, tail_text, head_tokens, tail_tokens = split_result            

            if self.model_type in ['xlnet', 'roberta', 'bart']:
                # xlent and roberta tokens include prefix (e.g. ▁ or Ġ')
                cleaned_head_tokens = [t.replace(self.model.get_subword_prefix(), '') for t in head_tokens]
            else:
                cleaned_head_tokens = head_tokens

            head_doc = Doc(head_text, head_tokens)
            aug_idxes = self._get_aug_idxes(head_tokens)
        
            # Filter out protected words
            aug_idxes = [idx for idx in aug_idxes if cleaned_head_tokens[idx].lower() not in protected_words]
        
            aug_idxes.sort(reverse=True)

            if reserved_stopword_tokens:
                head_doc, change_seq = self.substitute_back_reserved_stopwords(
                    head_doc, reserved_stopword_tokens, change_seq)
            head_tokens = head_doc.get_augmented_tokens()
        
            split_results[i] += (cleaned_head_tokens, head_doc, aug_idxes, )

        # Pad aug_idxes
        max_aug_size = max([len(split_result[6]) for split_result in split_results])
        for split_result in split_results:
            aug_idxes = split_result[6]
            for _ in range(max_aug_size - len(aug_idxes)):
                aug_idxes.append(-1)

        token_placeholder = self.model.get_mask_token()
        if self.model_type in ['xlnet', 'roberta', 'bart']:
            token_placeholder = self.model.get_subword_prefix() + token_placeholder  # Adding prefix for

        # Augment same index of aug by batch
        for i in range(max_aug_size):
            original_tokens = []
            masked_texts = []
            aug_input_poses = [] # store which input augmented. No record if padding

            change_seq += 1
            for j, split_result in enumerate(split_results):
                head_doc, aug_idx = split_result[5], split_result[6][i]

                # -1 if it is padding 
                if aug_idx == -1:
                    continue

                original_tokens.append(head_doc.get_token(aug_idx).get_latest_token().token)

                head_doc.add_change_log(aug_idx, new_token=token_placeholder, action=Action.SUBSTITUTE,
                    change_seq=self.parent_change_seq+change_seq)

                # remove continuous sub-word
                to_remove_idxes = []
                for k in range(aug_idx+1, head_doc.size()):
                    subword_token = head_doc.get_token(k).orig_token.token
                    if subword_token in string.punctuation:
                        break
                    if self.model_type in ['bert', 'electra'] and self.model.get_subword_prefix() in subword_token:
                        to_remove_idxes.append(k)
                    elif self.model_type in ['xlnet', 'roberta', 'bart'] and self.model.get_subword_prefix() not in subword_token:
                        to_remove_idxes.append(k)
                    else:
                        break
                for k in reversed(to_remove_idxes):
                    head_doc.add_change_log(k, new_token='', action=Action.SUBSTITUTE,
                        change_seq=self.parent_change_seq+change_seq)

                aug_input_poses.append(j)

                # some tokenizers handle special charas (e.g. don't can merge after decode)
                if self.model_type in ['bert', 'electra']:
                    ids = self.model.get_tokenizer().convert_tokens_to_ids(head_doc.get_augmented_tokens())
                    masked_text = self.model.get_tokenizer().decode(ids).strip()
                elif self.model_type in ['xlnet', 'roberta', 'bart']:
                    masked_text = self.model.get_tokenizer().convert_tokens_to_string(head_doc.get_augmented_tokens()).strip()

                masked_texts.append(masked_text)

            if not len(masked_texts):
                continue

            outputs = self.model.predict(masked_texts, target_words=original_tokens, n=2)

            # Update doc
            for original_token, aug_input_pos, output, masked_text in zip(original_tokens, aug_input_poses, outputs, masked_texts):
                split_result = split_results[aug_input_pos]
                head_doc = split_result[5]
                aug_idx = split_result[6][i] # augment position in text

                # TODO: Alternative method better than dropout
                candidate = ''
                if len(output) == 0:
                    # TODO: no result?
                    pass
                elif len(output) == 1:
                    candidate = output[0]
                elif len(output) > 1:
                    candidate = self.sample(output, 1)[0]

                # Fallback to original token if no candidate is appropriate or if it's a protected word
                if candidate == '' or candidate.lower() in protected_words:
                    candidate = original_token

                head_doc.update_change_log(aug_idx, token=candidate, action=Action.SUBSTITUTE,
                    change_seq=self.parent_change_seq+change_seq)

                # Early stop if number of token exceed max number
                if head_doc.size() > self.max_num_token:
                    for j in range(i+1, max_aug_size):
                        split_results[aug_input_pos][6][j] = -1

        augmented_texts = []
        for split_result in split_results:
            tail_text, head_doc = split_result[1], split_result[5]

            head_tokens = head_doc.get_augmented_tokens()

            ids = self.model.get_tokenizer().convert_tokens_to_ids(head_tokens)
            augmented_text = self.model.get_tokenizer().decode(ids)

            if tail_text is not None:
                augmented_text += ' ' + tail_text
            augmented_texts.append(augmented_text)

        if isinstance(data, list):
            return augmented_texts
        else:
            return augmented_texts[0]


# Data augmentation function for training set only 
def augment_text(Dataset, aug_factor=7, aug_p=0.15, 
                 protected_words = get_important_words(constants.IMPORTANT_WORD_PATH), model_path=constants.MODEL_PATH_AUG):
    """
    Augment the input text using synonym replacement. Minor class imbalance is handled by oversampling.
    
    Args:
        Dataset: CausalDataset object.
        aug_factor: Total number of augmented samples to generate.
    
    Returns:
        Augmented text dataset.
    """

    aug = ProtectedWordContextualWordEmbsAug(protected_words = protected_words, 
                                             model_path=model_path, action="substitute", 
                                             aug_p = aug_p,aug_min = 3, top_k = 50, 
                                             device='cuda', batch_size=128)
    augmented_texts = []
    augmented_labels = []
    
    # Count the occurrences of each label
    label_counts = Counter(Dataset.labels)
    max_count = max(label_counts.values())

    for idx in range(len(Dataset)):
        text = Dataset.texts[idx]
        label = Dataset.labels[idx]

        # Calculate the number of augmentations needed for this sample
        augmentations_needed = max(1, int(aug_factor * (max_count / label_counts[label])))

        # Augment the text
        for _ in range(augmentations_needed):
            augmented_text = aug.augment(text)[0]
            augmented_texts.append(augmented_text)
            augmented_labels.append(label)

        # Always include the original text
        augmented_texts.append(text)
        augmented_labels.append(label)

    # Trim the dataset to ensure the total size is (aug_factor + 1) * original size
    target_size = (aug_factor + 1) * len(Dataset)
    if len(augmented_texts) > target_size:
        indices = np.random.choice(len(augmented_texts), target_size, replace=False)
        augmented_texts = [augmented_texts[i] for i in indices]
        augmented_labels = [augmented_labels[i] for i in indices]

    return CausalDataset(augmented_texts, augmented_labels, Dataset.tokenizer)